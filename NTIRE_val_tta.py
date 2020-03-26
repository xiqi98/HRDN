import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision import utils as vutils
import time
import os
import sys
from utils.util import *
from model.hrnet_se import HighResolutionLayer, HighResolutionDehazingNet
import setproctitle
from PIL import Image
import ttach as tta

parser = argparse.ArgumentParser(description='Settings')
parser.add_argument('--val_batch_size', type=int, default=1, help='Validation batch size')
parser.add_argument('--num_workers', type=int, default=0, help='# of workers')
parser.add_argument('--input_root', type=str, default='', help='Input root directory')
parser.add_argument('--output_root', type=str, default='', help='Output root directory')
parser.add_argument('--pretrained_model', type=str, default='', help='Path to pretrained model')
opt = parser.parse_args()
print(opt)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

data_transforms = {
    'val': transforms.Compose([
        transforms.ToTensor()
    ])
}

tta_transform = tta.Compose(
    [
        tta.HorizontalFlip(),
        tta.VerticalFlip(),
        tta.Rotate90(angles=[0, 90, 180, 270]),
        # tta.Scale(scales=[0.5, 0.7, 1], interpolation="nearest"),
    ]
)
assert len(tta_transform) == 2 * 2 * 4 * 1  # all combinations for aug parameters

val_set = ValSet(
    opt.input_root,
    transform=data_transforms['val'],
    normalize=True
)
print('Val Set Length: {}'.format(len(val_set)))

dataloaders = {
    'val': torch.utils.data.DataLoader(
        val_set, 
        batch_size=opt.val_batch_size, 
        shuffle=False, 
        num_workers=opt.num_workers
    )
}

layers = []

layers.append(HighResolutionLayer({
    'num_modules': 1,
    'num_branches': 2,
    'block': 'BASIC',
    'num_blocks': [4, 4],
    'num_channels': [32, 64],
    'multi_scale_output': True
}))

layers.append(HighResolutionLayer({
    'num_modules': 1,
    'num_branches': 3,
    'block': 'BASIC',
    'num_blocks': [4, 4, 4],
    'num_channels': [32, 64, 128],
    'multi_scale_output': True
}))

layers.append(HighResolutionLayer({
    'num_modules': 1,
    'num_branches': 4,
    'block': 'BASIC',
    'num_blocks': [4, 4, 4, 4],
    'num_channels': [32, 64, 128, 256],
    'multi_scale_output': True
}))

model = HighResolutionDehazingNet(layers)

print(model)

num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('Number of trainable parameters: {}'.format(num_trainable_params))

if opt.pretrained_model:
    load_checkpoint(opt.pretrained_model, model)

model = model.to(device)

since = time.time()

for phase in ['val']:
    model.eval()

    for names, inputs in tqdm(dataloaders[phase]):
        inputs = inputs.to(device)

        model.zero_grad()

        with torch.set_grad_enabled(phase == 'train'):
            outputs = torch.zeros([len(tta_transform), inputs.size(1), inputs.size(2), inputs.size(3)])
            for idx, augmenter in enumerate(tta_transform):
                augmented_image = augmenter.augment_image(inputs)
                model_output = model(augmented_image)
                deaugmented_output = augmenter.deaugment_mask(model_output)
                outputs[idx, :, :, :] = deaugmented_output
            output = torch.mean(outputs, 0)
            output.clamp_(0, 1)
            vutils.save_image(output, os.path.join(opt.output_root, names[0]+'.png'))

duration = time.time() - since

print(duration / 5)