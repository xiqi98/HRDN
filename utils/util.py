import torch
import os
from PIL import Image
import numpy as np
import random
from .transforms import *
from torchvision import transforms
import matplotlib.pyplot as plt
import pytorch_msssim
from pytorch_msssim import ssim
from tqdm import tqdm
from torchvision import utils as vutils
import PIL


def PSNR(outputs, targets):
    se = (outputs - targets) ** 2
    mse = torch.mean(se.view(se.size(0), -1), 1)
    psnr = 10.0 * torch.log10(1.0 / mse)
    return torch.sum(psnr)


def SSIM(outputs, targets):
    ssim_val = ssim(outputs, targets, data_range=1.0, size_average=False)
    return torch.sum(ssim_val)


def SSIMLoss():
    return pytorch_msssim.SSIM(win_size=11, win_sigma=1.5, data_range=1.0, size_average=True, channel=3)


def rgb_loader(path):
    return Image.open(path).convert('RGB')


def save_checkpoint(checkpoint_path, model, description=''):
    state = { 'model': model.state_dict(), 'description': description }
    torch.save(state, checkpoint_path)
    print('Model saved to {}'.format(checkpoint_path))


def load_checkpoint(checkpoint_path, model):
    state = torch.load(checkpoint_path)
    model.load_state_dict(state['model'])
    print('Model loaded from {}'.format(checkpoint_path))
    print(state['description'])


def lr_annealing(optimizer, init_lr, anneal_epochs):
    lrd = init_lr / anneal_epochs
    
    old_lr = optimizer.param_groups[0]['lr']
    lr = max(old_lr - lrd, 0)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class TrainSet(torch.utils.data.Dataset):
    def __init__(self, input_root, target_root, pre_transform=None, transform=None, normalize=True, phase='train'):
        super(TrainSet, self).__init__()

        self.pre_transform = pre_transform

        self.transform = transform

        self.normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) if normalize else None

        self.files = []

        for file in os.listdir(input_root):
            if file.endswith('.png') or file.endswith('.jpg'):
                input_path = os.path.join(input_root, file)
                target_path = os.path.join(target_root, file)
                self.files.append({ "name": file[:-4], "input": input_path, "target": target_path })

        # self.epoch = 0

        # self.diffusal = Diffusal()

        self.phase = phase
 
    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        file = self.files[index]
 
        foggy_img = rgb_loader(file["input"])
        clear_img = rgb_loader(file["target"])

        # if self.phase == 'train':
            # if random.random() < 0.5:
            #     foggy_img = self.diffusal(foggy_img, clear_img, self.epoch)

        if self.pre_transform is not None:
            foggy_img = self.pre_transform(foggy_img)

        if self.transform is not None:
            seed = np.random.randint(2147483647)

            random.seed(seed)
            torch.manual_seed(seed)
            foggy = self.transform(foggy_img)

            random.seed(seed)
            torch.manual_seed(seed)
            clear = self.transform(clear_img)

        if self.normalize is not None:
            foggy = self.normalize(foggy)
    
        return file["name"], foggy, clear

    def set_epoch(self, epoch):
        self.epoch = epoch


class ValSet(torch.utils.data.Dataset):
    def __init__(self, input_root, transform=None, normalize=True):
        super(ValSet, self).__init__()

        self.transform = transform

        self.normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) if normalize else None

        self.files = []
        
        for file in os.listdir(input_root):
            if file.endswith('.png'):
                input_path = os.path.join(input_root, file)
                self.files.append({ "name": file[:-4], "input": input_path })
 
    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        file = self.files[index]
 
        foggy_img = rgb_loader(file["input"])

        if self.transform is not None:
            foggy = self.transform(foggy_img)

        if self.normalize is not None:
            foggy = self.normalize(foggy)
    
        return file["name"], foggy


class ValResizeSet(torch.utils.data.Dataset):
    def __init__(self, input_root, transform=None, normalize=True):
        super(ValResizeSet, self).__init__()

        self.transform = transform

        self.normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) if normalize else None

        self.files = []
        
        for file in os.listdir(input_root):
            if file.endswith('.png'):
                input_path = os.path.join(input_root, file)
                self.files.append({ "name": file[:-4], "input": input_path })
 
    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        file = self.files[index]
 
        foggy_img = rgb_loader(file["input"])

        if self.transform is not None:
            foggy = self.transform(foggy_img)

        if self.normalize is not None:
            foggy = self.normalize(foggy)
    
        return file["name"], foggy, np.array(foggy_img.size)


class OTS(torch.utils.data.Dataset):
    from PIL import ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    def __init__(self, input_root, target_root, pre_transform=None, transform=None, normalize=True):
        super(OTS, self).__init__()
        self.pre_transform = pre_transform
        self.transform = transform
        self.normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) if normalize else None

        self.files = []
        for file in os.listdir(input_root):
            if file.endswith('.jpg'):
                input_path = os.path.join(input_root, file)
                target_path = os.path.join(target_root, file[:4]+'.jpg')
                self.files.append({ "name": file[:-4], "input": input_path, "target": target_path })
 
    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        file = self.files[index]
 
        foggy_img = rgb_loader(file["input"])
        clear_img = rgb_loader(file["target"])

        if self.pre_transform is not None:
            foggy_img = self.pre_transform(foggy_img)

        if self.transform is not None:
            seed = np.random.randint(2147483647)

            random.seed(seed)
            torch.manual_seed(seed)
            foggy = self.transform(foggy_img)

            random.seed(seed)
            torch.manual_seed(seed)
            clear = self.transform(clear_img)

        if self.normalize is not None:
            foggy = self.normalize(foggy)
    
        return file["name"], foggy, clear


class SOTS(torch.utils.data.Dataset):
    def __init__(self, input_root, target_root, transform=None, normalize=True):
        super(SOTS, self).__init__()
        self.transform = transform
        self.normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) if normalize else None

        self.files = []
        for file in os.listdir(input_root):
            if file.endswith('.jpg'):
                input_path = os.path.join(input_root, file)
                target_path = os.path.join(target_root, file[:4]+'.png')
                self.files.append({ "name": file[:-4], "input": input_path, "target": target_path })
 
    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        file = self.files[index]
 
        foggy_img = rgb_loader(file["input"])
        clear_img = rgb_loader(file["target"])

        if self.transform is not None:
            seed = np.random.randint(2147483647)

            random.seed(seed)
            torch.manual_seed(seed)
            foggy = self.transform(foggy_img)

            random.seed(seed)
            torch.manual_seed(seed)
            clear = self.transform(clear_img)

        if self.normalize is not None:
            foggy = self.normalize(foggy)
    
        return file["name"], foggy, clear


if __name__ == '__main__':
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomCrop((224, 224)),
            # transforms.RandomCrop((256, 256)),
            # transforms.RandomHorizontalFlip(),
            transforms.Pad(16, padding_mode='reflect'),
            transforms.ToTensor()
        ]),
        'train_pre': transforms.Compose([
            transforms.ColorJitter(hue=0.5,saturation=0.5)
        ])
    }
    # train_set = TrainSet('../data/train/hazy', '../data/train/gt', pre_transform=data_transforms['train_pre'], transform=data_transforms['train'], normalize=False)

    train_set = TrainSet('../data/train/hazy', '../data/train/gt', transform=data_transforms['train'], normalize=False)

    _, foggy, clear = train_set.__getitem__(0)

    vutils.save_image(torch.cat([foggy.unsqueeze(0), clear.unsqueeze(0)], 0), '../test.png')
