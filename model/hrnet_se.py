import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True, normalization=None, activation='prelu'):
        super(ConvBlock, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)

        if normalization == 'batch':
            self.norm = nn.BatchNorm2d(out_channels)
        elif normalization == 'instance':
            self.norm = nn.InstanceNorm2d(out_channels)
        else:
            self.norm = None

        if activation == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif activation == 'prelu':
            self.act = nn.PReLU()
        elif activation == 'lrelu':
            self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        elif activation == 'tanh':
            self.act = nn.Tanh()
        elif activation == 'sigmoid':
            self.act = nn.Sigmoid()
        else:
            self.act = None

    def forward(self, x):
        out = self.conv(x)

        if self.norm is not None:
            out = self.norm(out)

        if self.act is not None:
            out = self.act(out)

        return out

class BasicResidualBlock(nn.Module):
    expansion = 1
    
    def __init__(self, in_channels, channels, stride=1, bias=True, normalization=None, activation='prelu', downsample=None):
        super(BasicResidualBlock, self).__init__()

        self.conv1 = ConvBlock(in_channels, channels, stride=stride, bias=bias, normalization=normalization, activation=activation)
        self.conv2 = ConvBlock(channels, channels, bias=bias, normalization=normalization, activation=None)
        self.downsample = downsample
        self.prelu = nn.PReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        out += x if self.downsample is None else self.downsample(x)

        out = self.prelu(out)

        return out

class BottleneckResidualBlock(nn.Module):
    expansion = 4
    
    def __init__(self, in_channels, channels, stride=1, bias=True, normalization=None, activation='prelu', downsample=None):
        super(BottleneckResidualBlock, self).__init__()

        self.conv1 = ConvBlock(in_channels, channels, kernel_size=1, padding=0, bias=bias, normalization=normalization, activation=activation)
        self.conv2 = ConvBlock(channels, channels, stride=stride, bias=bias, normalization=normalization, activation=activation)
        self.conv3 = ConvBlock(channels, self.expansion * channels, kernel_size=1, padding=0, bias=bias, normalization=normalization, activation=None)
        self.downsample = downsample
        self.prelu = nn.PReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        out += x if self.downsample is None else self.downsample(x)

        out = self.prelu(out)

        return out

class Adapter(nn.Module):
    def __init__(self, in_channels, contraction=4):
        super(Adapter, self).__init__()

        channels = in_channels // contraction

        self.pipeline = nn.Sequential(
            nn.Linear(in_channels, channels),
            nn.ReLU(inplace=True),
            nn.Linear(channels, in_channels)
        ) 

    def forward(self, x):
        out = self.pipeline(x)
        out = out.view(x.size(0), 1, 1)

        return out

class DomainAdaption(nn.Module):
    def __init__(self, in_channels):
        super(DomainAdaption, self).__init__()

        self.conv1 = ConvBlock(in_channels, in_channels)
        self.conv2 = ConvBlock(in_channels, in_channels, activation=None)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()

        adapters = []
        for i in range(3):
            adapters.append(Adapter(in_channels))
        self.adapters = nn.ModuleList(adapters)

        self.sigmoid = nn.Sigmoid()
        self.prelu = nn.PReLU()

    def forward(self, x, intensity):
        n, c, _, _ = x.size()

        out = self.conv1(x)
        out = self.conv2(out)

        x1 = self.pool(out)
        x1 = self.flatten(x1)

        x2 = x1.new_empty((n, c, 1, 1))
        for i in range(n):
            x2[i] = self.adapters[intensity[i]-1](x1[i])

        out *= self.sigmoid(x2)

        out += x
        out = self.prelu(out)

        return out

residual_blocks_dict = {
    'BASIC': BasicResidualBlock,
    'BOTTLENECK': BottleneckResidualBlock
}

class HighResolutionModule(nn.Module):
    def __init__(self, num_branches, block, num_blocks, num_inchannels, num_channels, multi_scale_output=True):
        super(HighResolutionModule, self).__init__()
        self._check_branches(num_branches, num_blocks, num_inchannels, num_channels)

        self.num_branches = num_branches
        self.num_inchannels = num_inchannels
        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(block, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()

        self.prelus = nn.ModuleList([nn.PReLU() for fuse_layer in self.fuse_layers])

    def _check_branches(self, num_branches, num_blocks, num_inchannels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) != NUM_BLOCKS({}).LENGTH({})'.format(num_branches, num_blocks, len(num_blocks))
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) != NUM_INCHANNELS({}).LENGTH({})'.format(num_branches, num_inchannels, len(num_inchannels))
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) != NUM_CHANNELS({}).LENGTH({})'.format(num_branches, num_channels, len(num_channels))
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels, stride=1):
        downsample = None

        if stride != 1 or self.num_inchannels[branch_index] != block.expansion * num_channels[branch_index]:
            downsample = nn.Conv2d(
                    self.num_inchannels[branch_index],
                    block.expansion * num_channels[branch_index],
                    1, stride=stride
                )

        layers = []
        layers.append(
            block(
                self.num_inchannels[branch_index],
                num_channels[branch_index],
                stride=stride,
                downsample=downsample
            )
        )
        self.num_inchannels[branch_index] = block.expansion * num_channels[branch_index]
        for i in range(1, num_blocks[branch_index]):
            layers.append(
                block(
                    self.num_inchannels[branch_index],
                    num_channels[branch_index]
                )
            )

        return nn.Sequential(*layers)

    def _make_branches(self, block, num_blocks, num_channels):
        branches = []

        for i in range(self.num_branches):
            branches.append(self._make_one_branch(i, block, num_blocks, num_channels))

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels

        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(
                        nn.ConvTranspose2d(
                            num_inchannels[j],
                            num_inchannels[i],
                            2**(j-i)+4,
                            stride=2**(j-i),
                            padding=2
                        )
                    )
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i-j):
                        if k == i - j - 1:
                            conv3x3s.append(
                                nn.Conv2d(
                                    num_inchannels[j],
                                    num_inchannels[i],
                                    3, stride=2, padding=1
                                )
                            )
                        else:
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(
                                        num_inchannels[j],
                                        num_inchannels[j],
                                        3, stride=2, padding=1
                                    ),
                                    nn.PReLU()
                                )
                            )
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        x1 = []
        for i in range(self.num_branches):
            x1.append(self.branches[i](x[i]))

        out = []

        for i in range(len(self.fuse_layers)):
            y = x1[0] if i == 0 else self.fuse_layers[i][0](x1[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x1[j]
                elif i < j:
                    y = y + self.fuse_layers[i][j](x1[j], output_size=x[i].size())
                else:
                    y = y + self.fuse_layers[i][j](x1[j])
            out.append(self.prelus[i](y))

        return out

class HighResolutionLayer():
    def __init__(self, cfg):
        super(HighResolutionLayer, self).__init__()

        self.num_modules = cfg['num_modules']
        self.num_branches = cfg['num_branches']
        self.block = cfg['block']
        self.num_blocks = cfg['num_blocks']
        self.num_channels = cfg['num_channels']
        self.multi_scale_output = cfg['multi_scale_output']

class HighResolutionDehazingNet(nn.Module):
    def __init__(self, layers):
        super(HighResolutionDehazingNet, self).__init__()

        self.branches = [layer.num_branches for layer in layers]

        self.in_channels = 64

        self.conv1 = ConvBlock(3, self.in_channels)
        self.conv2 = ConvBlock(self.in_channels, self.in_channels)

        num_bottleneck_blocks = 4
        self.layer1 = self._make_layer(BottleneckResidualBlock, self.in_channels, num_bottleneck_blocks)

        transitions = []
        backbones = []
        attentions = []

        pre_stage_channels = [self.in_channels]

        for i in range(len(layers)):
            layer = layers[i]
            num_modules = layer.num_modules
            num_branches = layer.num_branches
            block = residual_blocks_dict[layer.block]
            num_blocks = layer.num_blocks
            num_inchannels = [block.expansion * channels for channels in layer.num_channels]
            num_channels = layer.num_channels
            multi_scale_output = layer.multi_scale_output

            transition = self._make_transition_layer(pre_stage_channels, num_inchannels)
            transitions.append(transition)

            attention = DomainAdaption(num_inchannels[0])
            attentions.append(attention)

            backbone, pre_stage_channels = self._make_stage(
                num_modules,
                num_branches,
                block,
                num_blocks,
                num_inchannels,
                num_channels,
                multi_scale_output=multi_scale_output
            )
            backbones.append(backbone)

        self.transitions = nn.ModuleList(transitions)
        self.backbones = nn.ModuleList(backbones)
        self.attentions = nn.ModuleList(attentions)

        output_channels_per_layer = 3
        self.upsamples = self._make_final_layer(pre_stage_channels, output_channels_per_layer)

        self.refine = nn.Sequential(
            nn.PReLU(),
            nn.Conv2d(output_channels_per_layer, 3, 3, padding=1)
        )

        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _make_final_layer(self, num_channels_pre_layer, output_channels_per_layer):
        num_branches = len(num_channels_pre_layer)

        output_layers = [nn.Conv2d(num_channels_pre_layer[0], output_channels_per_layer, 1)]

        for i in range(1, num_branches):
            output_layers.append(nn.ConvTranspose2d(num_channels_pre_layer[i], output_channels_per_layer, 2**i+4, stride=2**i, padding=2))

        return nn.ModuleList(output_layers)

    def _make_transition_layer(self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_pre = len(num_channels_pre_layer)
        num_branches_cur = len(num_channels_cur_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(ConvBlock(num_channels_pre_layer[i], num_channels_cur_layer[i]))
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i-num_branches_pre+1):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] if j == i - num_branches_pre else inchannels
                    conv3x3s.append(ConvBlock(inchannels, outchannels, stride=2))
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, channels, blocks, stride=1):
        downsample = None

        if stride != 1 or self.in_channels != block.expansion * channels:
            downsample = nn.Conv2d(
                self.in_channels,
                block.expansion * channels,
                1, stride=stride
            )

        layers = []
        layers.append(block(self.in_channels, channels, stride=stride, downsample=downsample))
        self.in_channels = block.expansion * channels
        for i in range(1, blocks):
            layers.append(block(self.in_channels, channels))

        return nn.Sequential(*layers)

    def _make_stage(self, num_modules, num_branches, block, num_blocks, num_inchannels, num_channels, multi_scale_output=True):
        modules = []
        for i in range(num_modules):
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True

            modules.append(
                HighResolutionModule(
                    num_branches,
                    block,
                    num_blocks,
                    num_inchannels,
                    num_channels,
                    reset_multi_scale_output
                )
            )

            num_inchannels = modules[-1].num_inchannels

        return nn.Sequential(*modules), num_inchannels

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.conv2(x1)
        x1 = self.layer1(x1)

        ys = [x1]

        for i in range(len(self.branches)):
            transition = self.transitions[i]
            backbone = self.backbones[i]
            attention = self.attentions[i]

            xs = []
            for j in range(self.branches[i]):
                if i != 0 and j < self.branches[i-1]:
                    xs.append(ys[j] if transition[j] is None else transition[j](ys[j]))
                else:
                    xs.append(ys[-1] if transition[j] is None else transition[j](ys[-1]))

            ys = backbone(xs)

        out = self.upsamples[0](ys[0])

        for i in range(1, self.branches[-1]):
            out = out + self.upsamples[i](ys[i], output_size=x.size())

        out = self.refine(out)

        return out

if __name__ == '__main__':
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

    model = CompositeHighResolutionNet(layers)

    # state = model.state_dict()
    # for key, val in state.items():
    #     print(key)

    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total number of trainable parameters: {}'.format(num_trainable_params))
