import torch
import torch.nn as nn


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


class Block(nn.Module):
    def __init__(self, inplanes, outplanes, last=False):
        super(Block, self).__init__()
        self.conv = conv3x3(inplanes, outplanes, 2)
        self.bn = nn.BatchNorm3d(outplanes)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout()
        self.last = last

    def forward(self, x):
        x = self.conv(x)
        if not self.last:
            x = self.bn(x)
        x = self.relu(x)
        x = self.drop(x)
        return x


class SimpleConvolutionalEncoder(nn.Module):
    def __init__(self, input_layers=1):
        super(SimpleConvolutionalEncoder, self).__init__()

        # Layers
        # Layers
        self.input_layers = input_layers
        self.layer1 = Block(input_layers, 2)
        self.layer2 = Block(2, 4)
        self.layer3 = Block(4, 8)
        self.layer4 = Block(8, 16)
        self.layer5 = Block(16, 32, last=True)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

        for m in self.modules():
            print(m)
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # for layer in self.layers:
        #     x = layer(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        x = self.avgpool(x)
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3] * x.shape[4])

        return x


def convtranspose3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.ConvTranspose3d(in_planes, out_planes, kernel_size=3, stride=stride,
                              padding=dilation, groups=groups, bias=False, dilation=dilation, output_padding=1)


class BlockTranspose(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(BlockTranspose, self).__init__()
        self.conv = convtranspose3x3(inplanes, outplanes, stride=2)
        self.bn = nn.BatchNorm3d(outplanes)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout()

    def forward(self, x):
        x = self.conv(x)
        # print(x.shape)
        x = self.bn(x)
        x = self.relu(x)
        x = self.drop(x)
        return x


class SimpleConvolutionalDecoder(nn.Module):
    def __init__(self, input_layers=32):
        super(SimpleConvolutionalDecoder, self).__init__()

        self.input_layers = input_layers

        # Layers
        self.layer1 = BlockTranspose(input_layers, 16)
        self.layer2 = BlockTranspose(16, 8)
        self.layer3 = BlockTranspose(8, 4)
        self.layer4 = BlockTranspose(4, 2)
        self.layer5 = BlockTranspose(2, 1)

        self.avgpool = nn.AdaptiveAvgPool3d((32, 32, 32))

        for m in self.modules():
            print(m)
            if isinstance(m, nn.ConvTranspose3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # for layer in self.layers:
        x = x.view(-1, self.input_layers, 1, 1, 1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        x = self.avgpool(x)
        # x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3] * x.shape[4])

        return x
