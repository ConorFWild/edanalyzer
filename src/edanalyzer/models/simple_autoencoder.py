import torch
import torch.nn as nn


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv7x7(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=7, stride=stride,
                     padding=3, groups=groups, bias=False, dilation=dilation)

class Block(nn.Module):
    def __init__(self, inplanes, outplanes, last=False, drop=True, conv=conv3x3):
        super(Block, self).__init__()
        self.conv = conv(inplanes, outplanes, 2)
        self.bn = nn.BatchNorm3d(outplanes)
        self.relu = nn.ReLU(inplace=True)
        if drop:
            self.drop = nn.Dropout()
        else:
            self.drop=None
        self.last = last

    def forward(self, x):
        x = self.conv(x)
        if not self.last:
            x = self.bn(x)
        x = self.relu(x)
        if self.drop:
            x = self.drop(x)
        return x


class SimpleConvolutionalEncoder(nn.Module):
    def __init__(self, input_layers=1):
        super(SimpleConvolutionalEncoder, self).__init__()

        # Layers
        # Layers
        self.input_layers = input_layers
        self.layer1 = Block(input_layers, 8, drop=False, conv=conv7x7)
        self.layer2 = Block(8, 16)
        self.layer3 = Block(16, 32)
        self.layer4 = Block(32, 64)
        self.layer5 = Block(64, 128, last=True)
        self.drop = nn.Dropout()

        # self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

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
        # x = self.drop(x)

        # x = self.avgpool(x)
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3] * x.shape[4])

        return x


def convtranspose3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.ConvTranspose3d(in_planes, out_planes, kernel_size=3, stride=stride,
                              padding=dilation, groups=groups, bias=False, dilation=dilation, output_padding=1)

def convtranspose7x7(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.ConvTranspose3d(in_planes, out_planes, kernel_size=7, stride=stride,
                              padding=3, groups=groups, bias=False, dilation=dilation, output_padding=1)


class BlockTranspose(nn.Module):
    def __init__(self, inplanes, outplanes, drop=True, conv=convtranspose3x3):
        super(BlockTranspose, self).__init__()
        self.conv = conv(inplanes, outplanes, stride=2)
        self.bn = nn.BatchNorm3d(outplanes)
        self.relu = nn.ReLU(inplace=True)
        if drop:
            self.drop = nn.Dropout()
        else:
            self.drop=None

    def forward(self, x):
        x = self.conv(x)
        # print(x.shape)
        x = self.bn(x)
        x = self.relu(x)
        if self.drop:
            x = self.drop(x)
        return x


class SimpleConvolutionalDecoder(nn.Module):
    def __init__(self, input_layers=128):
        super(SimpleConvolutionalDecoder, self).__init__()

        self.input_layers = input_layers

        # Layers
        self.drop = nn.Dropout()
        self.layer1 = BlockTranspose(input_layers, 64)
        self.layer2 = BlockTranspose(64, 32)
        self.layer3 = BlockTranspose(32, 16)
        self.layer4 = BlockTranspose(16, 8)
        self.layer5 = BlockTranspose(8, 1, drop=False, conv=convtranspose7x7)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

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
        x = self.avgpool(x)
        # x = self.drop(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        # x = self.avgpool(x)
        # x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3] * x.shape[4])

        return x
