import torch
import torch.nn as nn


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


class Block(nn.Module):
    def __init__(self, inplanes, outplanes):
        self.conv = conv3x3(inplanes, outplanes, 2)
        self.bn = nn.BatchNorm3d(outplanes)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.drop(x)
        return x


class SimpleConvolutionalEncoder(nn.Module):
    def __init__(self):
        super(SimpleConvolutionalEncoder, self).__init__()

        # Layers
        self.layers = [Block(2 ^ j, 2 ^ (j + 1)) for j in range(5)]
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        x = self.avgpool(x)
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3] * x.shape[4])

        return x


def convtranspose3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


class BlockTranspose(nn.Module):
    def __init__(self, inplanes, outplanes):
        self.conv = convtranspose3x3(inplanes, outplanes, 2)
        self.bn = nn.BatchNorm3d(outplanes)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.drop(x)
        return x


class SimpleConvolutionalDecoder(nn.Module):
    def __init__(self):
        super(SimpleConvolutionalDecoder, self).__init__()

        # Layers
        self.layers = [BlockTranspose(2 ^ j, 2 ^ (j + 1)) for j in range(5)]
        # self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        x = self.avgpool(x)
        # x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3] * x.shape[4])

        return x
