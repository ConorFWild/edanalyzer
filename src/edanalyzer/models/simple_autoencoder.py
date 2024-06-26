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
    def __init__(self, inplanes, outplanes, stride, last=False, drop=True, conv=conv3x3):
        super(Block, self).__init__()
        self.pre_conv = conv(inplanes, outplanes, 1)
        self.pre_bn = nn.BatchNorm3d(outplanes)
        self.pre_relu = nn.ReLU(inplace=True)

        self.conv = conv(outplanes, outplanes, stride)
        self.bn = nn.BatchNorm3d(outplanes)
        self.relu = nn.ReLU(inplace=True)
        if drop:
            self.drop = nn.Dropout(p=0.5)
        else:
            self.drop=None
        self.last = last

    def forward(self, x):
        x = self.pre_conv(x)
        x = self.pre_bn(x)
        x = self.pre_relu(x)

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

        # self.mp1 = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = Block(input_layers, 4, 2, drop=False, conv=conv3x3)

        # self.mp2 = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer2 = Block(4, 4, 2,  drop=False)

        # self.mp3 = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer3 = Block(4, 4, 2, drop=False)

        # self.mp4 = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer4 = Block(4, 4, 2, drop=False)

        # self.mp5 = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer5 = Block(4, 16, 2, last=False, drop=False)
        # self.drop = nn.Dropout()

        # self.layer6 = Block(32, 32, 1, last=False, drop=False)


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
        # x = self.layer6(x)

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
            self.drop = nn.Dropout(p=0.25)
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
    def __init__(self, input_layers=256):
        super(SimpleConvolutionalDecoder, self).__init__()

        self.input_layers = input_layers

        # Layers
        self.drop = nn.Dropout()
        self.layer1 = BlockTranspose(input_layers, 128, drop=True )
        self.layer2 = BlockTranspose(128, 64, drop=True )
        self.layer3 = BlockTranspose(64, 32, drop=True)
        self.layer4 = BlockTranspose(32, 16, drop=True )
        self.layer5 = BlockTranspose(16, 1, drop=False, conv=convtranspose7x7)

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
