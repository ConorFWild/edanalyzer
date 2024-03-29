import torch
import torch.nn as nn
import torch.nn.init as init

__all__ = ['SqueezeNet', 'squeezenet1_0', 'squeezenet1_1']

model_urls = {
    'squeezenet1_0': 'https://download.pytorch.org/models/squeezenet1_0-a815701f.pth',
    'squeezenet1_1': 'https://download.pytorch.org/models/squeezenet1_1-f364aa15.pth',
}


# class Fire(nn.Module):
#
#     def __init__(self, inplanes, squeeze_planes,
#                  expand1x1_planes, expand3x3_planes):
#         super(Fire, self).__init__()
#         self.inplanes = inplanes
#         self.squeeze = nn.Conv3d(inplanes, squeeze_planes, kernel_size=1)
#         self.squeeze_activation = nn.ReLU(inplace=True)
#         self.expand1x1 = nn.Conv3d(squeeze_planes, expand1x1_planes,
#                                    kernel_size=1)
#         self.expand1x1_activation = nn.ReLU(inplace=True)
#         self.expand3x3 = nn.Conv3d(squeeze_planes, expand3x3_planes,
#                                    kernel_size=3, padding=1)
#         self.expand3x3_activation = nn.ReLU(inplace=True)
#
#     def forward(self, x):
#         x = self.squeeze_activation(self.squeeze(x))
#         return torch.cat([
#             self.expand1x1_activation(self.expand1x1(x)),
#             self.expand3x3_activation(self.expand3x3(x))
#         ], 1)
#
#
# class SqueezeNet(nn.Module):
#
#     def __init__(self, version='1_0', num_features=4, num_classes=1000):
#         super(SqueezeNet, self).__init__()
#         self.num_classes = num_classes
#         if version == '1_0':
#             self.features = nn.Sequential(
#                 nn.Conv3d(num_features, 96, kernel_size=7, stride=2),
#                 nn.ReLU(inplace=True),
#                 nn.MaxPool3d(kernel_size=3, stride=2, ceil_mode=True),
#                 Fire(96, 16, 64, 64),
#                 Fire(128, 16, 64, 64),
#                 Fire(128, 32, 128, 128),
#                 nn.MaxPool3d(kernel_size=3, stride=2, ceil_mode=True),
#                 Fire(256, 32, 128, 128),
#                 Fire(256, 48, 192, 192),
#                 Fire(384, 48, 192, 192),
#                 Fire(384, 64, 256, 256),
#                 nn.MaxPool3d(kernel_size=3, stride=2, ceil_mode=True),
#                 Fire(512, 64, 256, 256),
#             )
#         elif version == '1_1':
#             self.features = nn.Sequential(
#                 nn.Conv3d(num_features, 64, kernel_size=3, stride=2),
#                 nn.ReLU(inplace=True),
#                 nn.MaxPool3d(kernel_size=3, stride=2, ceil_mode=True),
#                 Fire(64, 16, 64, 64),
#                 Fire(128, 16, 64, 64),
#                 nn.MaxPool3d(kernel_size=3, stride=2, ceil_mode=True),
#                 Fire(128, 32, 128, 128),
#                 Fire(256, 32, 128, 128),
#                 nn.MaxPool3d(kernel_size=3, stride=2, ceil_mode=True),
#                 Fire(256, 48, 192, 192),
#                 Fire(384, 48, 192, 192),
#                 Fire(384, 64, 256, 256),
#                 Fire(512, 64, 256, 256),
#             )
#         else:
#             # FIXME: Is this needed? SqueezeNet should only be called from the
#             # FIXME: squeezenet1_x() functions
#             # FIXME: This checking is not done for the other models
#             raise ValueError("Unsupported SqueezeNet version {version}:"
#                              "1_0 or 1_1 expected".format(version=version))
#
#         # Final convolution is initialized differently from the rest
#         final_conv = nn.Conv3d(512, self.num_classes, kernel_size=1)
#         self.classifier = nn.Sequential(
#             nn.Dropout(p=0.5),
#             final_conv,
#             nn.ReLU(inplace=True),
#             nn.AdaptiveAvgPool3d((1, 1, 1))
#         )
#
#         for m in self.modules():
#             if isinstance(m, nn.Conv3d):
#                 if m is final_conv:
#                     init.normal_(m.weight, mean=0.0, std=0.01)
#                 else:
#                     init.kaiming_uniform_(m.weight)
#                 if m.bias is not None:
#                     init.constant_(m.bias, 0)
#
#         self.act = nn.Softmax()
#
#     def forward(self, x):
#         x = self.features(x)
#         x = self.classifier(x)
#
#         x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3] * x.shape[4])
#
#         return self.act(x)


class Fire(nn.Module):
    def __init__(self, inplanes: int, squeeze_planes: int, expand1x1_planes: int, expand3x3_planes: int) -> None:
        super().__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Conv3d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv3d(squeeze_planes, expand1x1_planes, kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv3d(squeeze_planes, expand3x3_planes, kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat(
            [self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1
        )


class SqueezeNet(nn.Module):
    def __init__(self, version: str = "1_0", num_classes: int = 1000, num_features=4, dropout: float = 0.5) -> None:
        super().__init__()
        # _log_api_usage_once(self)
        self.num_classes = num_classes
        if version == "1_0":
            self.features = nn.Sequential(
                nn.Conv3d(num_features, 96, kernel_size=7, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool3d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(96, 16, 64, 64),
                Fire(128, 16, 64, 64),
                Fire(128, 32, 128, 128),
                nn.MaxPool3d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 32, 128, 128),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                nn.MaxPool3d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(512, 64, 256, 256),
            )
        elif version == "1_1":
            self.features = nn.Sequential(
                nn.Conv3d(num_features, 64, kernel_size=3, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool3d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(64, 16, 64, 64),
                Fire(128, 16, 64, 64),
                nn.MaxPool3d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(128, 32, 128, 128),
                Fire(256, 32, 128, 128),
                nn.MaxPool3d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                Fire(512, 64, 256, 256),
            )
        else:
            # FIXME: Is this needed? SqueezeNet should only be called from the
            # FIXME: squeezenet1_x() functions
            # FIXME: This checking is not done for the other models
            raise ValueError(f"Unsupported SqueezeNet version {version}: 1_0 or 1_1 expected")

        # Final convolution is initialized differently from the rest
        final_conv = nn.Conv3d(512, self.num_classes, kernel_size=1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout), final_conv, nn.ReLU(inplace=True), nn.AdaptiveAvgPool3d((1, 1, 1))
        )

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                if m is final_conv:
                    init.normal_(m.weight, mean=0.0, std=0.01)
                else:
                    init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

        self.act = nn.Softmax()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        print(f"Features shape: {x.shape}")

        # print(f"Features shape: {x[0,:10,0,0,0]}")
        x = self.classifier(x)
        # print(f"Classified shape: {x[0,:,0,0,0]}")
        # print(f"Classified shape: {x.shape}")
        x = torch.flatten(x, 1)
        # print(f"Before softmax: {x.shape}")
        x = self.act(x)
        # print(f"After softmax: {x.shape}")
        return x


def _squeezenet(version, pretrained, progress, **kwargs):
    model = SqueezeNet(version, **kwargs)
    if pretrained:
        arch = 'squeezenet' + version
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def squeezenet1_0(pretrained=False, progress=True, **kwargs):
    r"""SqueezeNet model architecture from the `"SqueezeNet: AlexNet-level
    accuracy with 50x fewer parameters and <0.5MB model size"
    <https://arxiv.org/abs/1602.07360>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _squeezenet('1_0', pretrained, progress, **kwargs)



def squeezenet1_1(pretrained=False, progress=True, **kwargs):
    r"""SqueezeNet 1.1 model from the `official SqueezeNet repo
    <https://github.com/DeepScale/SqueezeNet/tree/master/SqueezeNet_v1.1>`_.
    SqueezeNet 1.1 has 2.4x less computation and slightly fewer parameters
    than SqueezeNet 1.0, without sacrificing accuracy.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _squeezenet('1_1', pretrained, progress, **kwargs)