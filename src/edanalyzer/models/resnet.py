# from functools import partial
# from typing import Any, Callable, List, Optional, Type, Union
#
# import torch
# import torch.nn as nn
# from torch import Tensor
#
# from torchvision.transforms._presets import ImageClassification
# from torchvision.utils import _log_api_usage_once
# from torchvision.models._api import register_model, Weights, WeightsEnum
# from torchvision.models._meta import _IMAGENET_CATEGORIES
# from torchvision.models._utils import _ovewrite_named_param, handle_legacy_interface
#
#
# __all__ = [
#     "ResNet",
#     "ResNet18_Weights",
#     "ResNet34_Weights",
#     "ResNet50_Weights",
#     "ResNet101_Weights",
#     "ResNet152_Weights",
#     "ResNeXt50_32X4D_Weights",
#     "ResNeXt101_32X8D_Weights",
#     "ResNeXt101_64X4D_Weights",
#     "Wide_ResNet50_2_Weights",
#     "Wide_ResNet101_2_Weights",
#     "resnet18",
#     "resnet34",
#     "resnet50",
#     "resnet101",
#     "resnet152",
#     "resnext50_32x4d",
#     "resnext101_32x8d",
#     "resnext101_64x4d",
#     "wide_resnet50_2",
#     "wide_resnet101_2",
# ]
#
#
# def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv3d:
#     """3x3 convolution with padding"""
#     return nn.Conv3d(
#         in_planes,
#         out_planes,
#         kernel_size=3,
#         stride=stride,
#         padding=dilation,
#         groups=groups,
#         bias=False,
#         dilation=dilation,
#     )
#
#
# def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv3d:
#     """1x1 convolution"""
#     return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
#
#
# class BasicBlock(nn.Module):
#     expansion: int = 1
#
#     def __init__(
#         self,
#         inplanes: int,
#         planes: int,
#         stride: int = 1,
#         downsample: Optional[nn.Module] = None,
#         groups: int = 1,
#         base_width: int = 64,
#         dilation: int = 1,
#         norm_layer: Optional[Callable[..., nn.Module]] = None,
#     ) -> None:
#         super().__init__()
#         if norm_layer is None:
#             norm_layer = nn.BatchNorm3d
#         if groups != 1 or base_width != 64:
#             raise ValueError("BasicBlock only supports groups=1 and base_width=64")
#         if dilation > 1:
#             raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
#         # Both self.conv1 and self.downsample layers downsample the input when stride != 1
#         self.conv1 = conv3x3(inplanes, planes, stride)
#         self.bn1 = norm_layer(planes)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = conv3x3(planes, planes)
#         self.bn2 = norm_layer(planes)
#         self.downsample = downsample
#         self.stride = stride
#
#     def forward(self, x: Tensor) -> Tensor:
#         identity = x
#
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#
#         if self.downsample is not None:
#             identity = self.downsample(x)
#
#         out += identity
#         out = self.relu(out)
#
#         return out
#
#
# class Bottleneck(nn.Module):
#     # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
#     # while original implementation places the stride at the first 1x1 convolution(self.conv1)
#     # according to "Deep residual learning for image recognition" https://arxiv.org/abs/1512.03385.
#     # This variant is also known as ResNet V1.5 and improves accuracy according to
#     # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.
#
#     expansion: int = 4
#
#     def __init__(
#         self,
#         inplanes: int,
#         planes: int,
#         stride: int = 1,
#         downsample: Optional[nn.Module] = None,
#         groups: int = 1,
#         base_width: int = 64,
#         dilation: int = 1,
#         norm_layer: Optional[Callable[..., nn.Module]] = None,
#     ) -> None:
#         super().__init__()
#         if norm_layer is None:
#             norm_layer = nn.BatchNorm3d
#         width = int(planes * (base_width / 64.0)) * groups
#         # Both self.conv2 and self.downsample layers downsample the input when stride != 1
#         self.conv1 = conv1x1(inplanes, width)
#         self.bn1 = norm_layer(width)
#         self.conv2 = conv3x3(width, width, stride, groups, dilation)
#         self.bn2 = norm_layer(width)
#         self.conv3 = conv1x1(width, planes * self.expansion)
#         self.bn3 = norm_layer(planes * self.expansion)
#         self.relu = nn.ReLU(inplace=True)
#         self.downsample = downsample
#         self.stride = stride
#
#     def forward(self, x: Tensor) -> Tensor:
#         identity = x
#
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.relu(out)
#
#         out = self.conv3(out)
#         out = self.bn3(out)
#
#         if self.downsample is not None:
#             identity = self.downsample(x)
#
#         out += identity
#         out = self.relu(out)
#
#         return out
#
#
# class ResNet(nn.Module):
#     def __init__(
#         self,
#         block: Type[Union[BasicBlock, Bottleneck]],
#         layers: List[int],
#         num_input=2,
#         num_classes: int = 1000,
#         zero_init_residual: bool = False,
#         groups: int = 1,
#         width_per_group: int = 64,
#         replace_stride_with_dilation: Optional[List[bool]] = None,
#         norm_layer: Optional[Callable[..., nn.Module]] = None,
#     ) -> None:
#         super().__init__()
#         _log_api_usage_once(self)
#         if norm_layer is None:
#             norm_layer = nn.BatchNorm3d
#         self._norm_layer = norm_layer
#
#         self.inplanes = 64
#         self.dilation = 1
#         if replace_stride_with_dilation is None:
#             # each element in the tuple indicates if we should replace
#             # the 2x2 stride with a dilated convolution instead
#             replace_stride_with_dilation = [False, False, False]
#         if len(replace_stride_with_dilation) != 3:
#             raise ValueError(
#                 "replace_stride_with_dilation should be None "
#                 f"or a 3-element tuple, got {replace_stride_with_dilation}"
#             )
#         self.groups = groups
#         self.base_width = width_per_group
#         self.conv1 = nn.Conv3d(num_input, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
#         self.bn1 = norm_layer(self.inplanes)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
#         self.layer1 = self._make_layer(block, 64, layers[0])
#         self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
#         self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
#         self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
#         self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
#         self.fc = nn.Linear(512 * block.expansion, num_classes)
#         self.act = nn.Softmax()
#
#         for m in self.modules():
#             if isinstance(m, nn.Conv3d):
#                 nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
#             elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#
#         # Zero-initialize the last BN in each residual branch,
#         # so that the residual branch starts with zeros, and each residual block behaves like an identity.
#         # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
#         if zero_init_residual:
#             for m in self.modules():
#                 if isinstance(m, Bottleneck) and m.bn3.weight is not None:
#                     nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
#                 elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
#                     nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]
#
#     def _make_layer(
#         self,
#         block: Type[Union[BasicBlock, Bottleneck]],
#         planes: int,
#         blocks: int,
#         stride: int = 1,
#         dilate: bool = False,
#     ) -> nn.Sequential:
#         norm_layer = self._norm_layer
#         downsample = None
#         previous_dilation = self.dilation
#         if dilate:
#             self.dilation *= stride
#             stride = 1
#         if stride != 1 or self.inplanes != planes * block.expansion:
#             downsample = nn.Sequential(
#                 conv1x1(self.inplanes, planes * block.expansion, stride),
#                 norm_layer(planes * block.expansion),
#             )
#
#         layers = []
#         layers.append(
#             block(
#                 self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
#             )
#         )
#         self.inplanes = planes * block.expansion
#         for _ in range(1, blocks):
#             layers.append(
#                 block(
#                     self.inplanes,
#                     planes,
#                     groups=self.groups,
#                     base_width=self.base_width,
#                     dilation=self.dilation,
#                     norm_layer=norm_layer,
#                 )
#             )
#
#         return nn.Sequential(*layers)
#
#     def _forward_impl(self, x: Tensor) -> Tensor:
#         # See note [TorchScript super()]
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)
#
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#
#         x = self.avgpool(x)
#         x = torch.flatten(x, 1)
#         x = self.fc(x)
#
#         return x
#
#         # return self.act(x)
#
#     def forward(self, x: Tensor) -> Tensor:
#         # print(f"Tesnor input shape: {x.shape}. Type: {x.dtype}")
#
#         return self._forward_impl(x)
#
#
# def _resnet(
#     block: Type[Union[BasicBlock, Bottleneck]],
#     layers: List[int],
#     weights: Optional[WeightsEnum],
#     progress: bool,
#     **kwargs: Any,
# ) -> ResNet:
#     if weights is not None:
#         _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))
#
#     model = ResNet(block, layers, **kwargs)
#
#     if weights is not None:
#         model.load_state_dict(weights.get_state_dict(progress=progress, check_hash=True))
#
#     return model
#
#
# _COMMON_META = {
#     "min_size": (1, 1),
#     "categories": _IMAGENET_CATEGORIES,
# }
#
#
# class ResNet18_Weights(WeightsEnum):
#     IMAGENET1K_V1 = Weights(
#         url="https://download.pytorch.org/models/resnet18-f37072fd.pth",
#         transforms=partial(ImageClassification, crop_size=224),
#         meta={
#             **_COMMON_META,
#             "num_params": 11689512,
#             "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#resnet",
#             "_metrics": {
#                 "ImageNet-1K": {
#                     "acc@1": 69.758,
#                     "acc@5": 89.078,
#                 }
#             },
#             "_ops": 1.814,
#             "_file_size": 44.661,
#             "_docs": """These weights reproduce closely the results of the paper using a simple training recipe.""",
#         },
#     )
#     DEFAULT = IMAGENET1K_V1
#
#
# class ResNet34_Weights(WeightsEnum):
#     IMAGENET1K_V1 = Weights(
#         url="https://download.pytorch.org/models/resnet34-b627a593.pth",
#         transforms=partial(ImageClassification, crop_size=224),
#         meta={
#             **_COMMON_META,
#             "num_params": 21797672,
#             "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#resnet",
#             "_metrics": {
#                 "ImageNet-1K": {
#                     "acc@1": 73.314,
#                     "acc@5": 91.420,
#                 }
#             },
#             "_ops": 3.664,
#             "_file_size": 83.275,
#             "_docs": """These weights reproduce closely the results of the paper using a simple training recipe.""",
#         },
#     )
#     DEFAULT = IMAGENET1K_V1
#
#
# class ResNet50_Weights(WeightsEnum):
#     IMAGENET1K_V1 = Weights(
#         url="https://download.pytorch.org/models/resnet50-0676ba61.pth",
#         transforms=partial(ImageClassification, crop_size=224),
#         meta={
#             **_COMMON_META,
#             "num_params": 25557032,
#             "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#resnet",
#             "_metrics": {
#                 "ImageNet-1K": {
#                     "acc@1": 76.130,
#                     "acc@5": 92.862,
#                 }
#             },
#             "_ops": 4.089,
#             "_file_size": 97.781,
#             "_docs": """These weights reproduce closely the results of the paper using a simple training recipe.""",
#         },
#     )
#     IMAGENET1K_V2 = Weights(
#         url="https://download.pytorch.org/models/resnet50-11ad3fa6.pth",
#         transforms=partial(ImageClassification, crop_size=224, resize_size=232),
#         meta={
#             **_COMMON_META,
#             "num_params": 25557032,
#             "recipe": "https://github.com/pytorch/vision/issues/3995#issuecomment-1013906621",
#             "_metrics": {
#                 "ImageNet-1K": {
#                     "acc@1": 80.858,
#                     "acc@5": 95.434,
#                 }
#             },
#             "_ops": 4.089,
#             "_file_size": 97.79,
#             "_docs": """
#                 These weights improve upon the results of the original paper by using TorchVision's `new training recipe
#                 <https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/>`_.
#             """,
#         },
#     )
#     DEFAULT = IMAGENET1K_V2
#
#
# class ResNet101_Weights(WeightsEnum):
#     IMAGENET1K_V1 = Weights(
#         url="https://download.pytorch.org/models/resnet101-63fe2227.pth",
#         transforms=partial(ImageClassification, crop_size=224),
#         meta={
#             **_COMMON_META,
#             "num_params": 44549160,
#             "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#resnet",
#             "_metrics": {
#                 "ImageNet-1K": {
#                     "acc@1": 77.374,
#                     "acc@5": 93.546,
#                 }
#             },
#             "_ops": 7.801,
#             "_file_size": 170.511,
#             "_docs": """These weights reproduce closely the results of the paper using a simple training recipe.""",
#         },
#     )
#     IMAGENET1K_V2 = Weights(
#         url="https://download.pytorch.org/models/resnet101-cd907fc2.pth",
#         transforms=partial(ImageClassification, crop_size=224, resize_size=232),
#         meta={
#             **_COMMON_META,
#             "num_params": 44549160,
#             "recipe": "https://github.com/pytorch/vision/issues/3995#new-recipe",
#             "_metrics": {
#                 "ImageNet-1K": {
#                     "acc@1": 81.886,
#                     "acc@5": 95.780,
#                 }
#             },
#             "_ops": 7.801,
#             "_file_size": 170.53,
#             "_docs": """
#                 These weights improve upon the results of the original paper by using TorchVision's `new training recipe
#                 <https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/>`_.
#             """,
#         },
#     )
#     DEFAULT = IMAGENET1K_V2
#
#
# class ResNet152_Weights(WeightsEnum):
#     IMAGENET1K_V1 = Weights(
#         url="https://download.pytorch.org/models/resnet152-394f9c45.pth",
#         transforms=partial(ImageClassification, crop_size=224),
#         meta={
#             **_COMMON_META,
#             "num_params": 60192808,
#             "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#resnet",
#             "_metrics": {
#                 "ImageNet-1K": {
#                     "acc@1": 78.312,
#                     "acc@5": 94.046,
#                 }
#             },
#             "_ops": 11.514,
#             "_file_size": 230.434,
#             "_docs": """These weights reproduce closely the results of the paper using a simple training recipe.""",
#         },
#     )
#     IMAGENET1K_V2 = Weights(
#         url="https://download.pytorch.org/models/resnet152-f82ba261.pth",
#         transforms=partial(ImageClassification, crop_size=224, resize_size=232),
#         meta={
#             **_COMMON_META,
#             "num_params": 60192808,
#             "recipe": "https://github.com/pytorch/vision/issues/3995#new-recipe",
#             "_metrics": {
#                 "ImageNet-1K": {
#                     "acc@1": 82.284,
#                     "acc@5": 96.002,
#                 }
#             },
#             "_ops": 11.514,
#             "_file_size": 230.474,
#             "_docs": """
#                 These weights improve upon the results of the original paper by using TorchVision's `new training recipe
#                 <https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/>`_.
#             """,
#         },
#     )
#     DEFAULT = IMAGENET1K_V2
#
#
# class ResNeXt50_32X4D_Weights(WeightsEnum):
#     IMAGENET1K_V1 = Weights(
#         url="https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth",
#         transforms=partial(ImageClassification, crop_size=224),
#         meta={
#             **_COMMON_META,
#             "num_params": 25028904,
#             "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#resnext",
#             "_metrics": {
#                 "ImageNet-1K": {
#                     "acc@1": 77.618,
#                     "acc@5": 93.698,
#                 }
#             },
#             "_ops": 4.23,
#             "_file_size": 95.789,
#             "_docs": """These weights reproduce closely the results of the paper using a simple training recipe.""",
#         },
#     )
#     IMAGENET1K_V2 = Weights(
#         url="https://download.pytorch.org/models/resnext50_32x4d-1a0047aa.pth",
#         transforms=partial(ImageClassification, crop_size=224, resize_size=232),
#         meta={
#             **_COMMON_META,
#             "num_params": 25028904,
#             "recipe": "https://github.com/pytorch/vision/issues/3995#new-recipe",
#             "_metrics": {
#                 "ImageNet-1K": {
#                     "acc@1": 81.198,
#                     "acc@5": 95.340,
#                 }
#             },
#             "_ops": 4.23,
#             "_file_size": 95.833,
#             "_docs": """
#                 These weights improve upon the results of the original paper by using TorchVision's `new training recipe
#                 <https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/>`_.
#             """,
#         },
#     )
#     DEFAULT = IMAGENET1K_V2
#
#
# class ResNeXt101_32X8D_Weights(WeightsEnum):
#     IMAGENET1K_V1 = Weights(
#         url="https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth",
#         transforms=partial(ImageClassification, crop_size=224),
#         meta={
#             **_COMMON_META,
#             "num_params": 88791336,
#             "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#resnext",
#             "_metrics": {
#                 "ImageNet-1K": {
#                     "acc@1": 79.312,
#                     "acc@5": 94.526,
#                 }
#             },
#             "_ops": 16.414,
#             "_file_size": 339.586,
#             "_docs": """These weights reproduce closely the results of the paper using a simple training recipe.""",
#         },
#     )
#     IMAGENET1K_V2 = Weights(
#         url="https://download.pytorch.org/models/resnext101_32x8d-110c445d.pth",
#         transforms=partial(ImageClassification, crop_size=224, resize_size=232),
#         meta={
#             **_COMMON_META,
#             "num_params": 88791336,
#             "recipe": "https://github.com/pytorch/vision/issues/3995#new-recipe-with-fixres",
#             "_metrics": {
#                 "ImageNet-1K": {
#                     "acc@1": 82.834,
#                     "acc@5": 96.228,
#                 }
#             },
#             "_ops": 16.414,
#             "_file_size": 339.673,
#             "_docs": """
#                 These weights improve upon the results of the original paper by using TorchVision's `new training recipe
#                 <https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/>`_.
#             """,
#         },
#     )
#     DEFAULT = IMAGENET1K_V2
#
#
# class ResNeXt101_64X4D_Weights(WeightsEnum):
#     IMAGENET1K_V1 = Weights(
#         url="https://download.pytorch.org/models/resnext101_64x4d-173b62eb.pth",
#         transforms=partial(ImageClassification, crop_size=224, resize_size=232),
#         meta={
#             **_COMMON_META,
#             "num_params": 83455272,
#             "recipe": "https://github.com/pytorch/vision/pull/5935",
#             "_metrics": {
#                 "ImageNet-1K": {
#                     "acc@1": 83.246,
#                     "acc@5": 96.454,
#                 }
#             },
#             "_ops": 15.46,
#             "_file_size": 319.318,
#             "_docs": """
#                 These weights were trained from scratch by using TorchVision's `new training recipe
#                 <https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/>`_.
#             """,
#         },
#     )
#     DEFAULT = IMAGENET1K_V1
#
#
# class Wide_ResNet50_2_Weights(WeightsEnum):
#     IMAGENET1K_V1 = Weights(
#         url="https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth",
#         transforms=partial(ImageClassification, crop_size=224),
#         meta={
#             **_COMMON_META,
#             "num_params": 68883240,
#             "recipe": "https://github.com/pytorch/vision/pull/912#issue-445437439",
#             "_metrics": {
#                 "ImageNet-1K": {
#                     "acc@1": 78.468,
#                     "acc@5": 94.086,
#                 }
#             },
#             "_ops": 11.398,
#             "_file_size": 131.82,
#             "_docs": """These weights reproduce closely the results of the paper using a simple training recipe.""",
#         },
#     )
#     IMAGENET1K_V2 = Weights(
#         url="https://download.pytorch.org/models/wide_resnet50_2-9ba9bcbe.pth",
#         transforms=partial(ImageClassification, crop_size=224, resize_size=232),
#         meta={
#             **_COMMON_META,
#             "num_params": 68883240,
#             "recipe": "https://github.com/pytorch/vision/issues/3995#new-recipe-with-fixres",
#             "_metrics": {
#                 "ImageNet-1K": {
#                     "acc@1": 81.602,
#                     "acc@5": 95.758,
#                 }
#             },
#             "_ops": 11.398,
#             "_file_size": 263.124,
#             "_docs": """
#                 These weights improve upon the results of the original paper by using TorchVision's `new training recipe
#                 <https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/>`_.
#             """,
#         },
#     )
#     DEFAULT = IMAGENET1K_V2
#
#
# class Wide_ResNet101_2_Weights(WeightsEnum):
#     IMAGENET1K_V1 = Weights(
#         url="https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth",
#         transforms=partial(ImageClassification, crop_size=224),
#         meta={
#             **_COMMON_META,
#             "num_params": 126886696,
#             "recipe": "https://github.com/pytorch/vision/pull/912#issue-445437439",
#             "_metrics": {
#                 "ImageNet-1K": {
#                     "acc@1": 78.848,
#                     "acc@5": 94.284,
#                 }
#             },
#             "_ops": 22.753,
#             "_file_size": 242.896,
#             "_docs": """These weights reproduce closely the results of the paper using a simple training recipe.""",
#         },
#     )
#     IMAGENET1K_V2 = Weights(
#         url="https://download.pytorch.org/models/wide_resnet101_2-d733dc28.pth",
#         transforms=partial(ImageClassification, crop_size=224, resize_size=232),
#         meta={
#             **_COMMON_META,
#             "num_params": 126886696,
#             "recipe": "https://github.com/pytorch/vision/issues/3995#new-recipe",
#             "_metrics": {
#                 "ImageNet-1K": {
#                     "acc@1": 82.510,
#                     "acc@5": 96.020,
#                 }
#             },
#             "_ops": 22.753,
#             "_file_size": 484.747,
#             "_docs": """
#                 These weights improve upon the results of the original paper by using TorchVision's `new training recipe
#                 <https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/>`_.
#             """,
#         },
#     )
#     DEFAULT = IMAGENET1K_V2
#
#
# # @register_model()
# @handle_legacy_interface(weights=("pretrained", ResNet18_Weights.IMAGENET1K_V1))
# def resnet18(*, weights: Optional[ResNet18_Weights] = None, progress: bool = True, **kwargs: Any) -> ResNet:
#     """ResNet-18 from `Deep Residual Learning for Image Recognition <https://arxiv.org/pdf/1512.03385.pdf>`__.
#
#     Args:
#         weights (:class:`~torchvision.models.ResNet18_Weights`, optional): The
#             pretrained weights to use. See
#             :class:`~torchvision.models.ResNet18_Weights` below for
#             more details, and possible values. By default, no pre-trained
#             weights are used.
#         progress (bool, optional): If True, displays a progress bar of the
#             download to stderr. Default is True.
#         **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
#             base class. Please refer to the `source code
#             <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
#             for more details about this class.
#
#     .. autoclass:: torchvision.models.ResNet18_Weights
#         :members:
#     """
#     weights = ResNet18_Weights.verify(weights)
#
#     return _resnet(BasicBlock, [2, 2, 2, 2], weights, progress, **kwargs)
#
#
# # @register_model()
# @handle_legacy_interface(weights=("pretrained", ResNet34_Weights.IMAGENET1K_V1))
# def resnet34(*, weights: Optional[ResNet34_Weights] = None, progress: bool = True, **kwargs: Any) -> ResNet:
#     """ResNet-34 from `Deep Residual Learning for Image Recognition <https://arxiv.org/pdf/1512.03385.pdf>`__.
#
#     Args:
#         weights (:class:`~torchvision.models.ResNet34_Weights`, optional): The
#             pretrained weights to use. See
#             :class:`~torchvision.models.ResNet34_Weights` below for
#             more details, and possible values. By default, no pre-trained
#             weights are used.
#         progress (bool, optional): If True, displays a progress bar of the
#             download to stderr. Default is True.
#         **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
#             base class. Please refer to the `source code
#             <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
#             for more details about this class.
#
#     .. autoclass:: torchvision.models.ResNet34_Weights
#         :members:
#     """
#     weights = ResNet34_Weights.verify(weights)
#
#     return _resnet(BasicBlock, [3, 4, 6, 3], weights, progress, **kwargs)
#
#
# # @register_model()
# @handle_legacy_interface(weights=("pretrained", ResNet50_Weights.IMAGENET1K_V1))
# def resnet50(*, weights: Optional[ResNet50_Weights] = None, progress: bool = True, **kwargs: Any) -> ResNet:
#     """ResNet-50 from `Deep Residual Learning for Image Recognition <https://arxiv.org/pdf/1512.03385.pdf>`__.
#
#     .. note::
#        The bottleneck of TorchVision places the stride for downsampling to the second 3x3
#        convolution while the original paper places it to the first 1x1 convolution.
#        This variant improves the accuracy and is known as `ResNet V1.5
#        <https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch>`_.
#
#     Args:
#         weights (:class:`~torchvision.models.ResNet50_Weights`, optional): The
#             pretrained weights to use. See
#             :class:`~torchvision.models.ResNet50_Weights` below for
#             more details, and possible values. By default, no pre-trained
#             weights are used.
#         progress (bool, optional): If True, displays a progress bar of the
#             download to stderr. Default is True.
#         **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
#             base class. Please refer to the `source code
#             <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
#             for more details about this class.
#
#     .. autoclass:: torchvision.models.ResNet50_Weights
#         :members:
#     """
#     weights = ResNet50_Weights.verify(weights)
#
#     return _resnet(Bottleneck, [3, 4, 6, 3], weights, progress, **kwargs)
#
#
# # @register_model()
# @handle_legacy_interface(weights=("pretrained", ResNet101_Weights.IMAGENET1K_V1))
# def resnet101(*, weights: Optional[ResNet101_Weights] = None, progress: bool = True, **kwargs: Any) -> ResNet:
#     """ResNet-101 from `Deep Residual Learning for Image Recognition <https://arxiv.org/pdf/1512.03385.pdf>`__.
#
#     .. note::
#        The bottleneck of TorchVision places the stride for downsampling to the second 3x3
#        convolution while the original paper places it to the first 1x1 convolution.
#        This variant improves the accuracy and is known as `ResNet V1.5
#        <https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch>`_.
#
#     Args:
#         weights (:class:`~torchvision.models.ResNet101_Weights`, optional): The
#             pretrained weights to use. See
#             :class:`~torchvision.models.ResNet101_Weights` below for
#             more details, and possible values. By default, no pre-trained
#             weights are used.
#         progress (bool, optional): If True, displays a progress bar of the
#             download to stderr. Default is True.
#         **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
#             base class. Please refer to the `source code
#             <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
#             for more details about this class.
#
#     .. autoclass:: torchvision.models.ResNet101_Weights
#         :members:
#     """
#     weights = ResNet101_Weights.verify(weights)
#
#     return _resnet(Bottleneck, [3, 4, 23, 3], weights, progress, **kwargs)
#
#
# # @register_model()
# @handle_legacy_interface(weights=("pretrained", ResNet152_Weights.IMAGENET1K_V1))
# def resnet152(*, weights: Optional[ResNet152_Weights] = None, progress: bool = True, **kwargs: Any) -> ResNet:
#     """ResNet-152 from `Deep Residual Learning for Image Recognition <https://arxiv.org/pdf/1512.03385.pdf>`__.
#
#     .. note::
#        The bottleneck of TorchVision places the stride for downsampling to the second 3x3
#        convolution while the original paper places it to the first 1x1 convolution.
#        This variant improves the accuracy and is known as `ResNet V1.5
#        <https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch>`_.
#
#     Args:
#         weights (:class:`~torchvision.models.ResNet152_Weights`, optional): The
#             pretrained weights to use. See
#             :class:`~torchvision.models.ResNet152_Weights` below for
#             more details, and possible values. By default, no pre-trained
#             weights are used.
#         progress (bool, optional): If True, displays a progress bar of the
#             download to stderr. Default is True.
#         **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
#             base class. Please refer to the `source code
#             <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
#             for more details about this class.
#
#     .. autoclass:: torchvision.models.ResNet152_Weights
#         :members:
#     """
#     weights = ResNet152_Weights.verify(weights)
#
#     return _resnet(Bottleneck, [3, 8, 36, 3], weights, progress, **kwargs)
#
#
# # @register_model()
# @handle_legacy_interface(weights=("pretrained", ResNeXt50_32X4D_Weights.IMAGENET1K_V1))
# def resnext50_32x4d(
#     *, weights: Optional[ResNeXt50_32X4D_Weights] = None, progress: bool = True, **kwargs: Any
# ) -> ResNet:
#     """ResNeXt-50 32x4d model from
#     `Aggregated Residual Transformation for Deep Neural Networks <https://arxiv.org/abs/1611.05431>`_.
#
#     Args:
#         weights (:class:`~torchvision.models.ResNeXt50_32X4D_Weights`, optional): The
#             pretrained weights to use. See
#             :class:`~torchvision.models.ResNext50_32X4D_Weights` below for
#             more details, and possible values. By default, no pre-trained
#             weights are used.
#         progress (bool, optional): If True, displays a progress bar of the
#             download to stderr. Default is True.
#         **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
#             base class. Please refer to the `source code
#             <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
#             for more details about this class.
#     .. autoclass:: torchvision.models.ResNeXt50_32X4D_Weights
#         :members:
#     """
#     weights = ResNeXt50_32X4D_Weights.verify(weights)
#
#     _ovewrite_named_param(kwargs, "groups", 32)
#     _ovewrite_named_param(kwargs, "width_per_group", 4)
#     return _resnet(Bottleneck, [3, 4, 6, 3], weights, progress, **kwargs)
#
#
# # @register_model()
# @handle_legacy_interface(weights=("pretrained", ResNeXt101_32X8D_Weights.IMAGENET1K_V1))
# def resnext101_32x8d(
#     *, weights: Optional[ResNeXt101_32X8D_Weights] = None, progress: bool = True, **kwargs: Any
# ) -> ResNet:
#     """ResNeXt-101 32x8d model from
#     `Aggregated Residual Transformation for Deep Neural Networks <https://arxiv.org/abs/1611.05431>`_.
#
#     Args:
#         weights (:class:`~torchvision.models.ResNeXt101_32X8D_Weights`, optional): The
#             pretrained weights to use. See
#             :class:`~torchvision.models.ResNeXt101_32X8D_Weights` below for
#             more details, and possible values. By default, no pre-trained
#             weights are used.
#         progress (bool, optional): If True, displays a progress bar of the
#             download to stderr. Default is True.
#         **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
#             base class. Please refer to the `source code
#             <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
#             for more details about this class.
#     .. autoclass:: torchvision.models.ResNeXt101_32X8D_Weights
#         :members:
#     """
#     weights = ResNeXt101_32X8D_Weights.verify(weights)
#
#     _ovewrite_named_param(kwargs, "groups", 32)
#     _ovewrite_named_param(kwargs, "width_per_group", 8)
#     return _resnet(Bottleneck, [3, 4, 23, 3], weights, progress, **kwargs)
#
#
# # @register_model()
# @handle_legacy_interface(weights=("pretrained", ResNeXt101_64X4D_Weights.IMAGENET1K_V1))
# def resnext101_64x4d(
#     *, weights: Optional[ResNeXt101_64X4D_Weights] = None, progress: bool = True, **kwargs: Any
# ) -> ResNet:
#     """ResNeXt-101 64x4d model from
#     `Aggregated Residual Transformation for Deep Neural Networks <https://arxiv.org/abs/1611.05431>`_.
#
#     Args:
#         weights (:class:`~torchvision.models.ResNeXt101_64X4D_Weights`, optional): The
#             pretrained weights to use. See
#             :class:`~torchvision.models.ResNeXt101_64X4D_Weights` below for
#             more details, and possible values. By default, no pre-trained
#             weights are used.
#         progress (bool, optional): If True, displays a progress bar of the
#             download to stderr. Default is True.
#         **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
#             base class. Please refer to the `source code
#             <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
#             for more details about this class.
#     .. autoclass:: torchvision.models.ResNeXt101_64X4D_Weights
#         :members:
#     """
#     weights = ResNeXt101_64X4D_Weights.verify(weights)
#
#     _ovewrite_named_param(kwargs, "groups", 64)
#     _ovewrite_named_param(kwargs, "width_per_group", 4)
#     return _resnet(Bottleneck, [3, 4, 23, 3], weights, progress, **kwargs)
#
#
# # @register_model()
# @handle_legacy_interface(weights=("pretrained", Wide_ResNet50_2_Weights.IMAGENET1K_V1))
# def wide_resnet50_2(
#     *, weights: Optional[Wide_ResNet50_2_Weights] = None, progress: bool = True, **kwargs: Any
# ) -> ResNet:
#     """Wide ResNet-50-2 model from
#     `Wide Residual Networks <https://arxiv.org/abs/1605.07146>`_.
#
#     The model is the same as ResNet except for the bottleneck number of channels
#     which is twice larger in every block. The number of channels in outer 1x1
#     convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
#     channels, and in Wide ResNet-50-2 has 2048-1024-2048.
#
#     Args:
#         weights (:class:`~torchvision.models.Wide_ResNet50_2_Weights`, optional): The
#             pretrained weights to use. See
#             :class:`~torchvision.models.Wide_ResNet50_2_Weights` below for
#             more details, and possible values. By default, no pre-trained
#             weights are used.
#         progress (bool, optional): If True, displays a progress bar of the
#             download to stderr. Default is True.
#         **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
#             base class. Please refer to the `source code
#             <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
#             for more details about this class.
#     .. autoclass:: torchvision.models.Wide_ResNet50_2_Weights
#         :members:
#     """
#     weights = Wide_ResNet50_2_Weights.verify(weights)
#
#     _ovewrite_named_param(kwargs, "width_per_group", 64 * 2)
#     return _resnet(Bottleneck, [3, 4, 6, 3], weights, progress, **kwargs)
#
#
# # @register_model()
# @handle_legacy_interface(weights=("pretrained", Wide_ResNet101_2_Weights.IMAGENET1K_V1))
# def wide_resnet101_2(
#     *, weights: Optional[Wide_ResNet101_2_Weights] = None, progress: bool = True, **kwargs: Any
# ) -> ResNet:
#     """Wide ResNet-101-2 model from
#     `Wide Residual Networks <https://arxiv.org/abs/1605.07146>`_.
#
#     The model is the same as ResNet except for the bottleneck number of channels
#     which is twice larger in every block. The number of channels in outer 1x1
#     convolutions is the same, e.g. last block in ResNet-101 has 2048-512-2048
#     channels, and in Wide ResNet-101-2 has 2048-1024-2048.
#
#     Args:
#         weights (:class:`~torchvision.models.Wide_ResNet101_2_Weights`, optional): The
#             pretrained weights to use. See
#             :class:`~torchvision.models.Wide_ResNet101_2_Weights` below for
#             more details, and possible values. By default, no pre-trained
#             weights are used.
#         progress (bool, optional): If True, displays a progress bar of the
#             download to stderr. Default is True.
#         **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
#             base class. Please refer to the `source code
#             <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
#             for more details about this class.
#     .. autoclass:: torchvision.models.Wide_ResNet101_2_Weights
#         :members:
#     """
#     weights = Wide_ResNet101_2_Weights.verify(weights)
#
#     _ovewrite_named_param(kwargs, "width_per_group", 64 * 2)
#     return _resnet(Bottleneck, [3, 4, 23, 3], weights, progress, **kwargs)


import torch
import torch.nn as nn
import torch.nn.init as init

__all__ = ['SqueezeNet', 'squeezenet1_0', 'squeezenet1_1']

model_urls = {
    'squeezenet1_0': 'https://download.pytorch.org/models/squeezenet1_0-a815701f.pth',
    'squeezenet1_1': 'https://download.pytorch.org/models/squeezenet1_1-f364aa15.pth',
}


class Fire(nn.Module):

    def __init__(self, inplanes, squeeze_planes,
                 expand1x1_planes, expand3x3_planes):
        super(Fire, self).__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Conv3d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv3d(squeeze_planes, expand1x1_planes,
                                   kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv3d(squeeze_planes, expand3x3_planes,
                                   kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))
        ], 1)


class SqueezeNet(nn.Module):

    def __init__(self, version='1_0', num_input=2, num_classes=1000):
        super(SqueezeNet, self).__init__()
        self.num_classes = num_classes
        if version == '1_0':
            self.features = nn.Sequential(
                nn.Conv3d(num_input, 96, kernel_size=7, stride=2),
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
        elif version == '1_1':
            self.features = nn.Sequential(
                nn.Conv3d(num_input, 64, kernel_size=3, stride=2),
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
            raise ValueError("Unsupported SqueezeNet version {version}:"
                             "1_0 or 1_1 expected".format(version=version))

        # Final convolution is initialized differently from the rest
        final_conv = nn.Conv3d(512, self.num_classes, kernel_size=1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            final_conv,
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((1, 1, 1))
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

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)

        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3] * x.shape[4])

        # return self.act(x)
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


import torch
import torch.nn as nn

# from .utils import load_state_dict_from_url


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 num_input=2,
                 num_classes=2,
                 zero_init_residual=False,
                 groups=1,
                 width_per_group=64,
                 replace_stride_with_dilation=None,
                 norm_layer=None,
                 headless=False

                 ):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.headless = headless
        self.conv1 = nn.Conv3d(num_input, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.drop1 = nn.Dropout()
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.drop2 = nn.Dropout()
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.drop3 = nn.Dropout()
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.drop4 = nn.Dropout()
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.drop5 = nn.Dropout()
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # self.act = nn.Softmax()

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.drop1(x)

        x = self.layer1(x)
        x = self.drop2(x)
        x = self.layer2(x)
        x = self.drop3(x)
        x = self.layer3(x)
        x = self.drop4(x)
        x = self.layer4(x)
        x = self.drop5(x)

        x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3] * x.shape[4])
        if not self.headless:
            x = self.fc(x)

        return x
        # return self.act(x)


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet10(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [1, 1, 1, 1], pretrained, progress,
                   **kwargs)


def resnet18(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def resnet34(pretrained=False, progress=True, **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet50(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet101(pretrained=False, progress=True, **kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)


def resnet152(pretrained=False, progress=True, **kwargs):
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress,
                   **kwargs)


def resnext50_32x4d(pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def resnext101_32x8d(pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)


def wide_resnet50_2(pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def wide_resnet101_2(pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet101_2', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)


#######################################
# Transpose
#######################################


def trans_conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.ConvTranspose3d(in_planes, out_planes, kernel_size=3, stride=stride,
                              padding=dilation, groups=groups, bias=False, dilation=dilation)


def trans_conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.ConvTranspose3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class TransBasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = trans_conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = trans_conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, out):
        identity = out

        if self.downsample is not None:
            identity = self.downsample(out)

        out = self.bn2(out)
        # out = self.bn2(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.bn1(out)

        out = self.conv1(out)

        out += identity
        out = self.relu(out)

        return out


# class Bottleneck(nn.Module):
#     expansion = 4
#
#     def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
#                  base_width=64, dilation=1, norm_layer=None):
#         super(Bottleneck, self).__init__()
#         if norm_layer is None:
#             norm_layer = nn.BatchNorm3d
#         width = int(planes * (base_width / 64.)) * groups
#         # Both self.conv2 and self.downsample layers downsample the input when stride != 1
#         self.conv1 = conv1x1(inplanes, width)
#         self.bn1 = norm_layer(width)
#         self.conv2 = conv3x3(width, width, stride, groups, dilation)
#         self.bn2 = norm_layer(width)
#         self.conv3 = conv1x1(width, planes * self.expansion)
#         self.bn3 = norm_layer(planes * self.expansion)
#         self.relu = nn.ReLU(inplace=True)
#         self.downsample = downsample
#         self.stride = stride
#
#     def forward(self, x):
#         identity = x
#
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.relu(out)
#
#         out = self.conv3(out)
#         out = self.bn3(out)
#
#         if self.downsample is not None:
#             identity = self.downsample(x)
#
#         out += identity
#         out = self.relu(out)
#
#         return out


class TransResNet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 num_input=2,
                 num_classes=2,
                 zero_init_residual=False,
                 groups=1,
                 width_per_group=64,
                 replace_stride_with_dilation=None,
                 norm_layer=None,
                 headless=False

                 ):
        super(TransResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.headless = headless
        self.conv1 = nn.Conv3d(
            self.inplanes,
            1,
            kernel_size=7,
            stride=2, padding=3,
            bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.drop1 = nn.Dropout()
        self.layer1 = self._make_layer(block, 128, 64)
        self.drop2 = nn.Dropout()
        self.layer2 = self._make_layer(block, 256, 128, stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.drop3 = nn.Dropout()
        self.layer3 = self._make_layer(block, 512, 256, stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.drop4 = nn.Dropout()
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.drop5 = nn.Dropout()
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # self.act = nn.Softmax()

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, TransBasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        # if stride != 1 or self.inplanes != planes * block.expansion:
        #     downsample = nn.Sequential(
        #         conv1x1(self.inplanes, planes * block.expansion, stride),
        #         norm_layer(planes * block.expansion),
        #     )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):

        x = x.view(-1, self.input_layers, 1, 1, 1)
        x = self.avgpool(x)
        x = self.drop5(x)
        x = self.layer4(x)
        x = self.drop4(x)
        x = self.layer3(x)
        x = self.drop3(x)
        x = self.layer2(x)
        x = self.drop2(x)
        x = self.layer1(x)
        x = self.drop1(x)
        # x = self.maxpool(x)
        x = self.relu(x)
        x = self.bn1(x)

        x = self.conv1(x)

        # x = torch.flatten(x, 1)
        # x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3] * x.shape[4])
        # if not self.headless:
        #     x = self.fc(x)

        return x
        # return self.act(x)


def _trans_resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = TransResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def trans_resnet10(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('trans_resnet18', TransBasicBlock, [1, 1, 1, 1], pretrained, progress,
                   **kwargs)
