# -*- coding: utf-8 -*-
"""Fixed DenseNet Model.

All blocks consist of ConvBNReLU for quantization.

- Author: Curt-Park
- Email: jwpark@jmarple.ai
- References:
    https://github.com/bearpaw/pytorch-classification
    https://github.com/gpleiss/efficient_densenet_pytorch
"""

import math
from typing import Any, Tuple

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp

from src.models.common_layers import ConvBNReLU


class Bottleneck(nn.Module):
    """Bottleneck block for DenseNet."""

    def __init__(
        self,
        inplanes: int,
        expansion: int,
        growthRate: int,
        efficient: bool,
    ) -> None:
        """Initialize."""
        super(Bottleneck, self).__init__()
        planes = expansion * growthRate
        self.conv1 = ConvBNReLU(inplanes, planes, kernel_size=1)
        self.conv2 = ConvBNReLU(planes, growthRate, kernel_size=3)
        self.efficient = efficient

    def _expand(self, *features: torch.Tensor) -> torch.Tensor:
        """Bottleneck foward function."""
        concated_features = torch.cat(features, 1)
        bottleneck_output = self.conv1(concated_features)
        return bottleneck_output

    def forward(self, *prev_features: torch.Tensor) -> torch.Tensor:
        """Forward."""
        if self.efficient and any(feat.requires_grad for feat in prev_features):
            out = cp.checkpoint(self._expand, *prev_features)
        else:
            out = self._expand(*prev_features)
        out = self.conv2(out)
        return out


class DenseBlock(nn.Module):
    def __init__(
        self,
        inplanes: int,
        blocks: int,
        expansion: int,
        growth_rate: int,
        efficient: bool,
        Layer: "type" = Bottleneck,
    ):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(blocks):
            layer = Layer(
                inplanes=inplanes + i * growth_rate,
                expansion=expansion,
                growthRate=growth_rate,
                efficient=efficient,
            )
            self.layers.append(layer)

    def forward(self, init_features: torch.Tensor) -> torch.Tensor:
        features = [init_features]
        for layer in self.layers:
            new_features = layer(*features)
            features.append(new_features)
        return torch.cat(features, dim=1)


class Transition(nn.Module):
    """Transition between blocks."""

    def __init__(self, inplanes: int, outplanes: int) -> None:
        """Initialize."""
        super(Transition, self).__init__()
        self.conv = ConvBNReLU(inplanes, outplanes, kernel_size=1)
        self.avg_pool = nn.AvgPool2d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward."""
        out = self.conv(x)
        out = self.avg_pool(out)
        return out


class DenseNet(nn.Module):
    """DenseNet architecture."""

    def __init__(
        self,
        num_classes: int,
        inplanes: int,
        expansion: int = 4,
        growthRate: int = 12,
        compressionRate: int = 2,
        block_configs: Tuple[int, ...] = (6, 12, 24, 16),
        small_input: bool = True,  # e.g. CIFAR100
        efficient: bool = False,  # memory efficient dense block
        Block: "type" = DenseBlock,
    ) -> None:
        """Initialize."""
        super(DenseNet, self).__init__()

        self.growthRate = growthRate
        self.inplanes = inplanes
        self.expansion = expansion

        if small_input:
            self.stem = ConvBNReLU(3, self.inplanes, kernel_size=3, stride=1)
        else:
            self.stem = nn.Sequential(
                ConvBNReLU(3, self.inplanes, kernel_size=7, stride=2),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False),
            )

        layers = []
        for i, n_bottleneck in enumerate(block_configs):
            dense_block = Block(
                self.inplanes, n_bottleneck, expansion, growthRate, efficient
            )
            layers.append(dense_block)
            self.inplanes += n_bottleneck * self.growthRate
            # not add transition at the end
            if i < len(block_configs) - 1:
                layers.append(self._make_transition(compressionRate))
        self.dense_blocks = nn.Sequential(*layers)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()  # type: ignore
        self.fc = nn.Linear(self.inplanes, num_classes)

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_transition(self, compressionRate: int) -> nn.Module:
        """Make a transition."""
        inplanes = self.inplanes
        outplanes = int(math.floor(self.inplanes // compressionRate))
        self.inplanes = outplanes
        return Transition(inplanes, outplanes)

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        """Actual forward procedures."""
        x = self.stem(x)
        x = self.dense_blocks(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward."""
        return self._forward_impl(x)


def get_model(**kwargs: Any) -> nn.Module:
    """Constructs a ResNet model. """
    return DenseNet(**kwargs)
