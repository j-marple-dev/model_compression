# -*- coding: utf-8 -*-
"""Fixed DenseNet Model.

All blocks consist of ConvBNReLU for quantization.

- Author: Curt-Park
- Email: jwpark@jmarple.ai
"""

import math
from typing import Any

import torch
import torch.nn as nn

from src.models.common_layers import ConvBNReLU


class Bottleneck(nn.Module):
    """Bottleneck block for DenseNet."""

    def __init__(
        self, inplanes: int, expansion: int = 4, growthRate: int = 12,
    ) -> None:
        """Initialize."""
        super(Bottleneck, self).__init__()
        planes = expansion * growthRate
        self.conv1 = ConvBNReLU(inplanes, planes, kernel_size=1)
        self.conv2 = ConvBNReLU(planes, growthRate, kernel_size=3)

    def _cat(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Concat channels."""
        return torch.cat((x, y), 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward."""
        out = self.conv1(x)
        out = self.conv2(out)
        return self._cat(x, out)


class BasicBlock(nn.Module):
    """BasicBlock for DenseNet."""

    def __init__(
        self, inplanes: int, expansion: int = 1, growthRate: int = 12,
    ) -> None:
        """Initialize."""
        super(BasicBlock, self).__init__()
        self.conv = ConvBNReLU(inplanes, growthRate, kernel_size=3)

    def _cat(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Concat channels."""
        return torch.cat((x, y), 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward."""
        out = self.conv(x)
        return self._cat(x, out)


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
        depth: int = 22,
        growthRate: int = 12,
        compressionRate: int = 2,
        block: "type" = Bottleneck,
    ) -> None:
        """Initialize."""
        assert (depth - 4) % 3 == 0, "depth should be 3n+4"

        super(DenseNet, self).__init__()
        n = (depth - 4) // 3 if block is BasicBlock else (depth - 4) // 6

        self.growthRate = growthRate

        # self.inplanes is a global variable used across multiple
        self.inplanes = growthRate * 2
        self.stem = ConvBNReLU(3, self.inplanes, kernel_size=3)

        self.dense1 = self._make_denseblock(block, n)
        self.trans1 = self._make_transition(compressionRate)
        self.dense2 = self._make_denseblock(block, n)
        self.trans2 = self._make_transition(compressionRate)
        self.dense3 = self._make_denseblock(block, n)

        self.avgpool = nn.AvgPool2d(8)
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

    def _make_denseblock(self, block: "type", blocks: int) -> nn.Module:
        """Make a dense block."""
        layers = []
        for _ in range(blocks):
            # Currently we fix the expansion ratio as the default value
            layers.append(block(self.inplanes, growthRate=self.growthRate))
            self.inplanes += self.growthRate

        return nn.Sequential(*layers)

    def _make_transition(self, compressionRate: int) -> nn.Module:
        """Make a transition."""
        inplanes = self.inplanes
        outplanes = int(math.floor(self.inplanes // compressionRate))
        self.inplanes = outplanes
        return Transition(inplanes, outplanes)

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        """Actual forward procedures."""
        x = self.stem(x)

        x = self.dense1(x)
        x = self.trans1(x)
        x = self.dense2(x)
        x = self.trans2(x)
        x = self.dense3(x)

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
