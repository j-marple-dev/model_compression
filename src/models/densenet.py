# -*- coding: utf-8 -*-
"""DenseNet Model.

- Copied and modified from: https://github.com/bearpaw/pytorch-classification
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.quantization import fuse_modules


class Bottleneck(nn.Module):
    """Bottleneck block for DenseNet."""

    def __init__(
        self, inplanes: int, expansion: int = 4, growthRate: int = 12, dropRate: int = 0
    ) -> None:
        """Initialize."""
        super(Bottleneck, self).__init__()
        planes = expansion * growthRate
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, growthRate, kernel_size=3, padding=1, bias=False)
        self.dropRate = dropRate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward."""
        out = self.bn1(x)
        out = self.relu1(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv2(out)
        if self.dropRate > 0:
            out = F.dropout(out, p=self.dropRate, training=self.training)

        out = torch.cat((x, out), 1)

        return out

    def fuse_model(self):
        fuse_modules(self, ["conv1", "bn2", "relu2"], inplace=True)


class BasicBlock(nn.Module):
    """BasicBlock for DenseNet."""

    def __init__(
        self, inplanes: int, expansion: int = 1, growthRate: int = 12, dropRate: int = 0
    ) -> None:
        """Initialize."""
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(
            inplanes, growthRate, kernel_size=3, padding=1, bias=False
        )
        self.relu = nn.ReLU(inplace=True)
        self.dropRate = dropRate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward."""
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        if self.dropRate > 0:
            out = F.dropout(out, p=self.dropRate, training=self.training)

        out = torch.cat((x, out), 1)

        return out


class Transition(nn.Module):
    """Transition between blocks."""

    def __init__(self, inplanes: int, outplanes: int) -> None:
        """Initialize."""
        super(Transition, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward."""
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = F.avg_pool2d(out, 2)
        return out


class DenseNet(nn.Module):
    """DenseNet architecture."""

    def __init__(
        self,
        num_classes: int,
        depth: int = 22,
        growthRate: int = 12,
        compressionRate: int = 2,
        dropRate: int = 0,
        use_basicblock: bool = False,
    ) -> None:
        """Initialize."""
        assert (depth - 4) % 3 == 0, "depth should be 3n+4"

        super(DenseNet, self).__init__()
        block = BasicBlock if use_basicblock else Bottleneck
        n = (depth - 4) // 3 if use_basicblock else (depth - 4) // 6

        self.growthRate = growthRate
        self.dropRate = dropRate

        # self.inplanes is a global variable used across multiple
        self.inplanes = growthRate * 2
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, padding=1, bias=False)
        self.dense1 = self._make_denseblock(block, n)
        self.trans1 = self._make_transition(compressionRate)
        self.dense2 = self._make_denseblock(block, n)
        self.trans2 = self._make_transition(compressionRate)
        self.dense3 = self._make_denseblock(block, n)
        self.bn = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(8)
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
            layers.append(
                block(self.inplanes, growthRate=self.growthRate, dropRate=self.dropRate)
            )
            self.inplanes += self.growthRate

        return nn.Sequential(*layers)

    def _make_transition(self, compressionRate: int):
        """Make a transition."""
        inplanes = self.inplanes
        outplanes = int(math.floor(self.inplanes // compressionRate))
        self.inplanes = outplanes
        return Transition(inplanes, outplanes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward."""
        x = self.conv1(x)

        x = self.trans1(self.dense1(x))
        x = self.trans2(self.dense2(x))
        x = self.dense3(x)
        x = self.bn(x)
        x = self.relu(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def get_model(**kwargs: bool) -> nn.Module:
    """Constructs a ResNet model. """
    return DenseNet(**kwargs)
