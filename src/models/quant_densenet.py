# -*- coding: utf-8 -*-
"""DenseNet model for quantization.

- Author: Curt-Park
- Email: jwpark@jmarple.ai
"""

import math
from typing import Any

import torch
import torch.nn as nn
from torch.quantization import DeQuantStub, QuantStub, fuse_modules

from src.models.fixed_densenet import BasicBlock, Bottleneck, DenseNet, Transition


class QuantizableBottleneck(Bottleneck):
    """Quantizable Bottleneck layer."""

    def __init__(
        self, inplanes: int, expansion: int = 4, growthRate: int = 12,
    ) -> None:
        """Initialize."""
        super(QuantizableBottleneck, self).__init__(inplanes, expansion, growthRate)
        self.cat = nn.quantized.FloatFunctional()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Foward."""
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.cat.cat((x, out), dim=1)
        return out

    def fuse_model(self) -> None:
        fuse_modules(
            self, [["conv1", "bn1", "relu1"], ["conv2", "bn2", "relu2"]], inplace=True
        )


class QuantizableBasicBlock(BasicBlock):
    """Quantizable BasicBlock for DenseNet."""

    def __init__(
        self, inplanes: int, expansion: int = 1, growthRate: int = 12,
    ) -> None:
        """Initialize."""
        super(QuantizableBasicBlock, self).__init__(inplanes, expansion, growthRate)
        self.cat = nn.quantized.FloatFunctional()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward."""
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.cat.cat((x, out), dim=1)
        return out

    def fuse_model(self) -> None:
        fuse_modules(self, ["conv", "bn", "relu"], inplace=True)


class QuantizableTransition(Transition):
    """Quantizable transition between blocks."""

    def fuse_model(self) -> None:
        fuse_modules(self, ["conv", "bn", "relu"], inplace=True)


class QuantizableDenseNet(DenseNet):
    """Quantizable DenseNet architecture."""

    def __init__(
        self,
        num_classes: int,
        depth: int = 22,
        growthRate: int = 12,
        compressionRate: int = 2,
        block: "type" = QuantizableBottleneck,
    ) -> None:
        """Initialize."""
        self.inplanes = 0  # type annotation
        super(QuantizableDenseNet, self).__init__(
            num_classes, depth, growthRate, compressionRate, block
        )
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    def _make_transition(self, compressionRate: int) -> nn.Module:
        """Make a quantizable transition."""
        inplanes = self.inplanes
        outplanes = int(math.floor(self.inplanes // compressionRate))
        self.inplanes = outplanes
        return QuantizableTransition(inplanes, outplanes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward."""
        x = self.quant(x)
        x = self._forward_impl(x)
        output = self.dequant(x)
        return output

    def fuse_model(self) -> None:
        """Fuse modules and create intrinsic opterators."""
        fuse_modules(self, ["conv", "bn", "relu"], inplace=True)

        for m in self.modules():
            if type(m) in {
                QuantizableBasicBlock,
                QuantizableBottleneck,
                QuantizableTransition,
            }:
                m.fuse_model()


def get_model(**kwargs: Any) -> nn.Module:
    """Constructs a Simple model for quantization."""
    return QuantizableDenseNet(**kwargs)
