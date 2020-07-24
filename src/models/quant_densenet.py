# -*- coding: utf-8 -*-
"""DenseNet model for quantization.

- Author: Curt-Park
- Email: jwpark@jmarple.ai
"""

from typing import Any

import torch
import torch.nn as nn
from torch.quantization import DeQuantStub, QuantStub, fuse_modules

from src.models.common_layers import ConvBNReLU
from src.models.densenet import BasicBlock, Bottleneck, DenseNet


class QuantizableBottleneck(Bottleneck):
    """Quantizable Bottleneck layer."""

    def __init__(
        self, inplanes: int, expansion: int = 4, growthRate: int = 12,
    ) -> None:
        """Initialize."""
        super(QuantizableBottleneck, self).__init__(inplanes, expansion, growthRate)
        self.cat = nn.quantized.FloatFunctional()

    def _cat(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Concat channels."""
        return self.cat.cat((x, y), dim=1)


class QuantizableBasicBlock(BasicBlock):
    """Quantizable BasicBlock for DenseNet."""

    def __init__(
        self, inplanes: int, expansion: int = 1, growthRate: int = 12,
    ) -> None:
        """Initialize."""
        super(QuantizableBasicBlock, self).__init__(inplanes, expansion, growthRate)
        self.cat = nn.quantized.FloatFunctional()

    def _cat(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Concat channels."""
        return self.cat.cat((x, y), dim=1)


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward."""
        x = self.quant(x)
        x = self._forward_impl(x)
        output = self.dequant(x)
        return output

    def fuse_model(self) -> None:
        """Fuse modules and create intrinsic opterators."""
        for m in self.modules():
            if type(m) is ConvBNReLU:
                fuse_modules(m, ["conv", "bn", "relu"], inplace=True)


def get_model(**kwargs: Any) -> nn.Module:
    """Constructs a Simple model for quantization."""
    return QuantizableDenseNet(**kwargs)
