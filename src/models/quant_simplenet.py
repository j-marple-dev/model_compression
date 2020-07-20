# -*- coding: utf-8 -*-
"""Simple CNN Model for quantization.

- Author: Curt-Park
- Email: jwpark@jmarple.ai
"""

import torch
import torch.nn as nn
from torch.quantization import DeQuantStub, QuantStub

from src.models.common_layers import ConvBNReLU
from src.models.simplenet import SimpleNet


class QuantizableSimpleNet(SimpleNet):
    """Quantizable SimpleNet architecture."""

    def __init__(self, num_classes: int) -> None:
        """Initialize."""
        super(QuantizableSimpleNet, self).__init__(num_classes)
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward."""
        x = self.quant(x)
        x = self._forward_impl(x)
        output = self.dequant(x)
        return output

    def fuse_model(self) -> None:
        """Fuse modules and create intrinsic opterators.

        Fused modules are provided for common patterns in CNNs.
        Combining several operations together (like convolution and relu)
        allows for better quantization accuracy.

        References:
            https://pytorch.org/docs/stable/quantization.html#torch-nn-intrinsic
        """
        for m in self.modules():
            if type(m) is ConvBNReLU:
                torch.quantization.fuse_modules(m, ["conv", "bn", "relu"], inplace=True)


def get_model(**kwargs: bool) -> nn.Module:
    """Constructs a Simple model for quantization."""
    return QuantizableSimpleNet(**kwargs)
