# -*- coding: utf-8 -*-
"""DenseNet for quantization.

- Author: Curt-Park
- Email: jwpark@jmarple.ai
"""

import torch
import torch.nn as nn
from torch.quantization import DeQuantStub, QuantStub

from src.models.densenet import Bottleneck, DenseNet


class QuantDenseNet(nn.Module):
    """QuantDenseNet architecture."""

    def __init__(self, **kwargs: bool) -> None:
        """Initialize."""
        super(QuantDenseNet, self).__init__()
        self.quant = QuantStub()
        self.classifier = DenseNet(**kwargs)
        self.dequant = DeQuantStub()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward."""
        x = self.quant(x)
        x = self.classifier(x)
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
        for m in self.classifier.modules():
            if type(m) == Bottleneck:
                m.fuse_model()


def get_model(**kwargs: bool) -> nn.Module:
    """Constructs a Simple model for quantization."""
    return QuantDenseNet(**kwargs)
