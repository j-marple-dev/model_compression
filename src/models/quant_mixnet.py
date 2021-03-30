# -*- coding: utf-8 -*-
"""MixNet for quantization - S / M / L.

* Note: SHRINKING IS NOT SUPPORTED!

- Author: Curt-Park
- Email: jwpark@jmarple.ai
- Paper: https://arxiv.org/abs/1907.09595
- Reference: https://github.com/leaderj1001/Mixed-Depthwise-Convolutional-Kernels
"""


import torch
import torch.nn as nn
from torch.quantization import DeQuantStub, QuantStub, fuse_modules

from src.models.common_activations import QuantizableHSwish
from src.models.common_layers import (
    ConvBN,
    ConvBNReLU,
    Identity,
    QuantizableMDConvBlock,
    QuantizableSqueezeExcitation,
)
from src.models.mixnet import MixBlock, MixNet, get_model_kwargs


class QuantizableMixBlock(MixBlock):
    """MixBlock: Using different kernel sizes for each channel chunk."""

    def __init__(self, **kwargs: bool) -> None:
        """Initialize."""
        super(QuantizableMixBlock, self).__init__(**kwargs)
        self.add = nn.quantized.FloatFunctional()

        if self.in_channels != self.exp_channels:
            self.expand_conv = (
                nn.Sequential(
                    ConvBN(self.in_channels, self.exp_channels, kernel_size=1),
                    QuantizableHSwish(inplace=True),
                )
                if self.hswish
                else ConvBNReLU(self.in_channels, self.exp_channels, kernel_size=1)
            )

        self.mdconv = nn.Sequential(
            QuantizableMDConvBlock(
                in_channels=self.exp_channels,
                n_chunks=self.n_chunks,
                stride=self.stride,
                with_relu=not self.hswish,
            ),
            Identity() if not self.hswish else QuantizableHSwish(inplace=True),
        )

        self.se = (
            QuantizableSqueezeExcitation(
                in_channels=self.out_channels,
                se_ratio=self.se_ratio,
            )
            if self.has_se
            else Identity()
        )

    def _add(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Sum two tensors (elementwise)."""
        return self.add.add(x, y)


class QuantizableMixNet(MixNet):
    """MixNet architecture."""

    def __init__(self, **kwargs: bool) -> None:
        """Initialize."""
        super(QuantizableMixNet, self).__init__(**kwargs)
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
        for module in self.modules():
            if type(module) is ConvBNReLU:
                fuse_modules(module, ["conv", "bn", "relu"], inplace=True)
            if type(module) is ConvBN:
                fuse_modules(module, ["conv", "bn"], inplace=True)


def get_model(model_type: str, num_classes: int, dataset: str) -> nn.Module:
    """Constructs a MixNet model."""
    kwargs = get_model_kwargs(model_type, num_classes, dataset)
    kwargs.update(dict(Block=QuantizableMixBlock))
    return QuantizableMixNet(**kwargs)
