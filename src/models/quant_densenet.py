# -*- coding: utf-8 -*-
"""DenseNet model for quantization.

- Author: Curt-Park
- Email: jwpark@jmarple.ai
"""

from typing import Any, List, Tuple

import torch
import torch.nn as nn
from torch.quantization import DeQuantStub, QuantStub, fuse_modules

from src.models.common_layers import ConvBN, ConvBNReLU
from src.models.densenet import Bottleneck, DenseBlock, DenseNet


class QuantizableBottleneck(Bottleneck):
    """Quantizable Bottleneck layer."""

    def __init__(
        self,
        inplanes: int,
        expansion: int,
        growthRate: int,
        efficient: bool,
    ) -> None:
        """Initialize."""
        super(QuantizableBottleneck, self).__init__(
            inplanes, expansion, growthRate, efficient=False
        )
        self.cat = nn.quantized.FloatFunctional()

    # arbitrary sized input makes failure when quantizating models
    def forward(self, prev_features: List[torch.Tensor]) -> torch.Tensor:
        """Forward."""
        # checkpoint doesn't work in scripted models
        out = self.cat.cat(prev_features, dim=1)
        out = self.conv1(out)
        out = self.conv2(out)
        return out


class QuantizableDenseBlock(DenseBlock):
    def __init__(
        self,
        inplanes: int,
        blocks: int,
        expansion: int,
        growth_rate: int,
        efficient: bool,
        Layer: "type" = QuantizableBottleneck,
    ):
        super(QuantizableDenseBlock, self).__init__(
            inplanes, blocks, expansion, growth_rate, efficient, Layer
        )
        self.cat = nn.quantized.FloatFunctional()

    def forward(self, init_features: torch.Tensor) -> torch.Tensor:
        features = [init_features]
        for layer in self.layers:
            new_features = layer(features)
            features.append(new_features)
        return self.cat.cat(features, dim=1)


class QuantizableDenseNet(DenseNet):
    """Quantizable DenseNet architecture."""

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
        Block: "type" = QuantizableDenseBlock,
    ) -> None:
        """Initialize."""
        self.inplanes = 0  # type annotation
        super(QuantizableDenseNet, self).__init__(
            num_classes,
            inplanes,
            expansion,
            growthRate,
            compressionRate,
            block_configs,
            small_input,
            efficient,
            Block,
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
            if type(m) is ConvBN:
                fuse_modules(m, ["conv", "bn"], inplace=True)


def get_model(**kwargs: Any) -> nn.Module:
    """Constructs a Simple model for quantization."""
    return QuantizableDenseNet(**kwargs)
