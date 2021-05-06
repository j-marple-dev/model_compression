# -*- coding: utf-8 -*-
"""MixNet - S / M / L, MicroNet.

* Note: SHRINKING IS NOT SUPPORTED!

- Author: Curt-Park
- Email: jwpark@jmarple.ai
- Paper: https://arxiv.org/abs/1907.09595
- Differences from the original model:
    Every Mixblock has a skip connection
    Swish function replaced with HSwish
    Mixblock doesn't use group conv operation
    Squeeze-and-Excitation is located behind projection
- Reference:
    https://github.com/leaderj1001/Mixed-Depthwise-Convolutional-Kernels
    https://github.com/Kthyeon/micronet_neurips_challenge
"""


from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn

from src.models.common_activations import HSwish
from src.models.common_layers import (
    ConvBN,
    ConvBNReLU,
    Identity,
    MDConvBlock,
    SqueezeExcitation,
)


def round_filters(
    n_filters: int, multiplier: float = 1.0, divisor: int = 8, min_depth: int = None
) -> int:
    """Get the number of channels."""
    multiplier = multiplier
    divisor = divisor
    min_depth = min_depth

    if not multiplier:
        return n_filters

    n_filters = int(n_filters * multiplier)
    min_depth = min_depth or divisor
    n_filters_new = max(min_depth, int(n_filters + divisor / 2) // divisor * divisor)
    if n_filters_new < 0.9 * n_filters:
        n_filters_new += divisor
    return n_filters_new


class MixBlock(nn.Module):
    """MixBlock: Using different kernel sizes for each channel chunk."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_chunks: int,
        stride: int,
        expand_ratio: float,
        se_ratio: float,
        hswish: bool,
    ) -> None:
        """Initialize."""
        super(MixBlock, self).__init__()
        self.in_channels = in_channels
        self.exp_channels = int(in_channels * expand_ratio)
        self.out_channels = out_channels
        self.n_chunks = n_chunks
        self.stride = stride
        self.expand_ratio = expand_ratio
        self.has_se = se_ratio is not None
        self.se_ratio = se_ratio
        self.hswish = hswish

        self.expand_conv = Identity()
        if self.in_channels != self.exp_channels:
            self.expand_conv = (
                nn.Sequential(
                    ConvBN(self.in_channels, self.exp_channels, kernel_size=1),
                    HSwish(inplace=True),
                )
                if self.hswish
                else ConvBNReLU(self.in_channels, self.exp_channels, kernel_size=1)
            )

        self.mdconv = nn.Sequential(
            MDConvBlock(
                self.exp_channels,
                n_chunks=self.n_chunks,
                stride=self.stride,
                with_relu=not self.hswish,
            ),
            Identity() if not self.hswish else HSwish(inplace=True),
        )

        self.proj_conv = ConvBN(self.exp_channels, self.out_channels, kernel_size=1)

        self.se = (
            SqueezeExcitation(self.out_channels, self.se_ratio)
            if self.has_se
            else Identity()
        )

        self.downsample = (
            ConvBN(
                self.in_channels, self.out_channels, kernel_size=1, stride=self.stride
            )
            if self.stride != 1 or self.in_channels != self.out_channels
            else Identity()
        )

    def _add(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Sum two tensors (elementwise)."""
        return x + y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward."""
        out = self.expand_conv(x)
        out = self.mdconv(out)
        out = self.proj_conv(out)
        out = self.se(out)
        out = self._add(out, self.downsample(x))
        return out


class MixNet(nn.Module):
    """MixNet architecture."""

    def __init__(
        self,
        stem: int,
        stem_stride: int,
        head: int,
        last_out_channels: int,
        block_args: Tuple[List[Any], ...],
        dropout: float = 0.2,
        num_classes: int = 1000,
        Block: "type" = MixBlock,
    ) -> None:
        """Initialize."""
        super(MixNet, self).__init__()
        self.block_args = block_args

        self.stem = nn.Sequential(
            ConvBN(
                in_channels=3,
                out_channels=stem,
                kernel_size=3,
                stride=stem_stride,
            ),
            HSwish(inplace=True),
        )

        layers = []
        for (
            in_channels,
            out_channels,
            n_chunks,
            stride,
            expand_ratio,
            se_ratio,
            hswish,
        ) in block_args:
            layers.append(
                Block(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    n_chunks=n_chunks,
                    stride=stride,
                    expand_ratio=expand_ratio,
                    se_ratio=se_ratio,
                    hswish=hswish,
                )
            )
        self.layers = nn.Sequential(*layers)

        if head:
            self.head = nn.Sequential(
                ConvBN(
                    in_channels=last_out_channels,
                    out_channels=head,
                    kernel_size=1,
                ),
                HSwish(inplace=True),
            )
        else:
            self.head = Identity()
            head = last_out_channels

        self.adapt_avg_pool2d = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout)

        self.fc = nn.Linear(head, num_classes)

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        """Actual forward procedure."""
        out = self.stem(x)
        out = self.layers(out)
        out = self.head(out)
        out = self.adapt_avg_pool2d(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward."""
        return self._forward_impl(x)


def get_model_kwargs(model_type: str, num_classes: int, dataset: str) -> Dict[str, Any]:
    """Return the model kwargs according to the momdel type."""
    if model_type == "MICRONET":
        kwargs = micronet(num_classes=num_classes, dataset=dataset)
    elif model_type == "S":
        kwargs = mixnet_s(num_classes=num_classes, dataset=dataset)
    elif model_type == "M":
        kwargs = mixnet_m(num_classes=num_classes, dataset=dataset)
    elif model_type == "L":
        kwargs = mixnet_l(num_classes=num_classes, dataset=dataset)
    else:
        raise NotImplementedError
    return kwargs


def get_model(model_type: str, num_classes: int, dataset: str) -> nn.Module:
    """Constructs a MixNet model."""
    kwargs = get_model_kwargs(model_type, num_classes, dataset)
    return MixNet(**kwargs)


def micronet(
    num_classes: int = 100,
    multiplier: float = 1.0,
    divisor: int = 8,
    min_depth: int = None,
    dataset: str = "IMAGENET",
) -> Dict[str, Any]:
    """Build MixNet-SS."""
    if dataset == "CIFAR100":
        # in_channels, out_channels, n_chunks, stride, expand_ratio, se_ratio, hswish
        small = (
            [32, 16, 1, 1, 3, None, False],
            [16, 16, 1, 1, 3, None, False],
            [16, 32, 1, 2, 3, None, False],
            [32, 32, 1, 1, 3, 0.25, True],
            [32, 48, 1, 1, 3, 0.25, True],
            [48, 48, 1, 1, 3, 0.25, True],
            [48, 48, 1, 1, 3, 0.25, True],
            [48, 72, 1, 2, 3, 0.25, True],
            [72, 72, 1, 1, 3, 0.25, True],
            [72, 72, 1, 1, 3, 0.25, True],
            [72, 72, 1, 1, 3, 0.25, True],
            [72, 72, 1, 1, 3, 0.25, True],
            [72, 80, 1, 2, 3, 0.25, True],
            [80, 88, 1, 1, 3, 0.25, True],
            [88, 88, 1, 1, 3, 0.25, True],
            [88, 106, 1, 1, 3, 0.25, True],
        )
        stem = 32
        stem_stride = 1
        last_out_channels = 106
        dropout = 0.3
    else:
        raise NotImplementedError

    return dict(
        stem=stem,
        stem_stride=stem_stride,
        head=0,  # head not used
        last_out_channels=last_out_channels,
        block_args=small,
        num_classes=num_classes,
        dropout=dropout,
    )


def mixnet_s(
    num_classes: int = 100,
    multiplier: float = 1.0,
    divisor: int = 8,
    min_depth: int = None,
    dataset: str = "IMAGENET",
) -> Dict[str, Any]:
    """Build MixNet-S."""
    if dataset == "IMAGENET":
        # in_channels, out_channels, n_chunks, stride, expand_ratio, se_ratio, hswish
        small = (
            [16, 16, 1, 1, 1, None, False],
            [16, 24, 1, 2, 6, None, False],
            [24, 24, 1, 1, 3, None, False],
            [24, 40, 3, 2, 6, 0.5, True],
            [40, 40, 2, 1, 6, 0.5, True],
            [40, 40, 2, 1, 6, 0.5, True],
            [40, 40, 2, 1, 6, 0.5, True],
            [40, 80, 3, 2, 6, 0.25, True],
            [80, 80, 2, 1, 6, 0.25, True],
            [80, 80, 2, 1, 6, 0.25, True],
            [80, 120, 3, 1, 6, 0.5, True],
            [120, 120, 4, 1, 3, 0.5, True],
            [120, 120, 4, 2, 3, 0.5, True],
            [120, 200, 5, 1, 6, 0.5, True],
            [200, 200, 4, 1, 6, 0.5, True],
            [200, 200, 4, 1, 6, 0.5, True],
        )
        stem = round_filters(16, multiplier)
        stem_stride = 2
        last_out_channels = round_filters(200, multiplier)
        head = round_filters(1536, multiplier)
    elif dataset == "CIFAR100":
        small = (
            [16, 16, 1, 1, 1, None, False],
            [16, 24, 1, 1, 6, None, False],
            [24, 24, 1, 1, 3, None, False],
            [24, 40, 3, 2, 6, 0.5, True],
            [40, 40, 2, 1, 6, 0.5, True],
            [40, 40, 2, 1, 6, 0.5, True],
            [40, 40, 2, 1, 6, 0.5, True],
            [40, 80, 3, 2, 6, 0.25, True],
            [80, 80, 2, 1, 6, 0.25, True],
            [80, 80, 2, 1, 6, 0.25, True],
            [80, 120, 3, 1, 6, 0.5, True],
            [120, 120, 4, 1, 3, 0.5, True],
            [120, 120, 4, 2, 3, 0.5, True],
            [120, 200, 5, 1, 6, 0.5, True],
            [200, 200, 4, 1, 6, 0.5, True],
            [200, 200, 4, 1, 6, 0.5, True],
        )
        stem = round_filters(16, multiplier)
        stem_stride = 1
        last_out_channels = round_filters(200, multiplier)
        head = round_filters(1536, multiplier)
    else:
        raise NotImplementedError

    return dict(
        stem=stem,
        stem_stride=stem_stride,
        head=head,
        last_out_channels=last_out_channels,
        block_args=small,
        num_classes=num_classes,
    )


def mixnet_m(
    num_classes: int = 1000,
    multiplier: float = 1.0,
    divisor: int = 8,
    min_depth: int = None,
    dataset: str = "IMAGENET",
) -> Dict[str, Any]:
    """Build MixNet-M."""
    if dataset == "IMAGENET":
        medium: Tuple[List[Any], ...] = (
            [24, 24, 1, 1, 1, None, False],
            [24, 32, 3, 2, 6, None, False],
            [32, 32, 1, 1, 3, None, False],
            [32, 40, 4, 2, 6, 0.5, True],
            [40, 40, 2, 1, 6, 0.5, True],
            [40, 40, 2, 1, 6, 0.5, True],
            [40, 40, 2, 1, 6, 0.5, True],
            [40, 80, 3, 2, 6, 0.25, True],
            [80, 80, 4, 1, 6, 0.25, True],
            [80, 80, 4, 1, 6, 0.25, True],
            [80, 80, 4, 1, 6, 0.25, True],
            [80, 120, 1, 1, 6, 0.5, True],
            [120, 120, 4, 1, 3, 0.5, True],
            [120, 120, 4, 1, 3, 0.5, True],
            [120, 120, 4, 1, 3, 0.5, True],
            [120, 200, 4, 2, 6, 0.5, True],
            [200, 200, 4, 1, 6, 0.5, True],
            [200, 200, 4, 1, 6, 0.5, True],
            [200, 200, 4, 1, 6, 0.5, True],
        )
        stem = round_filters(24, multiplier)
        stem_stride = 2
        last_out_channels = round_filters(200, multiplier)
        head = round_filters(1536, multiplier=1.0)
    elif dataset == "CIFAR100":
        medium = (
            [24, 24, 1, 1, 1, None, False],
            [24, 32, 3, 1, 6, None, False],
            [32, 32, 1, 1, 3, None, False],
            [32, 40, 4, 2, 6, 0.5, True],
            [40, 40, 2, 1, 6, 0.5, True],
            [40, 40, 2, 1, 6, 0.5, True],
            [40, 40, 2, 1, 6, 0.5, True],
            [40, 80, 3, 2, 6, 0.25, True],
            [80, 80, 4, 1, 6, 0.25, True],
            [80, 80, 4, 1, 6, 0.25, True],
            [80, 80, 4, 1, 6, 0.25, True],
            [80, 120, 1, 1, 6, 0.5, True],
            [120, 120, 4, 1, 3, 0.5, True],
            [120, 120, 4, 1, 3, 0.5, True],
            [120, 120, 4, 1, 3, 0.5, True],
            [120, 200, 4, 2, 6, 0.5, True],
            [200, 200, 4, 1, 6, 0.5, True],
            [200, 200, 4, 1, 6, 0.5, True],
            [200, 200, 4, 1, 6, 0.5, True],
        )
        stem = round_filters(24, multiplier)
        stem_stride = 1
        last_out_channels = round_filters(200, multiplier)
        head = round_filters(1536, multiplier=1.0)
    else:
        raise NotImplementedError

    for line in medium:
        line[0] = round_filters(line[0], multiplier)
        line[1] = round_filters(line[1], multiplier)

    return dict(
        stem=stem,
        stem_stride=stem_stride,
        head=head,
        last_out_channels=last_out_channels,
        block_args=medium,
        dropout=0.25,
        num_classes=num_classes,
    )


def mixnet_l(num_classes: int = 1000, dataset: str = "IMAGENET") -> Dict[str, Any]:
    """Build MixNet-L."""
    return mixnet_m(num_classes=num_classes, multiplier=1.3, dataset=dataset)
