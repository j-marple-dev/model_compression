# -*- coding: utf-8 -*-
"""Common layer modules.

- Author: Curt-Park
- Email: jwpark@jmarple.ai
"""

from typing import List

import torch
import torch.nn as nn

from src.models.common_activations import HSigmoid, QuantizableHSigmoid
import src.models.utils as model_utils


class Identity(nn.Module):
    """Identity."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the input."""
        return x


class ConvBN(nn.Module):
    """Conv2d + BatchNorm2d.

    If you want Conv2d work as Deption-wise Conv2d, set in_channels = groups.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        bias: bool = False,
        momentum: float = 0.01,
    ) -> None:
        """Initialize."""
        super(ConvBN, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels, momentum=momentum)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward."""
        x = self.conv(x)
        x = self.bn(x)
        return x


class ConvBNReLU(nn.Module):
    """Conv2d + BatchNorm2d + ReLU.

    If you want Conv2d work as Deption-wise Conv2d, set in_channels = groups.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        bias: bool = False,
        momentum: float = 0.01,
    ) -> None:
        """Initialize."""
        super(ConvBNReLU, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels, momentum=momentum)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward."""
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class SqueezeExcitation(nn.Module):
    """Squeeze and Excitation layer."""

    def __init__(self, in_channels: int, se_ratio: float) -> None:
        """Initialize."""
        super(SqueezeExcitation, self).__init__()
        hidden_channels = max(1, int(in_channels * se_ratio))
        self.se_reduce = ConvBNReLU(in_channels, hidden_channels, bias=True)
        self.se_expand = ConvBN(hidden_channels, in_channels, bias=True)
        self.hsig = HSigmoid()

    def _mul(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Multiply two tensors (elementwise)."""
        return x * y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward."""
        se_tensor = torch.mean(x, dim=[2, 3], keepdim=True)
        out = self.se_reduce(se_tensor)
        out = self.se_expand(out)
        out = self._mul(self.hsig(out), x)
        return out


class QuantizableSqueezeExcitation(SqueezeExcitation):
    """Squeeze and Excitation layer."""

    def __init__(self, **kwargs: bool) -> None:
        """Initialize."""
        super(QuantizableSqueezeExcitation, self).__init__(**kwargs)
        self.mul = nn.quantized.FloatFunctional()
        self.hsig = QuantizableHSigmoid()

    def _mul(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Multiply two tensors (elementwise)."""
        return self.mul.mul(x, y)


class MDConvBlock(nn.Module):
    """Mixed-depthwise Conv2d-BN2d-(ReLU)."""

    def __init__(
        self, in_channels: int, n_chunks: int, stride: int = 1, with_relu: int = True
    ) -> None:
        """Initialize."""
        super(MDConvBlock, self).__init__()
        self.n_chunks = n_chunks
        self.split_in_channels = model_utils.split_channels(in_channels, n_chunks)

        self.blocks = nn.ModuleList()
        for idx in range(self.n_chunks):
            kernel_size = 2 * idx + 3
            in_channels = out_channels = self.split_in_channels[idx]
            kwargs = dict(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                groups=in_channels,
            )
            self.blocks.append(ConvBNReLU(**kwargs) if with_relu else ConvBN(**kwargs))

    def _cat(self, block_res: List[torch.Tensor]) -> torch.Tensor:
        """Concat channels of block results."""
        return torch.cat(block_res, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward."""
        split = torch.split(x, self.split_in_channels, dim=1)
        block_res = []
        # torch.jit.script doesn't recognize zip(self.blocks, split)
        for i, block in enumerate(self.blocks):
            block_res.append(block(split[i]))
        return self._cat(block_res)


class QuantizableMDConvBlock(MDConvBlock):
    """Mixed-depthwise Conv2d-BN2d-(ReLU)."""

    def __init__(self, **kwargs: bool) -> None:
        """Initialize."""
        super(QuantizableMDConvBlock, self).__init__(**kwargs)
        self.cat = nn.quantized.FloatFunctional()

    def _cat(self, block_res: List[torch.Tensor]) -> torch.Tensor:
        """Concat channels of block results."""
        return self.cat.cat(block_res, dim=1)
