# -*- coding: utf-8 -*-
"""Common activation modules.

- Author: Curt-Park
- Email: jwpark@jmarple.ai
"""

import torch
import torch.nn as nn


class HSigmoid(nn.Module):
    """Hard Sigmoid."""

    def __init__(self, inplace: bool = True) -> None:
        """Initialize."""
        super(HSigmoid, self).__init__()
        self.relu6 = nn.ReLU6(inplace=inplace)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward."""
        x = self.relu6(x + 3) / 6
        return x


class QuantizableHSigmoid(nn.Module):
    """Hard Sigmoid for quantization."""

    def __init__(self, inplace: bool = True) -> None:
        """Initialize."""
        super(QuantizableHSigmoid, self).__init__()
        self.relu6 = nn.ReLU6(inplace=inplace)
        self.add_scalar = nn.quantized.FloatFunctional()
        self.mul_scalar = nn.quantized.FloatFunctional()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward."""
        x = self.add_scalar.add_scalar(x, 3.0)
        x = self.relu6(x)
        x = self.mul_scalar.mul_scalar(x, 1 / 6)
        return x


class HSwish(nn.Module):
    """Hard swish."""

    def __init__(self, inplace: bool = True) -> None:
        """Initialize."""
        super(HSwish, self).__init__()
        self.hsig = HSigmoid(inplace=inplace)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward."""
        return x * self.hsig(x)


class QuantizableHSwish(nn.Module):
    """Hard Swish for quantization."""

    def __init__(self, inplace: bool = True) -> None:
        """Initialize."""
        super(QuantizableHSwish, self).__init__()
        self.hsig = QuantizableHSigmoid(inplace=inplace)
        self.mul = nn.quantized.FloatFunctional()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward."""
        return self.mul.mul(x, self.hsig(x))
