# -*- coding: utf-8 -*-
"""Collection of regularizers.

- Author: Junghoon Kim
- Email: jhkim@jmarple.ai
"""

from typing import Any, Dict

import torch
import torch.nn as nn


class BnWeight(nn.Module):
    """Apply L1 regularizer on BatchNorm weight.

    Reference:
        Learning Efficient Convolutional Networks through Network Slimming
        (https://arxiv.org/pdf/1708.06519.pdf)

    Attributes:
        model (nn.Module): Model to apply regularizer.
        coefficient (float): weight to regularize.
    """

    def __init__(self, coeff: float) -> None:
        """Initlaize."""
        super().__init__()
        self.coeff = coeff

    def forward(self, model: nn.Module) -> float:
        """Forward."""
        reg = 0.0
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                reg += self.coeff * torch.norm(input=m.weight.data, p=1)
        return reg


def get_regularizer(
    regularizer_name: str,
    regularizer_params: Dict[str, Any],
) -> nn.Module:
    """Create regularizer class."""
    if not regularizer_params:
        regularizer_params = dict()
    return eval(regularizer_name)(**regularizer_params)
