# -*- coding: utf-8 -*-
"""Simple CNN Model.

Reference: https://pytorch.org/tutorials/intermediate/pruning_tutorial.html
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.common_layers import ConvBNReLU


class SimpleNet(nn.Module):
    """SimpleNet architecture."""

    def __init__(self, num_classes: int) -> None:
        """Initialize."""
        super(SimpleNet, self).__init__()
        self.conv1 = ConvBNReLU(3, 32, kernel_size=3)
        self.conv2 = ConvBNReLU(32, 64, kernel_size=3)
        self.conv3 = ConvBNReLU(64, 128, kernel_size=3)
        self.conv4 = ConvBNReLU(128, 128, kernel_size=3)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()  # type: ignore
        self.fc1 = nn.Linear(128, num_classes)  # 5x5 image dimension

    def _forward_impl(self, x: torch.Tensor):
        """Actual forward procedures."""
        out = self.conv1(x)
        out = self.conv2(out)
        out = F.max_pool2d(out, (2, 2))
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.avgpool(out)
        out = self.flatten(out)
        out = self.fc1(out)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward."""
        return self._forward_impl(x)


def get_model(**kwargs: bool) -> nn.Module:
    """Constructs a Simple model."""
    return SimpleNet(**kwargs)
