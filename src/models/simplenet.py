# -*- coding: utf-8 -*-
"""Simple CNN Model.

Reference: https://pytorch.org/tutorials/intermediate/pruning_tutorial.html
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleNet(nn.Module):
    """SimpleNet architecture."""

    def __init__(self, num_classes: int) -> None:
        """Initialize."""
        super(SimpleNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 3)
        self.bn1 = nn.BatchNorm2d(6)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(6, 6, 3)
        self.bn2 = nn.BatchNorm2d(6)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(6, 16, 3)
        self.bn3 = nn.BatchNorm2d(16)
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv2d(16, 16, 3)
        self.bn4 = nn.BatchNorm2d(16)
        self.relu4 = nn.ReLU()

        self.fc1 = nn.Linear(16 * 5 * 5, num_classes)  # 5x5 image dimension

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward."""
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, (2, 2))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.relu4(self.bn4(self.conv4(x)))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)

        return x


def get_model(**kwargs: bool) -> nn.Module:
    """Constructs a Simple model."""
    return SimpleNet(**kwargs)
