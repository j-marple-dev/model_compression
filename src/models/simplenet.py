# -*- coding: utf-8 -*-
"""Simple CNN Model.

Reference: https://pytorch.org/tutorials/intermediate/pruning_tutorial.html
"""

import torch.nn as nn
import torch.nn.functional as F

__all__ = ["get_model"]


class SimpleNet(nn.Module):
    """SimpleNet architecture."""

    def __init__(self, num_classes: int = 10) -> None:
        """Initialize."""
        super(SimpleNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 3)
        self.conv2 = nn.Conv2d(6, 6, 3)
        self.conv3 = nn.Conv2d(6, 16, 3)
        self.conv4 = nn.Conv2d(16, 16, 3)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5x5 image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        """Forward."""
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(F.relu(self.conv4(x)), 2)
        x = x.view(-1, int(x.nelement() / x.shape[0]))

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


def get_model(**kwargs: bool) -> nn.Module:
    """Constructs a Simple model."""
    return SimpleNet(**kwargs)
