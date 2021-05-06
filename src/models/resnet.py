# -*- coding: utf-8 -*-
"""ResNet model loader.

* Note: SHRINKING IS NOT SUPPORTED!

- Author: Curt-Park
- Email: jwpark@jmarple.ai
"""

import torch.nn as nn


def get_model(model_type: str, num_classes: int, pretrained: bool = False) -> nn.Module:
    """Constructs a ResNet model."""
    assert model_type in [
        "resnet18",
        "resnet34",
        "resnet50",
        "resnet101",
        "resnet152",
        "resnext50_32x4d",
        "resnext101_32x8d",
        "wide_resnet50_2",
        "wide_resnet101_2",
    ]
    return getattr(
        __import__("torchvision.models", fromlist=[""]),
        model_type,
    )(pretrained=pretrained, num_classes=num_classes)
