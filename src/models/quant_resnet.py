# -*- coding: utf-8 -*-
"""Quantizable ResNet model loader.

* Note: SHRINKING IS NOT SUPPORTED!

- Author: Curt-Park
- Email: jwpark@jmarple.ai
"""


import torch.nn as nn


def get_model(model_type: str, num_classes: int, pretrained: bool = False) -> nn.Module:
    """Constructs a ResNet model."""
    assert model_type in ["resnet18", "resnet50", "resnext101_32x8d"]
    return getattr(
        __import__("torchvision.models.quantization", fromlist=[""]),
        model_type,
    )(pretrained=pretrained, num_classes=num_classes)
