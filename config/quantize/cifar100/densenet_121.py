# -*- coding: utf-8 -*-
"""Configurations for training densenet_121.

- Author: Curt-Park
- Email: jwpark@jmarple.ai
"""

from config.train.cifar100 import densenet_121

config = densenet_121.config
config.update(
    {
        "MODEL_NAME": "quant_densenet",
        "LR_SCHEDULER_PARAMS": dict(warmup_epochs=0, start_lr=1e-4),
        "LR": 1e-4,
        "EPOCHS": 5,
    }
)
