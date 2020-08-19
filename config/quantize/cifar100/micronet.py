# -*- coding: utf-8 -*-
"""Configurations for quantization for MicroNet (CIFAR100).

- Author: Curt-Park
- Email: jwpark@jmarple.ai
"""

from config.train.cifar100 import micronet

config = micronet.config
config.update(
    {
        "MODEL_NAME": "quant_mixnet",
        "LR_SCHEDULER_PARAMS": dict(warmup_epochs=0, start_lr=1e-4),
        "LR": 1e-4,
        "EPOCHS": 2,
    }
)
