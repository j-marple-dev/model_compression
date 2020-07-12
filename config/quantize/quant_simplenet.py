# -*- coding: utf-8 -*-
"""Configurations for quantization for simplenet.

- Author: Curt-Park
- Email: jwpark@jmarple.ai
"""

from config.train import simplenet

config = simplenet.config
config.update(
    {
        "MODEL_NAME": "quant_simplenet",
        "LR_SCHEDULER_PARAMS": dict(warmup_epochs=0, start_lr=1e-4),
        "LR": 1e-4,
        "EPOCHS": 2,
    }
)
