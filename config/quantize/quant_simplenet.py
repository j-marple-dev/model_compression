# -*- coding: utf-8 -*-
"""Configurations for quantization for simplenet.

- Author: Curt-Park
- Email: jwpark@jmarple.ai
"""

from config.train import quant_simplenet

train_config = quant_simplenet.config
train_config["LR_SCHEDULER_PARAMS"]["warmup_epochs"] = 0
train_config["LR"] = 1e-4
config = {
    "TRAIN_CONFIG": train_config,
    "EPOCHS": 2,
    "QUANT_BACKEND": "fbgemm",
}
