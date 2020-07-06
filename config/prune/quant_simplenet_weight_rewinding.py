# -*- coding: utf-8 -*-
"""Configurations for naive lottery ticket hypothesis.

- Author: Junghoon Kim
- Email: jhkim@jmarple.ai
"""

from config.train import quant_simplenet

train_config = quant_simplenet.config
config = {
    "TRAIN_CONFIG": train_config,
    "EPOCHS": 100,
    "N_PRUNING_ITER": 10,
    "PRUNE_AMOUNT": 0.2,
    "STORE_PARAM_BEFORE": 10,
    "PRUNE_START_FROM": 10,
}
