# -*- coding: utf-8 -*-
"""Configurations for naive lottery ticket hypothesis.

- Author: Junghoon Kim
- Email: jhkim@jmarple.ai
"""

from config.train import simplenet

train_config = simplenet.config
train_config["WARMUP_EPOCHS"] = 5
config = {
    "TRAIN_CONFIG": train_config,
    "EPOCHS": 100,
    "N_PRUNING_ITER": 10,
    "PRUNE_AMOUNT": 0.2,
    "STORE_PARAM_BEFORE": 10,
    "PRUNE_START_FROM": 10,
}
