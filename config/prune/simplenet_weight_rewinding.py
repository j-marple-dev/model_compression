# -*- coding: utf-8 -*-
"""Configurations for naive lottery ticket hypothesis.

- Author: Junghoon Kim
- Email: jhkim@jmarple.ai
"""

from config.train import simplenet

config = {
    "TRAIN_CONFIG": simplenet.config,
    "SEED": simplenet.config["SEED"],
    "N_PRUNING_ITER": 3,
    "EPOCHS": 5,
    "PRUNE_AMOUNT": 0.2,
    "STORE_PARAM_BEFORE": 0,
    "PRUNE_START_FROM": 0,
}
