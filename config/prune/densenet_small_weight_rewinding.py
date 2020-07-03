# -*- coding: utf-8 -*-
"""Configurations for naive lottery ticket hypothesis.

- Author: Curt-Park
- Email: jwpark@jmarple.ai
"""

from config.train import densenet_small

config = {
    "TRAIN_CONFIG": densenet_small.config,
    "SEED": densenet_small.config["SEED"],
    "N_PRUNING_ITER": 20,
    "EPOCHS": 300,
    "PRUNE_AMOUNT": 0.2,
    "STORE_PARAM_BEFORE": 0,  # final weights: EPOCHS
    "PRUNE_START_FROM": 0,
}
