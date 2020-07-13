# -*- coding: utf-8 -*-
"""Configurations for naive lottery ticket hypothesis.

- Author: Curt-Park
- Email: jwpark@jmarple.ai
"""

from config.train import fixed_densenet_small

config = {
    "TRAIN_CONFIG": fixed_densenet_small.config,
    "SEED": fixed_densenet_small.config["SEED"],
    "N_PRUNING_ITER": 15,
    "EPOCHS": 2,
    "PRUNE_METHOD": "NetworkSlimming",
    "PRUNE_AMOUNT": 0.2,
    "STORE_PARAM_BEFORE": 2,  # final weights: EPOCHS
    "PRUNE_START_FROM": 0,
}
