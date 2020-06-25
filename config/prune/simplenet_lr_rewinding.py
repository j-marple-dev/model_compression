# -*- coding: utf-8 -*-
"""Configurations for lr rewinding with a simple network.

- Author: Curt-Park
- Email: jwpark@jmarple.ai
"""

from config.train import simplenet

config = {
    "TRAIN_CONFIG": simplenet.config,
    "SEED": simplenet.config["SEED"],
    "N_PRUNING_ITER": 3,
    "EPOCHS": 5,
    "PRUNE_AMOUNT": 0.2,
    "STORE_PARAM_BEFORE": 5,
    "PRUNE_START_FROM": 0,
}
