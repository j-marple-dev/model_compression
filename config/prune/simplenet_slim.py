# -*- coding: utf-8 -*-
"""Configurations for lr rewinding with a simple network.

- Author: Curt-Park
- Email: jwpark@jmarple.ai
"""

from config.train import simplenet_bn_reg

config = {
    "TRAIN_CONFIG": simplenet_bn_reg.config,
    "SEED": simplenet_bn_reg.config["SEED"],
    "N_PRUNING_ITER": 5,
    "EPOCHS": 1,
    "PRUNE_METHOD": "NetworkSlimming",
    "PRUNE_AMOUNT": 0.1,
    "STORE_PARAM_BEFORE": 1,
    "PRUNE_START_FROM": 0,
}
