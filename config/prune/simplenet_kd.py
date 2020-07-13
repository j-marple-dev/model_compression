# -*- coding: utf-8 -*-
"""Configurations for naive lottery ticket hypothesis with kd, simplenet.

- Author: Junghoon Kim
- Email: jhkim@jmarple.ai
"""

from config.train import simplenet_kd

config = {
    "TRAIN_CONFIG": simplenet_kd.config,
    "SEED": simplenet_kd.config["SEED"],
    "N_PRUNING_ITER": 15,
    "EPOCHS": 5,
    "PRUNE_METHOD": "NetworkSlimming",
    "PRUNE_AMOUNT": 0.2,
    "STORE_PARAM_BEFORE": 0,  # final weights: EPOCHS
    "PRUNE_START_FROM": 0,
}
