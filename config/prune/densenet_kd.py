# -*- coding: utf-8 -*-
"""Configurations for knowledge distillation with densenet.

- Author: Junghoon Kim
- Email: jhkim@jmarple.ai
"""

from config.train import densenet_kd

config = {
    "TRAIN_CONFIG": densenet_kd.config,
    "SEED": densenet_kd.config["SEED"],
    "N_PRUNING_ITER": 3,
    "EPOCHS": 300,
    "PRUNE_AMOUNT": 0.2,
    "STORE_PARAM_BEFORE": 0,
    "PRUNE_START_FROM": 0,
}
