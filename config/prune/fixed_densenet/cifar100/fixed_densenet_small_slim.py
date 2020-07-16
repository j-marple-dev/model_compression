# -*- coding: utf-8 -*-
"""Configurations for naive lottery ticket hypothesis.

- Author: Curt-Park
- Email: jwpark@jmarple.ai
"""

from config.train.fixed_densenet.cifar100 import fixed_densenet_small

train_config = fixed_densenet_small.config
train_config.update({"REGULARIZER": "BnWeight", "REGULARIZER_PARAMS": dict(coeff=1e-5)})
config = {
    "TRAIN_CONFIG": train_config,
    "N_PRUNING_ITER": 15,
    "EPOCHS": 2,
    "PRUNE_METHOD": "NetworkSlimming",
    "PRUNE_AMOUNT": 0.2,
    "STORE_PARAM_BEFORE": 2,  # final weights: EPOCHS
    "PRUNE_START_FROM": 0,
}
