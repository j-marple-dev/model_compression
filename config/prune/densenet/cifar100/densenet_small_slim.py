# -*- coding: utf-8 -*-
"""Configurations for network slimming.

- Author: Curt-Park
- Email: jwpark@jmarple.ai
"""

from config.train.densenet.cifar100 import densenet_small

train_config = densenet_small.config
train_config.update({"REGULARIZER": "BnWeight", "REGULARIZER_PARAMS": dict(coeff=1e-5)})
config = {
    "TRAIN_CONFIG": train_config,
    "N_PRUNING_ITER": 5,
    "EPOCHS": 1,
    "PRUNE_METHOD": "NetworkSlimming",
    "PRUNE_PARAMS": dict(
        PRUNE_AMOUNT=0.2,
        NORM=2,
        STORE_PARAM_BEFORE=1,
        PRUNE_START_FROM=0,
        PRUNE_AT_BEST=False,
    ),
}
