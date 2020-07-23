# -*- coding: utf-8 -*-
"""Configurations for slimming simple network.

- Author: Curt-Park
- Email: jwpark@jmarple.ai
"""

from config.train.simplenet.cifar100 import simplenet

train_config = simplenet.config
train_config.update({"REGULARIZER": "BnWeight", "REGULARIZER_PARAMS": dict(coeff=1e-5)})
config = {
    "TRAIN_CONFIG": train_config,
    "N_PRUNING_ITER": 15,
    "EPOCHS": 10,
    "PRUNE_METHOD": "SlimMagnitude",
    "PRUNE_PARAMS": dict(
        PRUNE_AMOUNT=0.1, NORM=2, STORE_PARAM_BEFORE=10, PRUNE_START_FROM=0
    ),
}
