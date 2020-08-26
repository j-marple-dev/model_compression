# -*- coding: utf-8 -*-
"""Configurations for mangnitude channel-wise pruning.

- Author: Junghoon Kim
- Email: jhkim@jmarple.ai
"""

from config.train.cifar100 import densenet_small, densenet_small_finetune

train_config = densenet_small.config
regularizer_params = {
    "REGULARIZER": "BnWeight",
    "REGULARIZER_PARAMS": dict(coeff=1e-5),
    "EPOCHS": train_config["EPOCHS"],
}
train_config.update(regularizer_params)

finetune_config = densenet_small_finetune.config
regularizer_params = {
    "REGULARIZER": "BnWeight",
    "REGULARIZER_PARAMS": dict(coeff=1e-5),
    "EPOCHS": finetune_config["EPOCHS"],
}
finetune_config.update(regularizer_params)

config = {
    "TRAIN_CONFIG": train_config,
    "TRAIN_CONFIG_AT_PRUNE": finetune_config,
    "N_PRUNING_ITER": 15,
    "PRUNE_METHOD": "SlimMagnitude",
    "PRUNE_PARAMS": dict(
        PRUNE_AMOUNT=0.1,
        NORM=2,
        STORE_PARAM_BEFORE=train_config["EPOCHS"],
        TRAIN_START_FROM=0,
        PRUNE_AT_BEST=False,
    ),
}
