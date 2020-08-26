# -*- coding: utf-8 -*-
"""Configurations for Magnitude layerwise pruning.

- Author: Junghoon Kim
- Email: jhkim@jmarple.ai
"""

from config.train.cifar100 import simplenet

train_config = simplenet.config
train_config.update({"REGULARIZER": "BnWeight", "REGULARIZER_PARAMS": dict(coeff=1e-5)})
config = {
    "TRAIN_CONFIG": train_config,
    "N_PRUNING_ITER": 5,
    "PRUNE_METHOD": "SlimMagnitude",
    "PRUNE_PARAMS": dict(
        PRUNE_AMOUNT=0.2,
        NORM=2,
        STORE_PARAM_BEFORE=train_config["EPOCHS"],
        TRAIN_START_FROM=0,
        PRUNE_AT_BEST=False,
    ),
}
