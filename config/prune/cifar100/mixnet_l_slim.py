# -*- coding: utf-8 -*-
"""Configurations for naive lottery ticket hypothesis.

- Author: Curt-Park
- Email: jwpark@jmarple.ai
"""

from config.train.cifar100 import mixnet_l

train_config = mixnet_l.config
train_config.update(
    {
        "REGULARIZER": "BnWeight",
        "REGULARIZER_PARAMS": dict(coeff=1e-5),
        "BATCH_SIZE": 128,
    }
)
config = {
    "TRAIN_CONFIG": train_config,
    "N_PRUNING_ITER": 15,
    "PRUNE_METHOD": "NetworkSlimming",
    "PRUNE_PARAMS": dict(
        PRUNE_AMOUNT=0.2,
        STORE_PARAM_BEFORE=train_config["EPOCHS"],
        TRAIN_START_FROM=0,
        PRUNE_AT_BEST=False,
    ),
}
