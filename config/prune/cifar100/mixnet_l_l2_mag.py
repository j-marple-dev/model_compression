# -*- coding: utf-8 -*-
"""Configurations for mangnitude layerwise pruning.

- Author: Junghoon Kim
- Email: jhkim@jmarple.ai
"""

from config.train.cifar100 import mixnet_l

train_config = mixnet_l.config
train_config.update({"BATCH_SIZE": 128})
config = {
    "TRAIN_CONFIG": train_config,
    "N_PRUNING_ITER": 15,
    "PRUNE_METHOD": "Magnitude",
    "PRUNE_PARAMS": dict(
        PRUNE_AMOUNT=0.2,
        NORM=2,
        STORE_PARAM_BEFORE=train_config["EPOCHS"],
        TRAIN_START_FROM=0,
        PRUNE_AT_BEST=False,
    ),
}
