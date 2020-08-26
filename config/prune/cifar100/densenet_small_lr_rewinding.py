# -*- coding: utf-8 -*-
"""Configurations for Learning Rate Rewinding.

- Author: Curt-Park
- Email: jwpark@jmarple.ai
"""

from config.train.cifar100 import densenet_small

train_config = densenet_small.config
config = {
    "TRAIN_CONFIG": train_config,
    "N_PRUNING_ITER": 15,
    "PRUNE_METHOD": "LotteryTicketHypothesis",
    "PRUNE_PARAMS": dict(
        PRUNE_AMOUNT=0.2,
        STORE_PARAM_BEFORE=train_config["EPOCHS"],
        TRAIN_START_FROM=0,
        PRUNE_AT_BEST=False,
    ),
}
