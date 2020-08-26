# -*- coding: utf-8 -*-
"""Configurations for naive lottery ticket hypothesis with kd, simplenet.

- Author: Junghoon Kim
- Email: jhkim@jmarple.ai
"""

from config.train.cifar100 import simplenet_kd

config = {
    "TRAIN_CONFIG": simplenet_kd.config,
    "N_PRUNING_ITER": 15,
    "PRUNE_METHOD": "LotteryTicketHypothesis",
    "PRUNE_PARAMS": dict(
        PRUNE_AMOUNT=0.2,
        STORE_PARAM_BEFORE=simplenet_kd.config["EPOCHS"],
        TRAIN_START_FROM=0,
        PRUNE_AT_BEST=False,
    ),
}
