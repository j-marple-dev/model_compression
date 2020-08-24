# -*- coding: utf-8 -*-
"""Configurations for slimming simple network.

- Author: Curt-Park
- Email: jwpark@jmarple.ai
"""

from config.train.cifar100 import simplenet

config = {
    "TRAIN_CONFIG": simplenet.config,
    "N_PRUNING_ITER": 5,
    "PRUNE_METHOD": "LotteryTicketHypothesis",
    "PRUNE_PARAMS": dict(
        PRUNE_AMOUNT=0.2, STORE_PARAM_BEFORE=5, TRAIN_START_FROM=0, PRUNE_AT_BEST=False
    ),
}
