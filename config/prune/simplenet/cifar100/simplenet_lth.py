# -*- coding: utf-8 -*-
"""Configurations for slimming simple network.

- Author: Curt-Park
- Email: jwpark@jmarple.ai
"""

from config.train.simplenet.cifar100 import simplenet

config = {
    "TRAIN_CONFIG": simplenet.config,
    "N_PRUNING_ITER": 5,
    "EPOCHS": 20,
    "PRUNE_METHOD": "LotteryTicketHypothesis",
    "PRUNE_PARAMS": dict(PRUNE_AMOUNT=0.2, STORE_PARAM_BEFORE=20, PRUNE_START_FROM=0),
}
