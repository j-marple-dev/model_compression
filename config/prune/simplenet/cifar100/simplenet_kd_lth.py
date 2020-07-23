# -*- coding: utf-8 -*-
"""Configurations for naive lottery ticket hypothesis with kd, simplenet.

- Author: Junghoon Kim
- Email: jhkim@jmarple.ai
"""

from config.train.simplenet.cifar100 import simplenet_kd

config = {
    "TRAIN_CONFIG": simplenet_kd.config,
    "N_PRUNING_ITER": 15,
    "EPOCHS": 5,
    "PRUNE_METHOD": "LotteryTicketHypothesis",
    "PRUNE_PARAMS": dict(PRUNE_AMOUNT=0.2, STORE_PARAM_BEFORE=20, PRUNE_START_FROM=0),
}
