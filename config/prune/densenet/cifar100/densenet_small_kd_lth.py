# -*- coding: utf-8 -*-
"""Configurations for knowledge distillation with densenet.

- Author: Junghoon Kim
- Email: jhkim@jmarple.ai
"""

from config.train.densenet.cifar100 import densenet_small_kd

train_config = densenet_small_kd.config
config = {
    "TRAIN_CONFIG": train_config,
    "N_PRUNING_ITER": 15,
    "EPOCHS": 300,
    "PRUNE_METHOD": "LotteryTicketHypothesis",
    "PRUNE_AMOUNT": 0.2,
    "STORE_PARAM_BEFORE": 10,
    "PRUNE_START_FROM": 10,
}
