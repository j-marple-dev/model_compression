# -*- coding: utf-8 -*-
"""Configurations for knowledge distillation with densenet.

- Author: Junghoon Kim
- Email: jhkim@jmarple.ai
"""

from config.train import densenet_kd

train_config = densenet_kd.config
train_config["MODEL_NAME"] = "fixed_densenet"
config = {
    "TRAIN_CONFIG": train_config,
    "SEED": densenet_kd.config["SEED"],
    "N_PRUNING_ITER": 15,
    "EPOCHS": 300,
    "PRUNE_METHOD": "LotteryTicketHypothesis",
    "PRUNE_AMOUNT": 0.2,
    "STORE_PARAM_BEFORE": 0,
    "PRUNE_START_FROM": 0,
}
