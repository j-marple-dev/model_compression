# -*- coding: utf-8 -*-
"""Configurations for naive lottery ticket hypothesis with vanilla cnn.

- Author: Junghoon Kim
- Email: jhkim@jmarple.ai
"""

import os

config = {
    "SEED": 777,
    "AUG_TRAIN": "simple_augment_train_cifar100",
    "AUG_TEST": "simple_augment_test_cifar100",
    "DATASET": "CIFAR100",
    "MODEL_NAME": "simplenet",
    "MODEL_PARAMS": dict(num_classes=100),
    "BATCH_SIZE": 64,
    "START_LR": 1e-4,
    "LR": 0.1,
    "MOMENTUM": 0.9,
    "WEIGHT_DECAY": 1e-4,
    "WARMUP_EPOCHS": 3,
    "EPOCHS": 10,
    "N_PRUNING_ITER": 3,
    "PRUNE_AMOUNT": 0.2,
    "STORE_PARAM_BEFORE": 0,  # final weights: EPOCHS
    "PRUNE_START_FROM": 0,
    "N_WORKERS": os.cpu_count(),
}
