# -*- coding: utf-8 -*-
"""Configurations for training as baseline.

- Author: Curt-Park
- Email: jwpark@jmarple.ai
"""

import os

config = {
    "SEED": 777,
    "AUG_TRAIN": "simple_augment_train_cifar100",
    "AUG_TRAIN_PARAMS": dict(),
    "AUG_TEST": "simple_augment_test_cifar100",
    "DATASET": "CIFAR100",
    "MODEL_NAME": "simplenet",
    "MODEL_PARAMS": dict(num_classes=100),
    "CRITERION": "CrossEntropy",
    "CRITERION_PARAMS": dict(num_classes=100),
    "REGULARIZER": "BnWeight",
    "REGULARIZER_PARAMS": dict(coeff=1e-5),
    "LR_SCHEDULER": "WarmupCosineLR",
    "LR_SCHEDULER_PARAMS": dict(warmup_epochs=3, start_lr=1e-3),
    "BATCH_SIZE": 64,
    "LR": 0.1,
    "MOMENTUM": 0.9,
    "WEIGHT_DECAY": 1e-4,
    "EPOCHS": 5,
    "N_WORKERS": os.cpu_count(),
}
