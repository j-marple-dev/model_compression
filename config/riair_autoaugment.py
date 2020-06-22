# -*- coding: utf-8 -*-
"""Configurations for RIAIR's autoaugmentation test.

- Author: Curt-Park
- Email: jwpark@jmarple.ai
- Reference:
    https://github.com/wps712/MicroNetChallenge/blob/cifar100/Train.md
"""

import os

# available for trainer only
config = {
    "SEED": 777,
    "AUG_TRAIN": "autoaugment_train_cifar100_riair",
    "AUG_TEST": "simple_augment_test_cifar100",
    "DATASET": "CIFAR100",
    "MODEL_NAME": "densenet",
    "MODEL_PARAMS": dict(depth=100, num_classes=100, growthRate=12, compressionRate=2),
    "BATCH_SIZE": 64,
    "START_LR": 1e-3,
    "LR": 0.1,
    "MOMENTUM": 0.9,
    "WEIGHT_DECAY": 1e-4,
    "WARMUP_EPOCHS": 5,
    "EPOCHS": 300,
    "N_WORKERS": os.cpu_count(),
}
