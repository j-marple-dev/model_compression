# -*- coding: utf-8 -*-
"""Configurations for training resnet18 (cifar100).

- Author: Curt-Park
- Email: jwpark@jmarple.ai
"""

import os

config = {
    "SEED": 777,
    "AUG_TRAIN": "randaugment_train_cifar100",
    "AUG_TRAIN_PARAMS": dict(n_select=2, level=14),
    "AUG_TEST": "simple_augment_test_cifar100",
    "CUTMIX": dict(beta=1.0, prob=0.5),
    "DATASET": "CIFAR100",
    "MODEL_NAME": "resnet",
    "MODEL_PARAMS": dict(num_classes=100, model_type="resnet18"),
    "CRITERION": "CrossEntropy",
    "CRITERION_PARAMS": dict(num_classes=100),
    "LR_SCHEDULER": "WarmupCosineLR",
    "LR_SCHEDULER_PARAMS": dict(warmup_epochs=10, start_lr=1e-3),
    "BATCH_SIZE": 256,
    "LR": 0.1,
    "MOMENTUM": 0.9,
    "WEIGHT_DECAY": 1e-4,
    "NESTEROV": True,
    "EPOCHS": 300,
    "N_WORKERS": os.cpu_count(),
}
