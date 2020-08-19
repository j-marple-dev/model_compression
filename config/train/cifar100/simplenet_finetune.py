# -*- coding: utf-8 -*-
"""Configurations for slimming simple network.

- Author: Curt-Park
- Email: jwpark@jmarple.ai
"""
import os

config = {
    "SEED": 777,
    "AUG_TRAIN": "simple_augment_train_cifar100",
    "AUG_TEST": "simple_augment_test_cifar100",
    "DATASET": "CIFAR100",
    "MODEL_NAME": "simplenet",
    "MODEL_PARAMS": dict(num_classes=100),
    "CRITERION": "CrossEntropy",
    "CRITERION_PARAMS": dict(num_classes=100),
    "LR_SCHEDULER": "WarmupCosineLR",
    "LR_SCHEDULER_PARAMS": dict(
        warmup_epochs=5, start_lr=1e-3, min_lr=5e-6, n_rewinding=1
    ),
    "BATCH_SIZE": 64,
    "LR": 0.001,
    "MOMENTUM": 0.9,
    "WEIGHT_DECAY": 1e-4,
    "NESTEROV": True,
    "EPOCHS": 10,
    "N_WORKERS": os.cpu_count(),
}
