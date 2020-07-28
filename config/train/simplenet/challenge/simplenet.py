# -*- coding: utf-8 -*-
"""Configurations for training as baseline.

- Author: Curt-Park
- Email: jwpark@jmarple.ai
"""

import os

config = {
    "SEED": 777,
    "AUG_TRAIN": "simple_augment_train_ai_challenge",
    "AUG_TEST": "simple_augment_test_ai_challenge",
    "DATASET": "AI_CHALLENGE",
    "MULTI_DATALOADER_CONFIG": dict(
        iter_per_epoch=int(60000 / 128), stratified_sample=True, crawl_ratio=0.2
    ),
    "MODEL_NAME": "simplenet",
    "MODEL_PARAMS": dict(num_classes=41),
    "CRITERION": "CrossEntropy",
    "CRITERION_PARAMS": dict(num_classes=41),
    "LR_SCHEDULER": "WarmupCosineLR",
    "LR_SCHEDULER_PARAMS": dict(warmup_epochs=5, start_lr=1e-3),
    "BATCH_SIZE": 128,
    "LR": 0.1,
    "MOMENTUM": 0.9,
    "WEIGHT_DECAY": 1e-4,
    "NESTEROV": True,
    "EPOCHS": 120,
    "N_WORKERS": os.cpu_count(),
}
