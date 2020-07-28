# -*- coding: utf-8 -*-
"""Configurations for training densenet_small.

- Author: Curt-Park
- Email: jwpark@jmarple.ai
"""

import os

config = {"BATCH_SIZE": 128}
config.update(
    {
        "SEED": 777,
        "AUG_TRAIN": "randaugment_train_ai_challenge",
        "AUG_TRAIN_PARAMS": dict(n_select=2, level=None, image_size=112),
        "AUG_TEST": "simple_augment_test_ai_challenge",
        "AUG_TEST_PARAMS": dict(image_size=112),
        "BEST_ACC_METRIC": "f1mean",
        "CUTMIX": dict(beta=1, prob=0.5),
        "DATASET": "AI_CHALLENGE",
        "MODEL_NAME": "densenet",
        "MODEL_PARAMS": dict(
            num_classes=41,
            inplanes=24,
            growthRate=32,
            compressionRate=2,
            block_configs=(6, 12, 24, 16),
            small_input=False,
            efficient=True,
        ),
        "CRITERION": "CrossEntropy",
        "CRITERION_PARAMS": dict(num_classes=41, label_smoothing=0.1),
        "LR_SCHEDULER": "WarmupCosineLR",
        "LR_SCHEDULER_PARAMS": dict(
            warmup_epochs=5, start_lr=1e-3, min_lr=1e-5, n_rewinding=1
        ),
        "LR": 0.1,
        "MOMENTUM": 0.9,
        "WEIGHT_DECAY": 1e-4,
        "NESTEROV": True,
        "EPOCHS": 300,
        "N_WORKERS": os.cpu_count(),
    }
)
