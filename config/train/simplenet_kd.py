# -*- coding: utf-8 -*-
"""Configurations for naive lottery ticket hypothesis with kd, simplenet.

- Author: Junghoon Kim
- Email: jhkim@jmarple.ai
"""

from config.train import simplenet

config = simplenet.config
config_override = {
    "TEACHER_MODEL_NAME": "simplenet",  # simplenet, densenet_large
    "TEACHER_MODEL_PARAMS": dict(num_classes=100),
    "CRITERION": "HintonKLD",  # CrossEntropy, HintonKLD
    "CRITERION_PARAMS": dict(T=4, alpha=0.9),  # dict(), dict(T=4, alpha=0.9),
    "BATCH_SIZE": 512,
    "START_LR": 1e-4,
    "LR": 0.1,
    "MOMENTUM": 0.9,
    "WEIGHT_DECAY": 1e-4,
    "WARMUP_EPOCHS": 3,
    "EPOCHS": 5,
}
config.update(config_override)
