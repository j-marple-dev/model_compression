# -*- coding: utf-8 -*-
"""Configurations for RIAIR's autoaugmentation test.

- Author: Curt-Park
- Email: jwpark@jmarple.ai
- Reference:
    https://github.com/wps712/MicroNetChallenge/blob/cifar100/Train.md
"""

from config.train import densenet_small

config = densenet_small.config
config["AUG_TRAIN"] = "autoaugment_train_cifar100_riair"
