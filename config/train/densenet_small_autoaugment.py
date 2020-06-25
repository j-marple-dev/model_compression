# -*- coding: utf-8 -*-
"""Configurations for autoaugmentation test.

- Author: Curt-Park
- Email: jwpark@jmarple.ai
"""

from config.train import densenet_small

config = densenet_small.config
config["AUG_TRAIN"] = "autoaugment_train_cifar100"
