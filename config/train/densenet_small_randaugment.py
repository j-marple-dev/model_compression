# -*- coding: utf-8 -*-
"""Configurations for randaugmentation test.

- Author: Hyung-Seok Shin
- Email: hsshin@jmarple.ai
"""

from config.train import densenet_small

config = densenet_small.config
config["AUG_TRAIN"] = "randaugment_train_cifar100"
config["AUG_TRAIN_PARAMS"] = dict(n_select=2, level=14, n_level=31)
