# -*- coding: utf-8 -*-
"""Configurations for training densenet_small.

- Author: Curt-Park
- Email: jwpark@jmarple.ai
"""

from config.train.densenet.challenge import densenet_10x4

config = densenet_10x4.config
config.update({"EPOCHS": 150})
