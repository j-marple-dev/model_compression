# -*- coding: utf-8 -*-
"""Configurations for cutmix.

- Author: Hyung-Seok Shin
- Email: hsshin@jmarple.ai
"""

from config.train import densenet_small

config = densenet_small.config
config["CUTMIX"] = dict(beta=1, prob=0.5)
