# -*- coding: utf-8 -*-
"""Configurations for cutmix, simplenet.

- Author: Hyung-Seok Shin
- Email: hsshin@jmarple.ai
"""

from config.train import simplenet

config = simplenet.config
config["CUTMIX"] = dict(beta=1, prob=0.5)
