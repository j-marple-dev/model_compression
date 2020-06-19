# -*- coding: utf-8 -*-
"""Configurations for lr rewinding with a simple network.

- Author: Curt-Park
- Email: jwpark@jmarple.ai
"""

from src.config.naive_lth_simple import config

config.update(
    {
        "EPOCHS": 10,
        "STORE_PARAM_BEFORE": 10,  # final weights: EPOCHS
        "PRUNE_START_FROM": 0,
    }
)
