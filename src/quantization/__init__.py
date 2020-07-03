# -*- coding: utf-8 -*-
"""Initialization for quantization.

- Author: Curt-Park
- Email: jwpark@jmarple.ai
"""


import datetime
import os
from runpy import run_path
import shutil
from typing import Any, Dict, Tuple

import src.utils as utils

# create directories
curr_time = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
checkpt_path = os.path.join("save", "checkpoint")


def initialize(config_path: str) -> Tuple[Dict[str, Any], str]:
    """Intialize."""
    # create directory
    dirs_in_save = ["", "data", "checkpoint", os.path.join("checkpoint", curr_time)]
    for name in dirs_in_save:
        path = os.path.join("save", name)
        if not os.path.exists(path):
            os.mkdir(path)

    # load existing configurations
    dir_prefix = os.path.join(checkpt_path, curr_time)
    config_name = os.path.basename(config_path)
    shutil.copyfile(config_path, os.path.join(dir_prefix, config_name))
    config = run_path(config_path)["config"]

    # set logger
    config_name = os.path.splitext(config_name)[0]
    utils.set_logger(filename=os.path.join(dir_prefix, f"{config_name}.log"))

    # hold training config if it is a pruning config
    if "TRAIN_CONFIG" in config:
        config = config["TRAIN_CONFIG"]

    # set random seed
    utils.set_random_seed(config["SEED"])

    return config, dir_prefix
