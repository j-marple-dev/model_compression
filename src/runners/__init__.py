# -*- coding: utf-8 -*-
"""Initialization for training or pruning.

- Author: Curt-Park
- Email: jwpark@jmarple.ai
"""


import datetime
import glob
import os
from runpy import run_path
import shutil
from typing import Any, Dict, Tuple

import torch

from config.config_validator import (
    PruneConfigValidator,
    QuantizeConfigValidator,
    ShrinkConfigValidator,
    TrainConfigValidator,
)
import src.utils as utils

# create directories
curr_time = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
checkpt_path = os.path.join("save", "checkpoint")


def initialize(
    mode: str,
    config_path: str,
    resume: str = "",
    multi_gpu: bool = False,
    gpu_id: int = -1,
) -> Tuple[Dict[str, Any], str, torch.device]:
    """Intialize."""
    # setup device
    device = torch.device("cpu")
    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
            f"{i}" for i in range(torch.cuda.device_count())
        )
        if multi_gpu:
            device = torch.device("cuda")
        elif 0 <= gpu_id < torch.cuda.device_count():
            device = torch.device(f"cuda:{gpu_id}")

    # create directory
    dirs_in_save = ["", "data", "checkpoint"]
    dirs_in_save += [os.path.join("checkpoint", curr_time)] if not resume else []
    for name in dirs_in_save:
        path = os.path.join("save", name)
        if not os.path.exists(path):
            os.mkdir(path)

    # resume or load existing configurations
    if resume:
        dir_prefix = resume
        assert os.path.exists(dir_prefix), f"{dir_prefix} does not exist"
        config_path = glob.glob(os.path.join(dir_prefix, "*.py"))[0]
        config_name = os.path.basename(config_path)
    else:
        assert os.path.exists(config_path), "--config required"
        dir_prefix = os.path.join(checkpt_path, curr_time)
        config_name = os.path.basename(config_path)
        shutil.copyfile(config_path, os.path.join(dir_prefix, config_name))
    config = run_path(config_path)["config"]

    # set logger
    config_name = os.path.splitext(config_name)[0]
    utils.set_logger(filename=os.path.join(dir_prefix, f"{config_name}.log"))

    # config validation check
    if mode == "train":
        TrainConfigValidator(config).check()
    elif mode == "prune":
        PruneConfigValidator(config).check()
    elif mode == "quantize":
        QuantizeConfigValidator(config).check()
    elif mode == "shrink":
        ShrinkConfigValidator(config).check()
    else:
        raise NotImplementedError

    # set random seed
    utils.set_random_seed(config["SEED"])

    return config, dir_prefix, device
