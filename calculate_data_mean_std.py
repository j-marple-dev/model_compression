# -*- coding: utf-8 -*-
"""Image Mean Std Calculator.

- Author: Curt-Park
- Email: jwpark@jmarple.ai
"""


import argparse

import torch
import torch.utils.data as data

from src import utils
from src.runners import initialize

# arguments
parser = argparse.ArgumentParser(description="Data mean std calculator.")
parser.add_argument(
    "--config", type=str, default="config/train/simplenet.py", help="Configuration path"
)
parser.add_argument("--gpu", default=0, type=int, help="GPU id to use")
parser.set_defaults(wlog=False)
args = parser.parse_args()

# initialize
config, dir_prefix, device = initialize("train", args.config, gpu_id=args.gpu)
logger = utils.get_logger()

# calculate data mean, std
logger.info("Get training sets")
trainsets, _ = utils.get_dataset(
    config["DATASET"],
    config["AUG_TRAIN"],
    config["AUG_TEST"],
    config["AUG_TRAIN_PARAMS"],
    config["AUG_TEST_PARAMS"],
)

logger.info("Calculating trainsets' mean and std")
combined_dataset = data.ConcatDataset(trainsets)
mean, std, n_img, n_cnt = (
    torch.zeros(3),
    torch.zeros(3),
    len(combined_dataset),
    0,
)

logger.info(f"Total {n_img} images")
stack_images = torch.stack([img.mean(1).mean(1) for img, _ in combined_dataset])
stack_images.to(device)
mean = stack_images.mean(0)
std = stack_images.std(0)
logger.info(f"{n_img} train data's MEAN: {mean}, STD: {std}")
