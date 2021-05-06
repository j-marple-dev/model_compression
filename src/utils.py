# -*- coding: utf-8 -*-
"""Utils for model compression.

- Author: Curt-Park
- Email: jwpark@jmarple.ai
"""

import glob
import logging
import logging.handlers
import os
import random
import sys
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data
from torchvision.datasets import VisionDataset


def set_random_seed(seed: int) -> None:
    """Set random seed."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    # for CuDNN backend
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True  # type: ignore
        torch.backends.cudnn.benchmark = False  # type: ignore


def get_rand_bbox_coord(
    w: int, h: int, len_ratio: float
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """Get a coordinate of random box."""
    size_hole_w = int(len_ratio * w)
    size_hole_h = int(len_ratio * h)
    x = random.randint(0, w)  # [0, w]
    y = random.randint(0, h)  # [0, h]

    x0 = max(0, x - size_hole_w // 2)
    y0 = max(0, y - size_hole_h // 2)
    x1 = min(w, x + size_hole_w // 2)
    y1 = min(h, y + size_hole_h // 2)
    return (x0, y0), (x1, y1)


def to_onehot(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Convert index based labels into one-hot based labels.

    If labels are one-hot based already(e.g. [0.9, 0.01, 0.03,...]), do nothing.
    """
    if len(labels.size()) == 1:
        return F.one_hot(labels, num_classes).float()
    return labels


def get_dataset(
    dataset_name: str = "CIFAR100",
    transform_train: str = "simple_augment_train_cifar100",
    transform_test: str = "simple_augment_test_cifar100",
    transform_train_params: Dict[str, int] = None,
    transform_test_params: Dict[str, int] = None,
) -> Tuple[VisionDataset, VisionDataset]:
    """Get dataset for training and testing."""
    if not transform_train_params:
        transform_train_params = dict()

    # preprocessing policies
    transform_train = getattr(
        __import__("src.augmentation.policies", fromlist=[""]),
        transform_train,
    )(**transform_train_params)
    transform_test = getattr(
        __import__("src.augmentation.policies", fromlist=[""]),
        transform_test,
    )(**transform_test_params)

    # pytorch dataset
    Dataset = getattr(__import__("torchvision.datasets", fromlist=[""]), dataset_name)
    trainset = Dataset(
        root="save/data", train=True, download=True, transform=transform_train
    )
    testset = Dataset(
        root="save/data", train=False, download=False, transform=transform_test
    )

    return trainset, testset


def get_dataloader(
    trainset: VisionDataset,
    testset: VisionDataset,
    batch_size: int,
    n_workers: int,
) -> Tuple[data.DataLoader, data.DataLoader]:
    """Get dataloader for training and testing."""
    trainloader = data.DataLoader(
        trainset,
        pin_memory=(torch.cuda.is_available()),
        num_workers=n_workers,
        shuffle=True,
        batch_size=batch_size,
    )
    testloader = data.DataLoader(
        testset,
        pin_memory=(torch.cuda.is_available()),
        num_workers=n_workers,
        shuffle=False,
        batch_size=batch_size,
    )
    return trainloader, testloader


def get_latest_file(filepath: str, pattern: str = "*") -> str:
    """Get the latest file from the input filepath."""
    filelist = glob.glob(os.path.join(filepath, pattern))
    return max(filelist, key=os.path.getctime) if filelist else ""


def set_logger(
    filename: str,
    mode: str = "a",
    level: int = logging.DEBUG,
    maxbytes: int = 1024 * 1024 * 10,  # default: 10Mbyte
    backupcnt: int = 100,
) -> None:
    """Create and get the logger for the console and files."""
    logger = logging.getLogger("model_compression")
    logger.setLevel(level)

    chdlr = logging.StreamHandler(sys.stdout)
    chdlr.setLevel(logging.DEBUG)
    cfmts = "%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s"
    chdlr.setFormatter(logging.Formatter(cfmts))
    logger.addHandler(chdlr)

    fhdlr = logging.handlers.RotatingFileHandler(
        filename, mode=mode, maxBytes=maxbytes, backupCount=backupcnt
    )
    fhdlr.setLevel(logging.DEBUG)
    ffmts = "%(asctime)s - "
    ffmts += "%(processName)s - %(threadName)s - "
    ffmts += "%(filename)s:%(lineno)d - %(levelname)s - %(message)s"
    fhdlr.setFormatter(logging.Formatter(ffmts))
    logger.addHandler(fhdlr)


def get_logger() -> logging.Logger:
    """Get logger instance."""
    return logging.getLogger("model_compression")
