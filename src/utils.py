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
from typing import Tuple

import numpy as np
import torch
import torch.utils.data as data


def set_random_seed(seed: int) -> None:
    """Set random seed."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    # for CuDNN backend
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True  # type: ignore
        torch.backends.cudnn.benchmark = False  # type: ignore


def get_dataset(
    batch_size: int,
    n_workers: int,
    dataset_name: str = "CIFAR100",
    transform_train: str = "simple_augment_train_cifar100",
    transform_test: str = "simple_augment_test_cifar100",
) -> Tuple[data.DataLoader, data.DataLoader]:
    """Get dataset for training and testing."""
    # dataset
    dataset = getattr(__import__("torchvision.datasets", fromlist=[""]), dataset_name)

    # dataloader for training
    transform_train = getattr(
        __import__("src.augmentation.policies", fromlist=[""]), transform_train,
    )()
    trainset = dataset(
        root="save/data", train=True, download=True, transform=transform_train,
    )
    trainloader = data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=n_workers,
    )

    # dataloader for testing
    transform_test = getattr(
        __import__("src.augmentation.policies", fromlist=[""]), transform_test,
    )()
    testset = dataset(
        root="save/data", train=False, download=False, transform=transform_test,
    )
    testloader = data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=n_workers,
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
