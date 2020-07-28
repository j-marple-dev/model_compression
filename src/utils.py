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
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data
from torchvision.datasets import ImageFolder, VisionDataset


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
    crawl_ratio: float = 0.0,
) -> Tuple[List[VisionDataset], List[VisionDataset]]:
    """Get dataset for training and testing."""
    if not transform_train_params:
        transform_train_params = dict()

    # get transform
    transform_train = getattr(
        __import__("src.augmentation.policies", fromlist=[""]), transform_train,
    )(**transform_train_params)
    # test dataset
    transform_test = getattr(
        __import__("src.augmentation.policies", fromlist=[""]), transform_test,
    )(**transform_test_params)

    trainsets: List[VisionDataset] = []
    testsets: List[VisionDataset] = []

    trainset_args: Dict[str, Any] = dict(transform=transform_train)
    testset_args: Dict[str, Any] = dict(transform=transform_test)

    # challenge
    # Each dataset is described in the following link:
    # https://github.com/Curt-Park/model_compression/pull/70
    if dataset_name in "AI_CHALLENGE":
        dataset = ImageFolder
        file_path = f"save/data/{dataset_name.lower()}/total/"
        trainset_args.update(dict(root=file_path + "test/"))
        testset_args.update(dict(root=file_path + "test/"))

        # Future
        trainsets.append(dataset(**trainset_args))
        testsets.append(dataset(**testset_args))

        if crawl_ratio > 0.0:
            crawl_trainset_args: Dict[str, Any] = dict(transform=transform_train)
            crawl_testset_args: Dict[str, Any] = dict(transform=transform_test)
            crawl_trainset_args.update(dict(root=file_path + "crawl/train/"))
            crawl_testset_args.update(dict(root=file_path + "crawl/test/"))

            trainsets.append(dataset(**crawl_trainset_args))
            testsets.append(dataset(**crawl_testset_args))

    # pytorch dataset
    else:
        dataset = getattr(
            __import__("torchvision.datasets", fromlist=[""]), dataset_name
        )
        trainset_args.update(dict(root="save/data", train=True, download=True))
        testset_args.update(dict(root="save/data", train=False, download=False))

        trainsets.append(dataset(**trainset_args))
        testsets.append(dataset(**testset_args))

    return trainsets, testsets


def get_dataloader(
    trainsets: List[VisionDataset],
    testsets: List[VisionDataset],
    batch_size: int,
    n_workers: int,
    multidataloader_config: Dict[str, Any] = None,
) -> Tuple[List[data.DataLoader], List[data.DataLoader]]:
    """Get dataloader for training and testing."""
    trainloaders: List[data.DataLoader] = []
    testloaders: List[data.DataLoader] = []

    # dataloader config contains info for multi dataset
    if multidataloader_config:
        # get each batchsize for DataLoader
        crawl_ratio = multidataloader_config["crawl_ratio"]
        batch_sizes = [
            int(round(batch_size * (1 - crawl_ratio))),
            int(round(batch_size * (crawl_ratio))),
        ]
        batch_residual = sum(batch_sizes) - batch_size
        batch_sizes[0] -= batch_residual

        for train, test, bs in zip(trainsets, testsets, batch_sizes):
            train_args: Dict[str, Any] = dict({"shuffle": True, "batch_size": bs})
            if multidataloader_config["stratified_sample"]:
                indices = list(range(len(train)))
                # distribution of classes in the dataset
                label_to_count: Dict[str, int] = {}
                for idx in indices:
                    label = _get_label(train, idx)
                    if label in label_to_count:
                        label_to_count[label] += 1
                    else:
                        label_to_count[label] = 1

                # weight for each sample
                weights = [
                    1.0 / label_to_count[_get_label(train, idx)] for idx in indices
                ]
                weights = torch.DoubleTensor(weights)  # type: ignore
                sampler = torch.utils.data.sampler.WeightedRandomSampler(
                    weights=weights,
                    num_samples=int(bs * multidataloader_config["iter_per_epoch"]),
                    replacement=True,
                )
                # When sampler is specified, shuffle must be False
                train_args = {"sampler": sampler, "shuffle": False, "batch_size": bs}
            trainloaders.append(
                data.DataLoader(
                    train,
                    pin_memory=(torch.cuda.is_available()),
                    num_workers=n_workers,
                    **train_args,
                )
            )

            testloaders.append(
                data.DataLoader(
                    test,
                    batch_size=batch_size,
                    shuffle=False,
                    pin_memory=(torch.cuda.is_available()),
                    num_workers=n_workers,
                )
            )
    else:
        trainloaders.append(
            data.DataLoader(
                trainsets[0],
                batch_size=batch_size,
                shuffle=True,
                pin_memory=(torch.cuda.is_available()),
                num_workers=n_workers,
            )
        )
        testloaders.append(
            data.DataLoader(
                testsets[0],
                batch_size=batch_size,
                shuffle=False,
                pin_memory=(torch.cuda.is_available()),
                num_workers=n_workers,
            )
        )
    return trainloaders, testloaders


def _get_label(dataset: VisionDataset, idx: int):
    """Get label for diverse torchvision.datasets."""
    return dataset[idx][1]


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
