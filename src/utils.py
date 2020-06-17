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
import re
import sys
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import wandb


def set_random_seed(seed: int):
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
    dataset = getattr(
        __import__("torchvision.datasets", fromlist=["datasets"]), dataset_name
    )

    # dataloader for training
    transform_train = getattr(
        __import__("src.augment", fromlist=["augment"]), transform_train
    )()
    trainset = dataset(
        root="save/data", train=True, download=True, transform=transform_train,
    )
    trainloader = data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=n_workers,
    )

    # dataloader for testing
    transform_test = getattr(
        __import__("src.augment", fromlist=["augment"]), transform_test
    )()
    testset = dataset(
        root="save/data", train=False, download=False, transform=transform_test,
    )
    testloader = data.DataLoader(
        testset, batch_size=batch_size, shuffle=True, num_workers=n_workers,
    )

    return trainloader, testloader


def get_model(model_name: str, model_config: Dict[str, Any]) -> nn.Module:
    """Get PyTorch model."""
    # get model constructor
    return __import__("src.models." + model_name, fromlist=[model_name]).get_model(
        **model_config
    )


def set_logger(
    filename: str,
    mode: str = "a",
    level: int = logging.DEBUG,
    maxbytes: int = 1024 * 1024 * 10,  # default: 10Mbyte
    backupcnt: int = 100,
) -> None:
    """Create and get the logger for the console and files."""
    logger = logging.getLogger()
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


def save_checkpoint(state: Dict[str, Any], path: str, filename: str) -> None:
    """Save checkpoint including """
    filepath = os.path.join(path, filename)
    torch.save(state, filepath)


def get_latest_file(filepath: str) -> str:
    """Get the latest file from the input filepath."""
    return max(glob.glob(filepath + "/*"), key=os.path.getctime)


def wlog_weight(model: nn.Module) -> None:
    """Log weights on wandb."""
    wlog = dict()
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        layer_name, weight_type = name.rsplit(".", 1)

        # get params(weight, bias, weight_orig)
        if weight_type in ("weight", "bias", "weight_orig"):
            w_name = "params/" + layer_name + "." + weight_type
            weight = eval("model." + dot2bracket(layer_name) + "." + weight_type)
            weight = weight.cpu().data.numpy()
            wlog.update({w_name: wandb.Histogram(weight)})
        else:
            continue

        # get masked weights
        if weight_type == "weight_orig":
            w_name = "params/" + layer_name + ".weight"
            named_buffers = eval(
                "model." + dot2bracket(layer_name) + ".named_buffers()"
            )
            mask: Tuple[str, torch.Tensor] = next(
                x for x in list(named_buffers) if x[0] == "weight_mask"
            )[1].cpu().data.numpy()
            masked_weight = weight[np.where(mask == 1.0)]
            wlog.update({w_name: wandb.Histogram(masked_weight)})
    wandb.log(wlog, commit=False)


def dot2bracket(s: str) -> str:
    """Replace layer names with valid names for pruning.

    Test:
       >>> dot2bracket("dense2.1.bn1.bias")
       'dense2[1].bn1.bias'
       >>> dot2bracket("dense2.13.bn1.bias")
       'dense2[13].bn1.bias'
       >>> dot2bracket("conv2.123.bn1.bias")
       'conv2[123].bn1.bias'
       >>> dot2bracket("dense2.6.conv2.5.bn1.bias")
       'dense2[6].conv2[5].bn1.bias'
       >>> dot2bracket("model.6")
       'model[6]'
       >>> dot2bracket("vgg.2.conv2.bn.2")
       'vgg[2].conv2.bn[2]'
       >>> dot2bracket("features.11")
       'features[11]'
    """
    pattern = r"\.[0-9]+"
    s_list = list(s)
    for m in re.finditer(pattern, s):
        start, end = m.span()
        s_list[start] = "["
        if end < len(s) and s_list[end] == ".":
            s_list[end] = "]."
        else:
            s_list.insert(end, "]")
    return "".join(s_list)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
