# -*- coding: utf-8 -*-
"""Utils for model compression.

- Author: Curt-Park
- Email: jwpark@jmarple.ai
"""

import glob
import hashlib
import logging
import logging.handlers
import os
import random
import re
import sys
import tarfile
from typing import Any, Dict, Tuple

import gdown
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import wandb
import yaml


def initialize_params(model: Any, state_dict: Dict[str, Any], with_mask=True) -> None:
    """Initialize weights and masks."""
    model_dict = model.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict = {}
    for k, v in state_dict.items():
        if k in model_dict and (with_mask or "weight_mask" not in k):
            pretrained_dict[k] = v
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)


def set_random_seed(seed: int) -> None:
    """Set random seed."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    # for CuDNN backend
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True  # type: ignore
        torch.backends.cudnn.benchmark = False  # type: ignore


def get_model_hash(model: nn.Module) -> str:
    """Get model info as hash."""
    return hashlib.sha224(str(model).encode("UTF-8")).hexdigest()


def get_pretrained_model_info(model: nn.Module) -> Dict[str, str]:
    """Read yaml file, get pretrained model information(model_dir, gdrive_link) \
        given hash."""
    model_hash = str(get_model_hash(model))
    with open("config/pretrained_model_url.yaml", mode="r") as f:
        model_info = yaml.load(f, Loader=yaml.FullLoader)[model_hash]
    return model_info


def download_pretrained_model(file_path: str, download_link: str) -> None:
    """Get pretrained model from google drive."""
    model_folder, model_name, file_name = file_path.rsplit(os.path.sep, 2)
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    # Download, unzip
    zip_file_path = os.path.join(model_folder, model_name + ".tar.xz")
    gdown.download(download_link, zip_file_path)
    with tarfile.open(zip_file_path, "r:*") as f:
        f.extractall(model_folder)


def get_model(model_name: str, model_config: Dict[str, Any]) -> nn.Module:
    """Get PyTorch model."""
    # get model constructor
    return __import__("src.models." + model_name, fromlist=[model_name]).get_model(
        **model_config
    )


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
        __import__("src.augmentation.policies", fromlist=["augment"]), transform_train,
    )()
    trainset = dataset(
        root="save/data", train=True, download=True, transform=transform_train,
    )
    trainloader = data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=n_workers,
    )

    # dataloader for testing
    transform_test = getattr(
        __import__("src.augmentation.policies", fromlist=["augment"]), transform_test,
    )()
    testset = dataset(
        root="save/data", train=False, download=False, transform=transform_test,
    )
    testloader = data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=n_workers,
    )

    return trainloader, testloader


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


def get_latest_file(filepath: str, pattern: str = "*") -> str:
    """Get the latest file from the input filepath."""
    filelist = glob.glob(os.path.join(filepath, pattern))
    return max(filelist, key=os.path.getctime) if filelist else ""


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


def dict2str(d: Dict[str, Any]) -> str:
    """config dict, oreder keys with lexicographical order, print out as string.
        . -> _

    e.g.) {"test":3.5,
           "more_test":'yes'}
           return 'more_test_yes_test_3_5'

    Test:
        >>> dict2str({"test":3.5, "more_test":'yes'})
        '_more_test_yes_test_3_5'
        >>> dict2str(dict(depth=100, num_classes=100, growthRate=12, compressionRate=2))
        '_compressionRate_2_depth_100_growthRate_12_num_classes_100'
        >>> dict2str(dict(num_classes=100, growthRate=12, compressionRate=2, depth=100))
        '_compressionRate_2_depth_100_growthRate_12_num_classes_100'
    """
    s = []
    for key in sorted(d):
        s.append("_")
        s.append(key)
        s.append("_")
        s.append(str(d[key]).replace(".", "_"))
    return "".join(s)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
