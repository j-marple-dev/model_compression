# -*- coding: utf-8 -*-
"""Utils for model compression.

- Author: Curt-Park
- Email: jwpark@jmarple.ai
"""

import glob
import os
import random
import re
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
    getter = "get_model"
    constructor = getattr(
        __import__("src.models." + model_name, fromlist=[model_name]), getter
    )
    return constructor(**model_config)


def save_checkpoint(state: Dict[str, Any], path: str, filename: str) -> None:
    """Save checkpoint including """
    filepath = os.path.join(path, filename)
    torch.save(state, filepath)


def get_latest_file(filepath: str) -> str:
    """Get the latest file from the input filepath."""
    return max(glob.glob(filepath + "/*"), key=os.path.getctime)


def get_weight_tuple(
    model: nn.Module, bias: bool = False
) -> Tuple[Tuple[nn.Module, str], ...]:
    """Get weight and bias tuples for pruning."""
    t = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            layer_name, weight_type = name.rsplit(".", 1)
            if weight_type == "weight" or (bias and weight_type == "bias"):
                t.append((eval("model." + dot2bracket(layer_name)), weight_type))
    return tuple(t)


def wlog_pruned_weight(model: nn.Module) -> None:
    """Log pruned (masked) weights only on wandb."""
    w_dict = dict()
    for name, param in model.named_parameters():
        if param.requires_grad:
            layer_name, weight_type = name.rsplit(".", 1)
            if weight_type in ("weight", "bias", "weight_orig"):
                postfix = "." + weight_type
                module_name = "model." + dot2bracket(layer_name) + postfix
                weight_cpy = eval(module_name).cpu().data.numpy()
            else:
                continue

            # for getting masked weights
            if weight_type == "weight_orig":
                postfix = ".weight"
                module_name = "model." + dot2bracket(layer_name) + ".named_buffers()"
                weight_mask = list(eval(module_name))[0][1].cpu().data.numpy()
                weight_cpy = weight_cpy[np.where(weight_mask == 1.0)]

            key = "pruned/" + layer_name + postfix
            w_dict.update({key: wandb.Histogram(weight_cpy)})
    wandb.log(w_dict, commit=False)


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
    """
    pattern = r"\.[0-9]*\."
    s_list = list(s)
    for m in re.finditer(pattern, s):
        start, end = m.span()
        s_list[start], s_list[end - 1] = "[", "]."
    return "".join(s_list)


def get_sparsity(model: nn.Module, w_tuple: Tuple[Tuple[nn.Module, str], ...]) -> float:
    """Get the proportion of zeros in weights."""
    zero_element = 0
    total_element = 0

    for w in w_tuple:
        zero_element += torch.sum((getattr(w[0], w[1]) == 0.0).int()).item()  # type: ignore
        total_element += getattr(w[0], w[1]).nelement()

    return 100.0 * float(zero_element) / float(total_element)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
