# -*- coding: utf-8 -*-
"""Abstract Runner class which contains methods to implement.

- Author: Curt-Park
- Email: jwpark@jmarple.ai
"""

from abc import ABC, abstractmethod
import os
from typing import Any, Dict, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
import torch.utils.data as data
import wandb

from src.utils import dot2bracket


class Runner(ABC):
    """Abstract class used by all runner modules."""

    def __init__(self, config: Dict[str, Any], dir_prefix: str) -> None:
        """Initialize."""
        self.config = config
        self.dir_prefix = dir_prefix
        self.fileext = "pth.tar"
        self.checkpt_paths = "checkpt_paths.log"

    @abstractmethod
    def run(self) -> None:
        """Run the module."""
        pass

    @abstractmethod
    def resume(self, resume_info_path: str) -> int:
        """Setting to resume the training."""
        pass

    def get_model(self) -> nn.Module:
        """Get PyTorch model."""
        model_name = self.config["MODEL_NAME"]
        model_config = self.config["MODEL_PARAMS"]

        # get model constructor
        return __import__("src.models." + model_name, fromlist=[model_name]).get_model(
            **model_config
        )

    def get_dataset(
        self,
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

    def initialize_params(
        self,
        model: Union[nn.Module, Optimizer],
        state_dict: Dict[str, Any],
        with_mask=True,
    ) -> None:
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

    def wlog_weight(self, model: nn.Module) -> None:
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

    def _fetch_latest_checkpt(self) -> str:
        """Fetch the latest checkpoint file path from the log file."""
        checkpt_paths = os.path.join(self.dir_prefix, self.checkpt_paths)
        if not os.path.exists(checkpt_paths):
            return ""
        latest_file_path = ""
        with open(checkpt_paths, "r") as checkpts:
            checkpts_list = checkpts.readlines()
            if checkpts_list:
                latest_file_path = checkpts_list[-1][:-1]  # w/o '\n'
        return latest_file_path
