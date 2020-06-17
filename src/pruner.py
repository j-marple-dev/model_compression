# -*- coding: utf-8 -*-
"""Runner for model compression.

- Author: Junghoon Kim
- Email: jhkim@jmarple.ai
"""


import logging
import os
from typing import Any, Callable, Dict, List, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

from src.format import percent_format
from src.trainer import Trainer
import src.utils as utils

logger = logging.getLogger()


class Pruner:
    """Pruner for models."""

    def __init__(
        self,
        config: Dict[str, Any],
        dir_prefix: str,
        wandb_log: bool,
        wandb_init_params: Dict[str, Any],
        device: torch.device,
    ) -> None:
        """Initialize."""
        self.config = config
        self.dir_prefix = dir_prefix
        self.wandb_log = wandb_log
        self.init_params_path = ""

        # create an initial model
        self.trainer = Trainer(
            config=config,
            dir_prefix=dir_prefix,
            model_dir="pretrain",
            device=device,
            wandb_log=wandb_log,
            wandb_init_params=wandb_init_params,
        )
        self.model = self.trainer.model

        self.params_to_prune = self.get_weight_tuple(bias=False)
        self.params_all = self.get_weight_tuple(bias=True)
        self.param_names = self.get_param_names()

    def reset(
        self, prune_iter: int
    ) -> Tuple[int, List[Tuple[str, float, Callable[[float], str]]]]:
        """Reset the processes for pruning or pretraining.

        Args:
            prune_iter (int): the next pruning iteration.

        Returns:
            int: the starting epoch of training (rewinding point for pruning).
            List[Tuple[str, float, Callable[[float], str]]]: logging information for sparsity,
                which consists of key, value, and formatting function.

        """
        # pretraining
        if prune_iter == -1:
            start_from = 0
            sparsity = 0.0
            mask_sparsity = 0.0
            conv_sparsity = 0.0
            fc_sparsity = 0.0

            # directory names for checkpionts
            model_dir = "pretrain"
            logger.info("Initialized Pretraining Settings")
        # pruning
        else:
            start_from = self.config["PRUNE_START_FROM"]

            # prune
            prune.global_unstructured(
                self.params_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=self.config["PRUNE_AMOUNT"],
            )

            # sparsities
            sparsity = self.sparsity()
            mask_sparsity = self.mask_sparsity()
            conv_sparsity = self.sparsity(module_name="Conv")
            fc_sparsity = self.sparsity(module_name="Linear")

            # directory name for checkpoints
            model_dir = f"{(sparsity):.2f}_pruned".replace(".", "_")
            logger.info(
                "Initialized Pruning Settings: "
                f"[{prune_iter} | {self.config['N_PRUNING_ITER']-1}]"
            )

        # reset trainer
        self.trainer.reset(self.dir_prefix, model_dir)

        # sparsity info for logging
        sparsity_info = []
        sparsity_info.append(("sparsity/total", sparsity, percent_format))
        sparsity_info.append(("sparsity/mask", mask_sparsity, percent_format))
        sparsity_info.append(("sparsity/conv", conv_sparsity, percent_format))
        sparsity_info.append(("sparsity/fc", fc_sparsity, percent_format))

        return start_from, sparsity_info

    def run(self) -> None:
        """Run pruning."""
        if self.config["STORE_PARAM_BEFORE"] == 0:
            self.save_init_params()

        # pretraining and pruning iteration
        for i in range(-1, self.config["N_PRUNING_ITER"]):
            start_from, sparsity_info = self.reset(i)

            # initialize training
            if self.init_params_path:
                self.trainer.load_params(self.init_params_path, with_mask=False)

            for epoch in range(start_from, self.config["EPOCHS"]):
                self.trainer.run_one_epoch(epoch, sparsity_info)

                # store initial weights
                if self.config["STORE_PARAM_BEFORE"] - 1 == epoch:
                    self.save_init_params()

    def save_init_params(self) -> None:
        """Set initial weights."""
        filename = "init_weight"
        self.trainer.save_params(
            self.dir_prefix, filename, self.config["STORE_PARAM_BEFORE"] - 1,
        )
        logger.info("Stored initial weights")
        self.init_params_path = os.path.join(self.dir_prefix, f"{filename}.pth.tar")

    def sparsity(self, module_name: str = "") -> float:
        """Get the proportion of zeros in weights (default: model's sparsity)."""
        n_zero = n_total = 0

        for w in self.params_all:
            if module_name not in str(w[0]):
                continue
            n_zero += int(torch.sum(getattr(w[0], w[1]) == 0.0).item())
            n_total += getattr(w[0], w[1]).nelement()

        return (100.0 * n_zero / n_total) if n_total != 0 else 0.0

    def mask_sparsity(self) -> float:
        """Get the ratio of zeros in weight masks."""
        n_zero = n_total = 0

        for w in self.param_names:
            param_instance = eval("self.model." + utils.dot2bracket(w) + ".weight_mask")
            n_zero += int(torch.sum(param_instance == 0.0).item())
            n_total += param_instance.nelement()

        return (100.0 * n_zero / n_total) if n_total != 0 else 0.0

    def get_weight_tuple(self, bias: bool = False) -> Tuple[Tuple[nn.Module, str], ...]:
        """Get weight and bias tuples for pruning."""
        t = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            layer_name, weight_type = name.rsplit(".", 1)
            if weight_type == "weight" or (bias and weight_type == "bias"):
                t.append(
                    (eval("self.model." + utils.dot2bracket(layer_name)), weight_type)
                )
        return tuple(t)

    def get_param_names(self) -> Set[str]:
        """Get param names in the model."""
        t = set()
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            layer_name, weight_type = name.rsplit(".", 1)
            t.add(layer_name)
        return t


def run(
    config: Dict[str, Any],
    dir_prefix: str,
    device: torch.device,
    wandb_log: bool,
    wandb_init_params: Dict[str, Any],
) -> None:
    """Run pruning process."""
    pruner = Pruner(
        config=config,
        dir_prefix=dir_prefix,
        wandb_log=wandb_log,
        wandb_init_params=wandb_init_params,
        device=device,
    )
    pruner.run()
