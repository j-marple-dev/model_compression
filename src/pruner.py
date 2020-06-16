# -*- coding: utf-8 -*-
"""Runner for model compression.

- Author: Junghoon Kim
- Email: jhkim@jmarple.ai
"""


import logging
import os
from typing import Any, Dict, Tuple

import torch
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

        self.params_to_prune = utils.get_weight_tuple(self.model, bias=False)
        self.params_all = utils.get_weight_tuple(self.model, bias=True)

        # create a dir for checkpoints
        ckpnt_dir = f"{self.config['N_PRUNING_ITER']}_times_pruning"
        self.root_dir = os.path.join(dir_prefix, ckpnt_dir)
        if not os.path.exists(self.root_dir):
            os.mkdir(self.root_dir)

    def reset(self, prune_iter: int) -> Tuple[int, float]:
        """Reset the processes for pruning or pretraining."""
        # pretraining
        if prune_iter == -1:
            start_from = 0
            sparsity = 0.0
            dir_prefix = self.root_dir
            model_dir = "pretrain"
            logger.info("Initialized Pretraining Settings")
        # pruning
        else:
            start_from = self.config["PRUNE_START_FROM"]
            dir_prefix = self.dir_prefix

            # prune
            prune.global_unstructured(
                self.params_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=self.config["PRUNE_AMOUNT"],
            )
            sparsity = utils.get_sparsity(self.model, self.params_all)
            model_dir = f"{(sparsity):.2f}_sparsity".replace(".", "_")
            logger.info(
                "Initialized Pruning Settings: "
                f"[{prune_iter} | {self.config['N_PRUNING_ITER']-1}]"
            )

        # train
        self.trainer.reset(dir_prefix, model_dir)

        return start_from, sparsity

    def run(self) -> None:
        """Run pruning."""
        if not self.init_params_path and self.config["STORE_PARAM_BEFORE"] == 0:
            self.save_init_params()

        # pretraining and pruning iteration
        for i in range(-1, self.config["N_PRUNING_ITER"]):
            start_from, sparsity = self.reset(i)

            # initialize training
            if self.init_params_path:
                self.trainer.load_params(self.init_params_path, with_mask=False)

            for epoch in range(start_from, self.config["EPOCHS"]):
                log_info = [("Sparsity", sparsity, percent_format)]
                self.trainer.run_one_epoch(epoch, log_info)

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
