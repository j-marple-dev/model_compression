# -*- coding: utf-8 -*-
"""Runner for model compression.

- Author: Junghoon Kim
- Email: jhkim@jmarple.ai
"""


import logging
import os
from typing import Any, Dict

import torch.nn.utils.prune as prune

from src.train import Trainer
import src.utils as utils


class Pruner:
    """Pruner for models."""

    def __init__(
        self,
        trainer: Trainer,
        config: Dict[str, Any],
        dir_prefix: str,
        logger: logging.Logger,
    ) -> None:
        """Initialize."""
        self.trainer = trainer
        self.config = config
        self.logger = logger

        # create a dir for checkpoints
        ckpnt_dir = f"{self.config['N_PRUNING_ITER']}_times_pruning"
        self.root_dir = os.path.join(dir_prefix, ckpnt_dir)
        if not os.path.exists(self.root_dir):
            os.mkdir(self.root_dir)

    def run(self) -> None:
        """Run pruning."""
        params_to_prune = utils.get_weight_tuple(self.trainer.model, bias=False)
        params_all = utils.get_weight_tuple(self.trainer.model, bias=True)

        for i in range(self.config["N_PRUNING_ITER"]):
            self.logger.info(f"Pruning Iter: [{i} | {self.config['N_PRUNING_ITER']-1}]")
            # prune
            prune.global_unstructured(
                params_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=self.config["PRUNE_AMOUNT"],
            )
            sparsity = utils.get_sparsity(self.trainer.model, params_all)

            # train
            model_dir = f"{(sparsity):.2f}_sparsity".replace(".", "_")
            self.trainer.reset(self.root_dir, model_dir)
            self.trainer.run({"sparsity": sparsity})

            # logging
            self.logger.info(
                f"Pruning epoch: [{i} | {self.config['N_PRUNING_ITER']-1}] "
                f"Sparsity: {sparsity:.2f} Best acc: {self.trainer.best_acc}"
            )
