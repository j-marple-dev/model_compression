# -*- coding: utf-8 -*-
"""Runner for making a sparse model by weight pruning.

- Author: Junghoon Kim
- Email: jhkim@jmarple.ai
"""


import os
from typing import Any, Callable, Dict, List, Tuple

import torch
import torch.nn.utils.prune as prune

from src.format import percent_format
from src.models import utils as model_utils
from src.pruning.abstract import Abstract
from src.pruning.trainer import Trainer
from src.utils import get_logger

logger = get_logger()


class Pruner(Abstract):
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
        super(Pruner, self).__init__(config, dir_prefix)
        self.wandb_log = wandb_log
        self.pretrain_dir_name = "pretrain"
        self.dir_postfix = "pruned"
        self.init_params_name = "init_params"
        self.init_params_path = ""

        # create an initial model
        self.trainer = Trainer(
            config=self.config["TRAIN_CONFIG"],
            dir_prefix=dir_prefix,
            checkpt_dir=self.pretrain_dir_name,
            device=device,
            wandb_log=wandb_log,
            wandb_init_params=wandb_init_params,
        )
        self.model = self.trainer.model
        self.params_to_prune = model_utils.get_weight_tuple(self.model, bias=False)
        self.params_all = model_utils.get_weight_tuple(self.model, bias=True)

    def reset(
        self, prune_iter: int, resumed: bool = False,
    ) -> Tuple[int, List[Tuple[str, float, Callable[[float], str]]]]:
        """Reset the processes for pruning or pretraining.

        Args:
            prune_iter (int): the next pruning iteration.
            resumed (bool): has True if it is resumed.

        Returns:
            int: the starting epoch of training (rewinding point for pruning).
            List[Tuple[str, float, Callable[[float], str]]]: logging information for sparsity,
                which consists of key, value, and formatting function.

        """
        # pretraining
        if prune_iter == -1:
            start_epoch = 0
            sparsity = 0.0
            mask_sparsity = 0.0
            conv_sparsity = 0.0
            fc_sparsity = 0.0

            # directory names for checkpionts
            checkpt_dir = self.pretrain_dir_name
            logger.info("Initialized Pretraining Settings")

            # store initial weights
            if not resumed and self.config["STORE_PARAM_BEFORE"] == 0:
                self.save_init_params()
        # pruning
        else:
            start_epoch = self.config["PRUNE_START_FROM"]

            # prune
            if not resumed:
                prune.global_unstructured(
                    self.params_to_prune,
                    pruning_method=prune.L1Unstructured,
                    amount=self.config["PRUNE_AMOUNT"],
                )

            # sparsities
            sparsity = model_utils.sparsity(self.params_all)
            conv_sparsity = model_utils.sparsity(self.params_all, module_name="Conv")
            fc_sparsity = model_utils.sparsity(self.params_all, module_name="Linear")
            mask_sparsity = model_utils.mask_sparsity(self.model)

            # directory name for checkpoints
            checkpt_dir = f"{prune_iter}_{(sparsity):.2f}_{self.dir_postfix}".replace(
                ".", "_"
            )
            logger.info(
                "Initialized Pruning Settings: "
                f"[{prune_iter} | {self.config['N_PRUNING_ITER']-1}]"
            )

            # initialize trainer
            if not resumed and self.init_params_path:
                self.trainer.load_params(self.init_params_path, with_mask=False)

        # reset trainer
        self.trainer.reset(checkpt_dir)

        # sparsity info for logging
        sparsity_info = []
        sparsity_info.append(("sparsity/total", sparsity, percent_format))
        sparsity_info.append(("sparsity/mask", mask_sparsity, percent_format))
        sparsity_info.append(("sparsity/conv", conv_sparsity, percent_format))
        sparsity_info.append(("sparsity/fc", fc_sparsity, percent_format))

        return start_epoch, sparsity_info

    def resume(self) -> int:
        """Setting to resume the training."""
        # check if there is a saved initial parameters
        init_params_path = os.path.join(
            self.dir_prefix, f"{self.init_params_name}.{self.fileext}"
        )
        if os.path.exists(init_params_path):
            self.init_params_path = init_params_path

        # check the pruning iteration
        last_iter = self._check_pruning_iter_from_filepath()

        # model should contain weight_mask
        if last_iter > -1:
            model_utils.dummy_pruning(self.model)

        return last_iter

    def run(self, resume_info_path: str = "") -> None:
        """Run pruning."""
        # resume pruner if needed
        start_iter, epoch_to_resume = -1, 0
        if resume_info_path:
            start_iter = self.resume()
            epoch_to_resume = self.trainer.resume()
            logger.info("Run one epoch for warming-up")
            self.trainer.test_one_epoch()

        # pretraining and pruning iteration
        for i in range(start_iter, self.config["N_PRUNING_ITER"]):
            start_epoch, sparsity_info = self.reset(i, epoch_to_resume > 0)

            # if there is a valid file to resume
            if start_epoch < epoch_to_resume:
                start_epoch = epoch_to_resume
                epoch_to_resume = 0

            for epoch in range(start_epoch, self.config["EPOCHS"]):
                self.trainer.run_one_epoch(epoch, sparsity_info)

                # store weights with warmup
                if self.config["STORE_PARAM_BEFORE"] - 1 == epoch:
                    self.save_init_params()

            if i == -1:
                logger.info("Pretraining Done")
            else:
                logger.info(f"Pruning Done: [{i} | {self.config['N_PRUNING_ITER']-1}]")

    def save_init_params(self) -> None:
        """Set initial weights."""
        self.trainer.save_params(
            self.dir_prefix,
            self.init_params_name,
            self.config["STORE_PARAM_BEFORE"] - 1,
            record_path=False,
        )
        logger.info("Stored initial parameters")
        self.init_params_path = os.path.join(
            self.dir_prefix, f"{self.init_params_name}.{self.fileext}"
        )

    def _check_pruning_iter_from_filepath(self) -> int:
        """Check the last pruning iteration from filepath."""
        last_iter = -1
        latest_file_path = self._fetch_latest_checkpt()

        if latest_file_path and os.path.exists(latest_file_path):
            logger.info(f"Resume pruning from {self.dir_prefix}")
            _, checkpt_dir, _ = latest_file_path.rsplit(os.path.sep, 2)

            # fetch the last iter from the filename
            if checkpt_dir != self.pretrain_dir_name:
                last_iter = int(checkpt_dir.split("_", 1)[0])

        return last_iter
