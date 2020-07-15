# -*- coding: utf-8 -*-
"""Runner for making a sparse model by weight pruning.

- Author: Junghoon Kim
- Email: jhkim@jmarple.ai
"""

import abc
import os
from typing import Any, Callable, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

from src.format import percent_format
from src.models import utils as model_utils
from src.runners.runner import Runner
from src.runners.trainer import Trainer
from src.utils import get_logger

logger = get_logger()


class Pruner(Runner):
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

        self.params_all = model_utils.get_params(
            self.model,
            (
                (nn.Conv2d, "weight"),
                (nn.Conv2d, "bias"),
                (nn.BatchNorm2d, "weight"),
                (nn.BatchNorm2d, "bias"),
                (nn.Linear, "weight"),
                (nn.Linear, "bias"),
            ),
        )
        self.params_to_prune = self.get_params_to_prune()

        # to calculate sparsity properly
        prune.global_unstructured(
            self.params_all, pruning_method=prune.L1Unstructured, amount=0.0,
        )

    @abc.abstractmethod
    def prune_params(self, prune_iter: int) -> None:
        """Run pruning."""
        raise NotImplementedError

    @abc.abstractmethod
    def get_params_to_prune(self) -> Tuple[Tuple[nn.Module, str], ...]:
        """Get parameters to prune."""
        raise NotImplementedError

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
            mask_total_sparsity = zero_total_sparsity = 0
            mask_conv_sparsity = zero_conv_sparsity = 0
            mask_fc_sparsity = zero_fc_sparsity = 0
            mask_bn_sparsity = zero_bn_sparsity = 0

            # directory names for checkpionts
            checkpt_dir = self.pretrain_dir_name
            logger.info("Initialized Pretraining Settings")

            # store initial weights
            if not resumed and self.config["STORE_PARAM_BEFORE"] == 0:
                self.save_init_params()
        # pruning
        else:
            start_epoch = self.config["PRUNE_START_FROM"]

            if not resumed:
                self.prune_params(prune_iter)
            logger.info("Forward model for one iter to warmup")
            self.trainer.warmup_one_iter()

            # sparsities
            zero_total_sparsity = model_utils.sparsity(self.params_all)
            zero_conv_sparsity = model_utils.sparsity(
                self.params_all, module_types=(nn.Conv2d,)
            )
            zero_fc_sparsity = model_utils.sparsity(
                self.params_all, module_types=(nn.Linear,)
            )
            zero_bn_sparsity = model_utils.sparsity(
                self.params_all, module_types=(nn.BatchNorm2d,)
            )
            mask_total_sparsity = model_utils.mask_sparsity(self.params_all)
            mask_conv_sparsity = model_utils.mask_sparsity(
                self.params_all, module_types=(nn.Conv2d,)
            )
            mask_fc_sparsity = model_utils.mask_sparsity(
                self.params_all, module_types=(nn.Linear,)
            )
            mask_bn_sparsity = model_utils.mask_sparsity(
                self.params_all, module_types=(nn.BatchNorm2d,)
            )

            # directory name for checkpoints
            checkpt_dir = f"{prune_iter}_{(mask_total_sparsity):.2f}_"
            f"{self.dir_postfix}".replace(".", "_")
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
        sparsity_info: List[Tuple[str, float, Callable]] = []
        sparsity_info.append(
            ("zero_sparsity/total", zero_total_sparsity, percent_format)
        )
        sparsity_info.append(("zero_sparsity/conv", zero_conv_sparsity, percent_format))
        sparsity_info.append(("zero_sparsity/fc", zero_fc_sparsity, percent_format))
        sparsity_info.append(("zero_sparsity/bn", zero_bn_sparsity, percent_format))
        sparsity_info.append(
            ("mask_sparsity/total", mask_total_sparsity, percent_format)
        )
        sparsity_info.append(("mask_sparsity/conv", mask_conv_sparsity, percent_format))
        sparsity_info.append(("mask_sparsity/fc", mask_fc_sparsity, percent_format))
        sparsity_info.append(("mask_sparsity/bn", mask_bn_sparsity, percent_format))
        sparsity_info.append(
            (
                "mask_sparsity/conv_target",
                self.get_target_sparsity(prune_iter) * 100.0,
                percent_format,
            )
        )

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

        return last_iter

    def run(self, resume_info_path: str = "") -> None:
        """Run pruning."""
        # resume pruner if needed
        start_iter, epoch_to_resume = -1, 0
        if resume_info_path:
            start_iter = self.resume()
            epoch_to_resume = self.trainer.resume()

        for prune_iter in range(start_iter, self.config["N_PRUNING_ITER"]):
            start_epoch, sparsity_info = self.reset(prune_iter, epoch_to_resume > 0)

            # if there is a valid file to resume
            if start_epoch < epoch_to_resume:
                start_epoch = epoch_to_resume
                epoch_to_resume = 0

            for epoch in range(start_epoch, self.config["EPOCHS"]):
                self.trainer.run_one_epoch(epoch, sparsity_info)

                # store weights with warmup
                if self.config["STORE_PARAM_BEFORE"] - 1 == epoch:
                    self.save_init_params()

            if prune_iter == -1:
                logger.info("Pretraining Done")
            else:
                logger.info(
                    f"Pruning Done: [{prune_iter} | {self.config['N_PRUNING_ITER']-1}]"
                )

    def get_target_sparsity(self, prune_iter: int) -> float:
        """Get target sparsity for current prune epoch."""
        target_density = 1.0
        for _ in range(prune_iter + 1):
            target_density = target_density * (1 - self.config["PRUNE_AMOUNT"])
        return 1 - target_density

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


class LotteryTicketHypothesis(Pruner):
    """LTH on whole layer.

    Reference:
        The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks
        (https://arxiv.org/pdf/1803.03635.pdf)
        Stabilizing the Lottery Ticket Hypothesis
        (https://arxiv.org/pdf/1903.01611.pdf)
        Comparing Rewinding and Fine-tuning in Neural Network Pruning
        (https://arxiv.org/pdf/2003.02389.pdf)
    """

    def __init__(
        self,
        config: Dict[str, Any],
        dir_prefix: str,
        wandb_log: bool,
        wandb_init_params: Dict[str, Any],
        device: torch.device,
    ) -> None:
        """Initialize."""
        super(LotteryTicketHypothesis, self).__init__(
            config, dir_prefix, wandb_log, wandb_init_params, device
        )

    def get_params_to_prune(self) -> Tuple[Tuple[nn.Module, str], ...]:
        """Get parameters to prune."""
        return model_utils.get_params(
            self.model,
            ((nn.Conv2d, "weight"), (nn.BatchNorm2d, "weight"), (nn.Linear, "weight")),
        )

    def prune_params(self, prune_iter: int) -> None:
        """Apply prune."""
        prune.global_unstructured(
            self.params_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=self.config["PRUNE_AMOUNT"],
        )


class LotteryTicketHypothesisFC(LotteryTicketHypothesis):
    """LTH on fc layer only.

    Reference:
        The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks
        (https://arxiv.org/pdf/1803.03635.pdf)
        Stabilizing the Lottery Ticket Hypothesis
        (https://arxiv.org/pdf/1903.01611.pdf)
        Comparing Rewinding and Fine-tuning in Neural Network Pruning
        (https://arxiv.org/pdf/2003.02389.pdf)
    """

    def __init__(
        self,
        config: Dict[str, Any],
        dir_prefix: str,
        wandb_log: bool,
        wandb_init_params: Dict[str, Any],
        device: torch.device,
    ) -> None:
        """Initialize."""
        super(LotteryTicketHypothesisFC, self).__init__(
            config, dir_prefix, wandb_log, wandb_init_params, device
        )
        self.params_to_prune = model_utils.get_params(
            self.model, ((nn.Linear, "weight"),)
        )


class NetworkSlimming(Pruner):
    """Network slimming(slim) on bn 2d.

    Reference:
        Learning Efficient Convolutional Networks through Network Slimming
        (https://arxiv.org/pdf/1708.06519.pdf)
    """

    def __init__(
        self,
        config: Dict[str, Any],
        dir_prefix: str,
        wandb_log: bool,
        wandb_init_params: Dict[str, Any],
        device: torch.device,
    ) -> None:
        """Initialize."""
        self._bn_to_conv: Dict[nn.BatchNorm2d, nn.Conv2d] = dict()
        super(NetworkSlimming, self).__init__(
            config, dir_prefix, wandb_log, wandb_init_params, device
        )

    @property
    def bn_to_conv(self) -> Dict[nn.BatchNorm2d, nn.Conv2d]:
        """Create bn_to_conv when it is needed."""
        if not self._bn_to_conv:
            self._bn_to_conv = self.get_bn_to_conv()
        return self._bn_to_conv

    def get_bn_to_conv(self) -> Dict[nn.BatchNorm2d, nn.Conv2d]:
        """Get a dictionary key: bacthnorm2d, val: conv2d."""
        layers = [
            v for v in self.model.modules() if type(v) in {nn.Conv2d, nn.BatchNorm2d}
        ]
        bn_to_conv = dict()
        for i in range(1, len(layers)):
            if type(layers[i - 1]) is nn.Conv2d and type(layers[i]) is nn.BatchNorm2d:
                bn_to_conv[layers[i]] = layers[i - 1]
        return bn_to_conv

    def get_params_to_prune(self) -> Tuple[Tuple[nn.Module, str], ...]:
        """Get parameters to prune."""
        # Cases there is no corresponding conv layer with bn
        all_bn_weights = model_utils.get_params(
            self.model, ((nn.BatchNorm2d, "weight"),)
        )
        return tuple(
            (module, name)
            for (module, name) in all_bn_weights
            if module in self.bn_to_conv
        )

    def prune_params(self, prune_iter: int) -> None:
        """Apply prune."""
        target_sparsity = self.get_target_sparsity(prune_iter)
        sparsity = model_utils.mask_sparsity(self.params_all, (nn.Conv2d,)) / 100
        prev_sparsity = sparsity
        base_amount = amount = 0.001

        params_to_prune = drop_overly_pruned(self.params_to_prune)
        # Give little margin to target_sparsity
        # target_sparsity never goes less than amount when used as intended
        while sparsity <= target_sparsity - amount:
            prune.global_unstructured(
                params_to_prune, pruning_method=prune.L1Unstructured, amount=amount,
            )
            self.update_masks()
            prev_sparsity = sparsity
            sparsity = model_utils.mask_sparsity(self.params_all, (nn.Conv2d,)) / 100
            if sparsity == prev_sparsity:
                amount += base_amount

        # DEBUG: Check all parameters to be pruned
        # TODO: Will be deprecated after it shows stable performance
        # t = nn.utils.parameters_to_vector([getattr(*p) for p in params_to_prune_constrained])
        # print('t', t)
        self.update_masks()

    def update_masks(self) -> None:
        """Copy bn weight masks to bias masks and conv masks"""
        for bn, conv in self.bn_to_conv.items():
            # Copy batchnorm weight_mask to bias_mask
            bn_buffers = {name: buf for name, buf in bn.named_buffers()}
            bn_mask = bn_buffers["weight_mask"].detach().clone()
            bn_buffers["bias_mask"].set_(bn_mask)

            conv_buffers = {name: buf for name, buf in conv.named_buffers()}
            if "bias_mask" in conv_buffers:
                conv_buffers["bias_mask"].set_(bn_mask)

            # conv2d - batchnorm - activation (CBA)
            # bn_mask: [out], conv: [out, in, h, w]
            [o, i, h, w] = conv_buffers["weight_mask"].shape

            # check shape -> if its not shaped as CBA
            if bn_mask.shape[0] != o:
                continue
            # bn_mask: [out, 1, 1]
            bn_mask = bn_mask.view(o, 1, 1)
            # bn_mask: [out, h, w]
            bn_mask = bn_mask.repeat(1, h, w)
            # bn_mask: [out, 1, h, w]
            bn_mask = bn_mask.unsqueeze(1)
            # bn_mask: [out, in, h, w]
            bn_mask = bn_mask.repeat(1, i, 1, 1)

            conv_buffers["weight_mask"].set_(bn_mask)

        # DEBUG: Check mask information
        # TODO: Will be deprecated after it shows stable performance
        """
        layer_dict = {k: v for k, v in self.model.named_modules()}
        bn_zero = bn_total = 0
        cnn_zero = cnn_total = 0
        for name, layer in layer_dict.items():
            if isinstance(layer, nn.Conv2d):
                z = int(torch.sum(getattr(layer, "weight_mask") == 0.0).item())
                b = int(torch.sum(getattr(layer, "bias_mask") == 0.0).item())
                t = getattr(layer, "weight_mask").nelement()
                print(f"{name} {z} {b} {t} {z/t*100:.2f}")
                cnn_zero += z
                cnn_total += t
            if isinstance(layer, nn.BatchNorm2d):
                z = int(torch.sum(getattr(layer, "weight_mask") == 0.0).item())
                b = int(torch.sum(getattr(layer, "bias_mask") == 0.0).item())
                t = getattr(layer, "weight_mask").nelement()
                print(f"{name} {z} {b} {t} {z/t*100:.2f}")
                bn_zero += z
                bn_total += t
        print(f"bn {bn_zero/bn_total*100:.2f}, cnn {cnn_zero/cnn_total*100:.2f}")
        """


def drop_overly_pruned(
    params: Tuple[Tuple[Any, str], ...]
) -> Tuple[Tuple[Any, str], ...]:
    """Exclude excessively pruned params to preserve flow"""
    params_to_prune = []
    for layer, layer_type in params:
        total = int(layer.weight_mask.nelement())
        non_zero = int((layer.weight_mask != 0).sum().detach())
        # prune only when preserved more than 20% of filters
        if non_zero / total > 0.2:
            params_to_prune.append((layer, layer_type))
    return tuple(params_to_prune)
