# -*- coding: utf-8 -*-
"""Runner for making a sparse model by weight pruning.

- Author: Junghoon Kim
- Email: jhkim@jmarple.ai
"""

import abc
import copy
import itertools
import os
from typing import Any, Callable, Dict, List, Set, Tuple, cast

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

from src.format import percent_format
from src.models import utils as model_utils
from src.models.adjmodule_getter import AdjModuleGetter
from src.plotter import Plotter
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
        self.device = device

        self.plotter = Plotter(self.wandb_log)

        # create an initial model
        self.trainer = Trainer(
            config=self.config["TRAIN_CONFIG"],
            dir_prefix=dir_prefix,
            checkpt_dir=self.pretrain_dir_name,
            wandb_log=wandb_log,
            wandb_init_params=wandb_init_params,
            device=device,
        )
        self.model = self.trainer.model

        self.model_params = model_utils.get_params(
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
        model_utils.dummy_pruning(self.model_params)
        model_utils.dummy_pruning(self.params_to_prune)

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
            if not resumed and self.config["PRUNE_PARAMS"]["STORE_PARAM_BEFORE"] == 0:
                self.save_init_params()
        # pruning
        else:
            start_epoch = self.config["PRUNE_PARAMS"]["PRUNE_START_FROM"]
            logger.info("Change train configuration for pruning.")
            if prune_iter == 0 or resumed:
                self.trainer.setup_train_configuration(
                    self.config["TRAIN_CONFIG_AT_PRUNE"]
                )

            if not resumed:
                # Prune with current best model
                if self.config["PRUNE_PARAMS"]["PRUNE_AT_BEST"]:
                    self.trainer.load_best_model()
                logger.info("Prune model")
                self.prune_params(prune_iter)
            logger.info("Forward model for one iter to warmup")
            self.trainer.warmup_one_iter()

            # sparsities
            zero_total_sparsity = model_utils.sparsity(self.model_params)
            zero_conv_sparsity = model_utils.sparsity(
                self.model_params, module_types=(nn.Conv2d,)
            )
            zero_fc_sparsity = model_utils.sparsity(
                self.model_params, module_types=(nn.Linear,)
            )
            zero_bn_sparsity = model_utils.sparsity(
                self.model_params, module_types=(nn.BatchNorm2d,)
            )
            mask_total_sparsity = model_utils.mask_sparsity(self.model_params)
            mask_conv_sparsity = model_utils.mask_sparsity(
                self.model_params, module_types=(nn.Conv2d,)
            )
            mask_fc_sparsity = model_utils.mask_sparsity(
                self.model_params, module_types=(nn.Linear,)
            )
            mask_bn_sparsity = model_utils.mask_sparsity(
                self.model_params, module_types=(nn.BatchNorm2d,)
            )

            # directory name for checkpoints
            checkpt_dir = f"{prune_iter}_"
            checkpt_dir += f"{(mask_total_sparsity):.2f}_".replace(".", "_")
            checkpt_dir += f"{self.dir_postfix}"

            logger.info(
                "Initialized Pruning Settings: "
                f"[{prune_iter} | {self.config['N_PRUNING_ITER']-1}]"
            )

            # initialize trainer
            if not resumed and self.init_params_path:
                self.trainer.load_params(self.init_params_path, with_mask=False)

        # reset trainer
        self.trainer.reset(checkpt_dir)

        # plot result
        self.plotter.plot(self.model, self.trainer.get_model_save_dir())

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
                "mask_sparsity/target",
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
            self.trainer.warmup_one_iter()

        for prune_iter in range(start_iter, self.config["N_PRUNING_ITER"]):
            start_epoch, sparsity_info = self.reset(prune_iter, epoch_to_resume > 0)

            # if there is a valid file to resume
            if start_epoch < epoch_to_resume:
                start_epoch = epoch_to_resume
                epoch_to_resume = 0

            for epoch in range(start_epoch, self.trainer.total_epochs):
                self.trainer.run_one_epoch(epoch, sparsity_info)

                # store weights with warmup
                if self.config["PRUNE_PARAMS"]["STORE_PARAM_BEFORE"] - 1 == epoch:
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
            target_density = target_density * (
                1 - self.config["PRUNE_PARAMS"]["PRUNE_AMOUNT"]
            )
        return 1 - target_density

    def save_init_params(self) -> None:
        """Set initial weights."""
        self.trainer.save_params(
            self.dir_prefix,
            self.init_params_name,
            self.config["PRUNE_PARAMS"]["STORE_PARAM_BEFORE"] - 1,
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

    def early_stop(self) -> None:
        """Early stop."""
        logger.info("Prune cannot be done. Early stop")
        raise Exception("Early Stop")


class LotteryTicketHypothesis(Pruner):
    """LTH on whole layer.

    References:
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
            amount=self.config["PRUNE_PARAMS"]["PRUNE_AMOUNT"],
        )


class LotteryTicketHypothesisFC(LotteryTicketHypothesis):
    """LTH on fc layer only.

    References:
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


class ChannelwisePruning(Pruner):
    """Channel-wise pruning."""

    def __init__(
        self,
        config: Dict[str, Any],
        dir_prefix: str,
        wandb_log: bool,
        wandb_init_params: Dict[str, Any],
        device: torch.device,
    ) -> None:
        """Initialize."""
        super(ChannelwisePruning, self).__init__(
            config, dir_prefix, wandb_log, wandb_init_params, device
        )

    @property
    def channel_conv_bn(
        self,
    ) -> Tuple[Tuple[nn.Module, nn.Conv2d, nn.BatchNorm2d], ...]:
        """Create channel_conv when it is needed."""
        if not hasattr(self, "_channel_conv_bn"):
            self._channel_conv_bn = self.get_channel_conv_bn()
        return self._channel_conv_bn

    @abc.abstractmethod
    def get_channel_conv_bn(
        self,
    ) -> Tuple[Tuple[nn.Module, nn.Conv2d, nn.BatchNorm2d], ...]:
        """Get channel - conv pair tuple."""
        raise NotImplementedError

    def prune_params(self, prune_iter: int) -> None:
        """Apply prune."""
        self.update_channel_data()
        params_to_prune = self.drop_overly_pruned(prune_iter)
        self.prune_target_ratio(prune_iter, params_to_prune)

    def prune_target_ratio(
        self, prune_iter: int, params_to_prune: Tuple[Tuple[nn.Module, str], ...]
    ) -> None:
        """Prune iteratively to reach up to target sparsity."""
        target_sparsity = self.get_target_sparsity(prune_iter)
        sparsity = model_utils.mask_sparsity(self.model_params) / 100
        prev_sparsity = sparsity
        base_amount = amount = 0.005

        # create adjacent module getter
        input_size = (1, *self.trainer.input_size)
        adjmodule_getter = AdjModuleGetter(
            self.model, input_size=input_size, device=self.device
        )

        # Give little margin to target_sparsity
        # target_sparsity never goes less than amount when used as intended
        while sparsity <= target_sparsity - amount:
            # drop overly pruned to prevent layer to die
            prune.global_unstructured(
                params_to_prune, pruning_method=prune.L1Unstructured, amount=amount
            )
            self.update_masks(adjmodule_getter)
            prev_sparsity = sparsity
            sparsity = model_utils.mask_sparsity(self.model_params) / 100
            if sparsity == prev_sparsity:
                amount += base_amount
        self.update_masks(adjmodule_getter)

    def update_masks(self, adjmodule_getter: AdjModuleGetter) -> None:
        """Copy channel to bias masks and conv masks."""
        # Note: model must have nn.Flatten to get last conv shape info
        last_conv_shape = adjmodule_getter.last_conv_shape

        for channel, conv, bn in self.channel_conv_bn:
            # Copy channel weight_mask to bias_mask
            ch_buffers = {name: buf for name, buf in channel.named_buffers()}
            ch_mask = ch_buffers["weight_mask"].detach().clone()
            if "bias_mask" in ch_buffers:
                ch_buffers["bias_mask"].set_(ch_mask)

            # Copy channel weight_mask to bn weight_mask, bias_mask
            bn_buffers = {name: buf for name, buf in bn.named_buffers()}
            bn_buffers["weight_mask"].set_(ch_mask)
            bn_buffers["bias_mask"].set_(ch_mask)

            conv_buffers = {name: buf for name, buf in conv.named_buffers()}
            if "bias_mask" in conv_buffers:
                conv_buffers["bias_mask"].set_(ch_mask)

            # conv2d - batchnorm - activation (CBA)
            # bn_mask: [out], conv: [out, in, h, w]
            [o, i, h, w] = conv_buffers["weight_mask"].shape

            # check shape -> if its not shaped as CBA
            if ch_mask.shape[0] != o:
                continue
            # ch_mask: [out, 1, 1]
            ch_mask = ch_mask.view(o, 1, 1)
            # ch_mask: [out, h, w]
            ch_mask = ch_mask.repeat(1, h, w)
            # ch_mask: [out, 1, h, w]
            ch_mask = ch_mask.unsqueeze(1)
            # ch_mask: [out, in, h, w]
            ch_mask = ch_mask.repeat(1, i, 1, 1)

            conv_buffers["weight_mask"].set_(ch_mask)

        # Update fc layer mask
        fc_modules: Dict[str, nn.Linear] = dict()
        bn_modules: Dict[str, nn.BatchNorm2d] = dict()
        for k, v in self.model.named_modules():
            if type(v) is nn.Linear:
                fc_modules.update({k: v})
            elif type(v) is nn.BatchNorm2d:
                bn_modules.update({k: v})

        for fc in fc_modules.values():
            bns = adjmodule_getter.find_modules_ahead_of(fc, nn.BatchNorm2d)
            bn_connections = [bn.weight_mask for bn in bns]

            if not bn_connections:
                continue

            fc_mask = torch.cat(bn_connections)
            fc_mask = torch.flatten(
                fc_mask.view(-1, 1, 1).repeat(1, last_conv_shape, last_conv_shape)
            )

            o, i = fc.weight_mask.size()  # type: ignore
            fc_mask = fc_mask.repeat(o).reshape(o, i)
            fc.weight_mask.data = fc_mask

    def drop_overly_pruned(self, prune_iter: int) -> Tuple[Tuple[nn.Module, str], ...]:
        """Exclude excessively pruned params to preserve flow."""
        # exclude param(layer)s to prevent 100% sparsity
        exclude_param_index: Set[int] = set()
        while len(exclude_param_index) != len(self.params_to_prune):
            pruner_cpy = copy.deepcopy(self)
            params_to_prune = pruner_cpy.update_params_to_prune(exclude_param_index)

            # try pruning
            pruner_cpy.prune_target_ratio(prune_iter, params_to_prune)
            if pruner_cpy.new_allzero_params(exclude_param_index):
                continue
            else:
                break

        # nothing to prune -> early stop
        if len(exclude_param_index) == len(self.params_to_prune):
            self.early_stop()

        # safely prunes
        return self.update_params_to_prune(exclude_param_index)

    def new_allzero_params(self, exclude_params: Set[int]) -> bool:
        """Check if there is params all zeroed and put into exclude params,
        return whether there is zeored params."""
        exclude_len = len(exclude_params)
        for i, (param, _) in enumerate(self.params_to_prune):
            param.weight_mask = cast(torch.Tensor, param.weight_mask)
            count_zero = int((param.weight_mask == 0).sum().detach())
            num_params = param.weight_mask.nelement()
            if count_zero == num_params:
                exclude_params.add(i)
        return True if exclude_len != len(exclude_params) else False

    def update_params_to_prune(
        self, exclude_param_index: Set[int]
    ) -> Tuple[Tuple[nn.Module, str], ...]:
        """Remove params_to_prune tuple by checking exclude_param_index."""
        excluded_params_prune = []
        for tuple_index, (layer, type_) in enumerate(self.params_to_prune):
            if tuple_index not in exclude_param_index:
                excluded_params_prune.append((layer, type_))
        return tuple(excluded_params_prune)


class NetworkSlimming(ChannelwisePruning):
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
        super(NetworkSlimming, self).__init__(
            config, dir_prefix, wandb_log, wandb_init_params, device
        )

    def get_channel_conv_bn(
        self,
    ) -> Tuple[Tuple[nn.Module, nn.Conv2d, nn.BatchNorm2d], ...]:
        """Get channel - conv - bn tuple.

        Note:
            Channel contains a data for prune.
            For example, frobenius norm in case of L2 magnitude,
                         bn weight for network slimming.
            In case of network slimming, channel is simply batchnorm.
        """
        layers = [
            v for v in self.model.modules() if type(v) in (nn.Conv2d, nn.BatchNorm2d)
        ]
        channel_conv_bn = []
        for i in range(1, len(layers)):
            if type(layers[i - 1]) is nn.Conv2d and type(layers[i]) is nn.BatchNorm2d:
                channel_conv_bn.append((layers[i], layers[i - 1], layers[i]))
        return tuple(channel_conv_bn)

    def get_params_to_prune(self) -> Tuple[Tuple[nn.Module, str], ...]:
        """Get parameters to prune."""
        # Cases there is no corresponding conv layer with bn
        all_bn_weights = model_utils.get_params(
            self.model, ((nn.BatchNorm2d, "weight"),)
        )
        return tuple(
            (module, name)
            for (module, name) in all_bn_weights
            if module in itertools.chain(*self.channel_conv_bn)
        )

    @torch.no_grad()
    def update_channel_data(self) -> None:
        """Update channel info into channel data for prune."""
        for channel, _, bn in self.channel_conv_bn:
            # get norm
            w = copy.deepcopy(bn.weight)
            channel.weight_orig.data = w.abs()
            # get sample input for dummpy forward
            dummy_data = torch.zeros_like(channel.weight_orig.data).view(1, -1, 1, 1)
            channel.eval()
            channel(dummy_data)


class ChannelInfo(nn.Module):
    """Module contains channel info for pruning."""

    def __init__(self, out_channels: int) -> None:
        super(ChannelInfo, self).__init__()
        self.weight = nn.Parameter(torch.zeros(out_channels, requires_grad=False))
        self.bias = nn.Parameter(torch.zeros(out_channels, requires_grad=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class Magnitude(ChannelwisePruning):
    """Magnitude based channel-wise pruning.

    Set NORM in PRUNE_PARAMS, for type of norm.
    Possibly all types of norm that torch.norm supports.
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
        super(Magnitude, self).__init__(
            config, dir_prefix, wandb_log, wandb_init_params, device
        )

    def get_channel_conv_bn(
        self,
    ) -> Tuple[Tuple[nn.Module, nn.Conv2d, nn.BatchNorm2d], ...]:
        """Get channel - conv - bn tuple.

        Note:
            Channel contains a data for prune.
            For example, frobeinus norm in case of L2 magnitude,
            bn weight for network slimming.
        """
        layers = [
            v for v in self.model.modules() if type(v) in (nn.Conv2d, nn.BatchNorm2d)
        ]
        channel_conv_bn = []
        for i in range(1, len(layers)):
            if type(layers[i - 1]) is nn.Conv2d and type(layers[i]) is nn.BatchNorm2d:
                out_channel = getattr(layers[i - 1], "weight").size()[0]
                ch_info = ChannelInfo(int(out_channel)).to(self.device)
                channel_conv_bn.append((ch_info, layers[i - 1], layers[i]))
        return tuple(channel_conv_bn)

    def get_params_to_prune(self) -> Tuple[Tuple[nn.Module, str], ...]:
        """Get parameters to prune."""
        t = [(channel, "weight") for channel, _, _ in self.channel_conv_bn]
        return tuple(t)

    @torch.no_grad()
    def update_channel_data(self) -> None:
        """Update channel info into channel data for prune."""
        for channel, conv, _ in self.channel_conv_bn:
            # get norm
            w = copy.deepcopy(conv.weight)
            output_, input_, h_, w_ = w.size()
            w = w.view(output_, -1)
            normed_w = torch.norm(w, p=self.config["PRUNE_PARAMS"]["NORM"], dim=(1))

            channel.weight_orig.data = normed_w

            # dummy forward for hook
            dummy_data = torch.zeros_like(normed_w).view(1, -1, 1, 1)
            channel.eval()
            channel(dummy_data)


class SlimMagnitude(Magnitude):
    """Slim + Magnitude based layerwise pruning.

    Bn_weight * L2_mag
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
        super(Magnitude, self).__init__(
            config, dir_prefix, wandb_log, wandb_init_params, device
        )

    @torch.no_grad()
    def update_channel_data(self) -> None:
        """Update channel info into channel data for prune."""
        for channel, conv, bn in self.channel_conv_bn:
            # get norm
            w = copy.deepcopy(conv.weight)
            output_, input_, h_, w_ = w.size()
            w = w.view(output_, -1)
            normed_w = torch.norm(w, p=self.config["PRUNE_PARAMS"]["NORM"], dim=(1))
            bn_w = copy.deepcopy(bn.weight)

            channel.weight_orig.data = normed_w * bn_w.abs()

            # dummy forward for hook
            dummy_data = torch.zeros_like(normed_w).view(1, -1, 1, 1)
            channel.eval()
            channel(dummy_data)
