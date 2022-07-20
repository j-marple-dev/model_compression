# -*- coding: utf-8 -*-
"""Model shrinker.

- Author: Curt-Park
- Email: jwpark@jmarple.ai
"""

import os
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

from src.models import utils as model_utils
from src.models.adjmodule_getter import AdjModuleGetter
from src.runners.runner import Runner
from src.runners.trainer import Trainer
import src.utils as utils

logger = utils.get_logger()


class Shrinker(Runner):
    """Shrinker for pruned models.

    This removes masked channels and redefine the networks.

    Important Notes:
        Shrinker is now experimental. It only supports:
        - networks that consist of conv-bn-activation sequence
        - network blocks that has channel concatenation followed by skip connections
          (e.g. DenseNet)
        - networks that have only one last fully-connected layer

        On the other hads, it doesn't support:
        - network blocks that has element-wise sum followed by skip connections
          (e.g. ResNet, MixNet)
        - networks that have multiple fully-connected layers
    """

    def __init__(
        self,
        config: Dict[str, Any],
        checkpoint_path: str,
        dir_prefix: str,
        device: torch.device,
    ) -> None:
        """Initialize."""
        super(Shrinker, self).__init__(config, dir_prefix)
        self.train_config = self.config["TRAIN_CONFIG"]
        self.checkpoint_path = checkpoint_path
        self.device = device

        # create a trainer
        self.trainer = Trainer(
            config=self.config["TRAIN_CONFIG"],
            dir_prefix=dir_prefix,
            checkpt_dir="",
            device=self.device,
            wandb_log=False,
            wandb_init_params=None,
        )
        self.model = self.trainer.model

        # create adjacent module getter
        input_size = (1, *self.trainer.input_size)
        self.adjmodule_getter = AdjModuleGetter(
            self.model, input_size=input_size, device=self.device
        )

        # Note: model must have nn.Flatten to get last conv shape info
        self.last_conv_shape = self.adjmodule_getter.last_conv_shape

        # dummy pruning
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
        model_utils.dummy_pruning(self.params_all)

    def run(self, resume_info_path: str = "") -> None:
        """Run the module."""
        # initialize weights
        logger.info(f"Initialize the model from the checkpoint {self.checkpoint_path}")
        state_dict = torch.load(self.checkpoint_path, map_location=self.device)[
            "state_dict"
        ]
        model_utils.initialize_params(self.model, state_dict)

        # measure the model size
        # model has to run at least 1 time due to execution of forward hooks
        _, acc = self.trainer.test_one_epoch()
        size = model_utils.get_model_size_mb(self.model)
        sparsity = model_utils.sparsity(self.params_all)
        logger.info(
            f"Original model's Acc: {acc['model_acc']:.2f}, Size: {size} MB, "
            f"Sparsity: {sparsity:.2f} %"
        )

        shrinked_model = model_utils.get_model(
            self.train_config["MODEL_NAME"], self.train_config["MODEL_PARAMS"]
        ).to(self.device)
        shrinked_model = self.shrink_model(self.model, shrinked_model)

        # measure the shrinked model size
        _, acc = self.trainer.test_one_epoch_model(shrinked_model)
        size = model_utils.get_model_size_mb(shrinked_model)
        n_params = model_utils.count_model_params(shrinked_model)
        logger.info(
            f"Shrinked model's Acc: {acc['model_acc']:.2f}, Size: {size} MB, "
            f"Params: {(n_params * 1e-6):.2f} M"
        )

        # save the shrinked model
        shrinked_model_path = os.path.join(self.dir_prefix, "shrinked_model.pth")
        torch.save(shrinked_model, shrinked_model_path)
        logger.info(f"Saved shrinked model as {shrinked_model_path}")

        # load the shrinked model
        logger.info(f"Load a shrinked model from {shrinked_model_path}")
        loaded_model = torch.load(shrinked_model_path)
        loaded_model.eval()

        # measure the loaded model size
        _, acc = self.trainer.test_one_epoch_model(loaded_model)
        size = model_utils.get_model_size_mb(loaded_model)
        n_params = model_utils.count_model_params(loaded_model)
        logger.info(
            f"Loaded model's Acc: {acc['model_acc']:.2f}, Size: {size} MB, "
            f"Params: {(n_params * 1e-6):.2f} M"
        )

    @torch.no_grad()
    def shrink_model(self, old_model: nn.Module, new_model: nn.Module) -> nn.Module:
        """Shrink model by removing pruned layer.

        Return shrinked(new) model.
        """
        old_model.eval()
        new_model.eval()

        conv_named_modules: Dict[str, nn.Conv2d] = dict()
        bn_named_modules: Dict[str, nn.BatchNorm2d] = dict()
        fc_named_modules: Dict[str, nn.Linear] = dict()

        # get the information of Conv2d, BatchNorm2d, Linear modules
        for name, module in old_model.named_modules():
            if isinstance(module, nn.Conv2d):
                conv_named_modules.update({name: module})
            elif isinstance(module, nn.BatchNorm2d):
                bn_named_modules.update({name: module})
            elif isinstance(module, nn.Linear):
                fc_named_modules.update({name: module})

        # replace conv module
        self._reshape_convs(conv_named_modules, new_model)

        # replace bn module
        self._reshape_bns(bn_named_modules, new_model)

        # replace FC module
        self._reshape_fcs(fc_named_modules, new_model)

        return new_model

    def _reshape_convs(
        self, conv_named_modules: Dict[str, nn.Conv2d], new_model: nn.Module
    ) -> None:
        """Create resized conv and replace it in the existing one in the model."""
        for conv_name, conv in conv_named_modules.items():
            in_bns = self.adjmodule_getter.find_modules_ahead_of(conv, nn.BatchNorm2d)
            out_bn = self.adjmodule_getter.find_module_next_to(conv, nn.BatchNorm2d)

            in_bn_masks = [bn.weight_mask for bn in in_bns]
            # stem conv doesn't have input bn masks
            # this spports a single output or concatenated outputs before conv operator
            in_mask = torch.cat(in_bn_masks) if in_bn_masks else None

            # assume that every conv is followed by a single bn
            out_mask = out_bn.weight_mask

            # Create reshaped_conv
            reshaped_conv = self._generate_reshaped_conv(in_mask, out_mask, conv)

            # Set reshaped conv to new_model
            self._set_layer(new_model, conv_name, reshaped_conv)

    def _reshape_bns(
        self, bn_named_modules: Dict[str, nn.BatchNorm2d], new_model: nn.Module
    ) -> None:
        """Create resized bn and replace it in the existing one in the model."""
        # Replace bn module
        for bn_name, bn in bn_named_modules.items():
            # Get mask index
            mask_idx = self._generate_bn_mask_idx(bn)

            # Create reshaped_bn
            reshaped_bn = self._generate_reshaped_bn(bn, mask_idx)

            # Set reshaped_bn to new model
            self._set_layer(new_model, bn_name, reshaped_bn)

    def _reshape_fcs(
        self, fc_named_modules: Dict[str, nn.Linear], new_model: nn.Module
    ) -> None:
        """Create resized fc and replace it in the existing one in the model."""
        for fc_name, fc in fc_named_modules.items():
            in_bns = self.adjmodule_getter.find_modules_ahead_of(fc, nn.BatchNorm2d)

            # Get mask from bn
            in_bn_masks = [bn.weight_mask for bn in in_bns]
            # this spports a single output or concatenated outputs before conv operator
            in_mask = torch.cat(in_bn_masks) if in_bn_masks else None

            # in_mask is None if the model starts with nn.Linear
            if in_mask is None:
                continue

            # Create reshaped_fc
            reshaped_fc = self._generate_reshaped_fc(in_mask, fc)

            # Set reshaped_fc to new_model
            self._set_layer(new_model, fc_name, reshaped_fc)

    def _generate_reshaped_conv(
        self,
        in_mask: Optional[torch.Tensor],
        out_mask: torch.Tensor,
        conv: nn.Conv2d,
    ) -> nn.Conv2d:
        """Generate new conv given old conv and masks(in and out or out only)."""
        # Shrink both input, output channel of conv, and extract weight(orig, mask)
        [_, i, h, w] = getattr(conv, "weight").size()

        # Make mask for input
        if in_mask is not None:
            # make masking matrix[o, i]: in_mask.T * out_mask
            # mask_flattened : [o*i]
            mask_flattened = in_mask.unsqueeze(1).T * out_mask.unsqueeze(1)
            mask_flattened = mask_flattened.reshape(-1)
            mask_idx = (mask_flattened == 1).nonzero().view(-1, 1, 1).repeat(1, h, w)

            new_out = (out_mask == 1).nonzero().size()[0]
            new_in = (in_mask == 1).nonzero().size()[0]

            orig = conv.weight_orig.reshape(-1, h, w)  # type: ignore
            mask = conv.weight_mask.reshape(-1, h, w)  # type: ignore
            orig = torch.gather(orig, 0, mask_idx).reshape(new_out, new_in, h, w)
            mask = torch.gather(mask, 0, mask_idx).reshape(new_out, new_in, h, w)

        # Case only when there is out_mask
        else:
            # extract one masked index
            out_mask = (out_mask == 1).nonzero().view(-1, 1, 1).repeat(1, h, w)
            out_mask = out_mask.unsqueeze(1).repeat(1, i, 1, 1)

            orig = torch.gather(conv.weight_orig, 0, out_mask)  # type: ignore
            mask = torch.gather(conv.weight_mask, 0, out_mask)  # type: ignore

        # Create reshaped conv
        reshaped_conv = torch.nn.Conv2d(
            in_channels=mask.size()[1],
            out_channels=mask.size()[0],
            kernel_size=conv.kernel_size,  # type: ignore
            bias=conv.bias is not None,
            padding=conv.padding,  # type: ignore
            dilation=conv.dilation,  # type: ignore
            groups=conv.groups,
            stride=conv.stride,  # type: ignore
        ).to(self.device)

        # dummy prune to copy orig, mask to new conv
        # Note: pruned conv bias is not supported
        prune_conv = ((reshaped_conv, "weight"),)
        model_utils.dummy_pruning(prune_conv)

        # Overwrite data to new(reshaped) conv
        reshaped_conv.weight_orig.data = orig
        reshaped_conv.weight_mask.data = mask

        return reshaped_conv

    def _generate_bn_mask_idx(self, bn: nn.BatchNorm2d) -> torch.Tensor:
        """Generate batchnorm 2d mask index tensor."""
        bn_buffers = {name: buf for name, buf in bn.named_buffers()}
        mask_idx = (bn_buffers["weight_mask"] == 1).nonzero().squeeze(1)

        return mask_idx

    def _generate_reshaped_bn(
        self, bn: nn.BatchNorm2d, mask_idx: torch.Tensor
    ) -> nn.BatchNorm2d:
        """Generate new bn given old_bn, mask."""
        reshaped_bn = torch.nn.BatchNorm2d(
            num_features=mask_idx.size()[0],
            eps=bn.eps,
            momentum=bn.momentum,
            affine=bn.affine,
            track_running_stats=bn.track_running_stats,
        ).to(self.device)

        prune_bn: Tuple[Tuple[nn.BatchNorm2d, str], Tuple[nn.BatchNorm2d, str]] = (
            (reshaped_bn, "weight"),
            (reshaped_bn, "bias"),
        )
        model_utils.dummy_pruning(prune_bn)

        # set data to reshaped
        reshaped_bn.running_mean = torch.gather(bn.running_mean, 0, mask_idx)  # type: ignore
        reshaped_bn.running_var = torch.gather(bn.running_var, 0, mask_idx)  # type: ignore
        reshaped_bn.weight_mask.data = torch.gather(bn.weight_mask, 0, mask_idx)  # type: ignore
        reshaped_bn.weight_orig.data = torch.gather(bn.weight_orig, 0, mask_idx)  # type: ignore
        reshaped_bn.bias_mask.set_(torch.gather(bn.bias_mask, 0, mask_idx))  # type: ignore
        reshaped_bn.bias_orig.set_(torch.gather(bn.bias_orig, 0, mask_idx))  # type: ignore
        reshaped_bn.num_batches_tracked = bn.num_batches_tracked

        return reshaped_bn

    def _generate_reshaped_fc(self, mask: torch.Tensor, fc: nn.Linear) -> nn.Linear:
        """Generate new fc given old fc, mask, last_conv_shape."""
        # expand considering last_dim
        # ex) bn_dim * last_dim * last_dim feed into NN
        # repeat last_dim * last_dim
        in_mask = torch.flatten(
            mask.view(-1, 1, 1).repeat(1, self.last_conv_shape, self.last_conv_shape)
        )

        # Do shrink on fc
        in_features_size = int((in_mask == 1).sum())

        out_features, _ = fc.weight.size()
        weight_mask = in_mask.repeat(out_features)
        weight_mask_idx = (weight_mask == 1).nonzero().squeeze(1)
        weight = fc.weight.detach().clone()
        weight = torch.gather(torch.flatten(weight), 0, weight_mask_idx).reshape(
            out_features, in_features_size
        )

        reshaped_fc = torch.nn.Linear(
            in_features=in_features_size,
            out_features=out_features,
            bias=fc.bias is not None,
        ).to(self.device)
        param_to_prune = ((reshaped_fc, "weight"),)
        model_utils.dummy_pruning(param_to_prune)

        reshaped_fc.weight_orig.data = weight  # type: ignore
        reshaped_fc.weight_mask.data = torch.ones_like(weight)  # type: ignore
        # Note: this doesn't work if bias is pruned and dimension changed
        # Only available for the networks that have only one last fc
        if hasattr(fc, "bias"):
            reshaped_fc.bias.data = fc.bias
        prune.remove(reshaped_fc, "weight")

        return reshaped_fc

    def _set_layer(
        self, model: nn.Module, layer_name: str, new_module: nn.Module
    ) -> None:
        """Set layer to model."""
        split_layer_name = layer_name.rsplit(".", 1)
        if len(split_layer_name) == 1:
            module = model
            name = split_layer_name[0]
        else:
            parent_layer, name = split_layer_name
            module = eval("model" + "." + model_utils.dot2bracket(parent_layer))
        setattr(module, name, new_module)
