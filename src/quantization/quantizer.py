# -*- coding: utf-8 -*-
"""Quantizer for trained models.

- Author: Curt-Park
- Email: jwpark@jmarple.ai
"""


import os
import time
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.quantization

from src.models import utils as model_utils
from src.pruning.trainer import Trainer
import src.utils as utils

logger = utils.get_logger()


def print_datatypes(model: nn.Module, model_name: str, sep: str = "\n") -> None:
    """Print all datatypes in the model."""
    log = model_name + "'s datatypes:" + sep
    log += sep.join(str(t) for t in model_utils.get_model_tensor_datatype(model))
    logger.info(log)


def estimate_acc_size(model: nn.Module, trainer: Trainer) -> None:
    """Estimate the model's performance."""
    s = time.time()
    _, acc = trainer.test_one_epoch(model)
    inf_time = (time.time() - s) / len(trainer.testloader.dataset)
    size = model_utils.get_model_size_mb(model)
    logger.info(
        f"Acc: {acc['model_acc']:.2f} %\t"
        f"Size: {size:.6f} MB\t"
        f"Avg. Inference Time: {inf_time * 1000:.6f} ms"
    )


class Quantizer:
    """Quantizer for trained models."""

    def __init__(
        self,
        config: Dict[str, Any],
        checkpoint_path: str,
        dir_prefix: str,
        wandb_log: bool,
        wandb_init_params: Dict[str, Any],
    ) -> None:
        """Initialize."""
        self.config = config
        self.dir_prefix = dir_prefix
        self.best_acc = 0.0
        self.params_to_prune = None

        # create a trainer
        def test_preprocess_hook(_: nn.Module) -> nn.Module:
            return self._quantize()

        self.trainer = Trainer(
            config=self.config["TRAIN_CONFIG"],
            dir_prefix=dir_prefix,
            checkpt_dir="qat",
            wandb_log=wandb_log,
            wandb_init_params=wandb_init_params,
            device="cpu",
            test_preprocess_hook=test_preprocess_hook,
        )

        # initialize the model
        logger.info("Initialize the model")
        self.model = self.trainer.model
        self.mask = self._init_model(checkpoint_path)

    def run(self) -> None:
        """Run quantization."""
        logger.info("Estimate the original model")
        print_datatypes(self.model, "original model")
        estimate_acc_size(self.model, self.trainer)

        # fuse the model
        self._prepare()
        print_datatypes(self.model, "Fused model")

        # quantization-aware training
        self.trainer.run()
        self.model.apply(torch.quantization.disable_observer)
        self.model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)

        # load the best model
        self._load_best_model()

        # quantize the model
        quantized_model = self._quantize()
        print_datatypes(quantized_model, "Quantized model")
        logger.info("Estimate the quantized model")
        estimate_acc_size(quantized_model, self.trainer)
        torch.save(
            quantized_model.state_dict(),
            os.path.join(self.dir_prefix, "quantized_model.pt"),
        )

        # script the model
        scripted_model = torch.jit.script(quantized_model)
        print_datatypes(scripted_model, "Scripted model")
        logger.info("Estimate the scripted model")
        estimate_acc_size(scripted_model, self.trainer)
        torch.jit.save(
            scripted_model, os.path.join(self.dir_prefix, "scripted_model.pth.zip")
        )

    def _init_model(self, checkpoint_path: str) -> Dict[str, torch.Tensor]:
        """Create a model instance and load weights."""
        # load weights
        logger.info(f"Load weights from the checkpoint {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location=torch.device("cpu"))[
            "state_dict"
        ]

        # check the trained model is pruned
        is_pruned = (
            next((name for name in state_dict if "mask" in name), None) is not None
        )

        if is_pruned:
            logger.info("Dummy prunning to load pruned weights")
            params_to_prune = model_utils.dummy_pruning(self.model)

        # initialize weights
        logger.info("Initialize weights")
        model_utils.initialize_params(self.model, state_dict)

        # store masks and remove pruning reparameterization
        mask = model_utils.get_masks(self.model)

        if is_pruned:
            logger.info("Remove prunning reparameterization for prepare_qat")
            model_utils.remove_pruning_reparameterization(params_to_prune)

        return mask

    def _prepare(self) -> None:
        """Quantize the model."""
        self.model.fuse_model()

        # configuration
        self.model.qconfig = torch.quantization.get_default_qat_qconfig(
            self.config["QUANT_BACKEND"]
        )
        torch.quantization.prepare_qat(self.model, inplace=True)

        # load masks
        self._load_masks()

    def _load_masks(self) -> None:
        """Load masks."""
        if not self.mask:
            return

        self.params_to_prune = model_utils.dummy_pruning(self.model)
        for name, _ in self.model.named_buffers():
            if name in self.mask:
                module_name, mask_name = name.rsplit(".", 1)
                module = eval("self.model." + module_name)
                module._buffers[mask_name] = self.mask[name]

    def _load_best_model(self) -> None:
        """Load the trained model with the best accuracy."""
        self.trainer.resume()

    def _quantize(self) -> nn.Module:
        """Quantize the trained model."""
        if self.mask:
            model_utils.remove_pruning_reparameterization(self.params_to_prune)

        # check the accuracy after each epoch
        quantized_model = torch.quantization.convert(self.model.eval(), inplace=False)
        quantized_model.eval()

        # set masks again
        if self.mask:
            self._load_masks()

        return quantized_model
