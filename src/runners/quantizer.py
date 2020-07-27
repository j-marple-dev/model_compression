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
from src.runners.runner import Runner
from src.runners.trainer import Trainer
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
    _, acc = trainer.test_one_epoch_model(model)
    inf_time = (time.time() - s) / len(trainer.testloader.dataset)
    size = model_utils.get_model_size_mb(model)
    logger.info(
        f"Acc: {acc['model_acc']:.2f} %\t"
        f"Size: {size:.6f} MB\t"
        f"Avg. Inference Time: {inf_time * 1000:.6f} ms"
    )


class Quantizer(Runner):
    """Quantizer for trained models."""

    def __init__(
        self,
        config: Dict[str, Any],
        checkpoint_path: str,
        dir_prefix: str,
        static: bool,
        backend: str,
        wandb_log: bool,
        wandb_init_params: Dict[str, Any],
    ) -> None:
        """Initialize."""
        super(Quantizer, self).__init__(config, dir_prefix)
        self.mask: Dict[str, torch.Tensor] = dict()
        self.params_pruned = None
        self.static = static
        self.backend = backend

        # create a trainer
        self.trainer = Trainer(
            config=self.config,
            dir_prefix=dir_prefix,
            checkpt_dir="qat",
            wandb_log=wandb_log,
            wandb_init_params=wandb_init_params,
            device="cpu",
            test_preprocess_hook=self._quantize,
        )

        # initialize the model
        logger.info("Initialize the model")
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
        self._init_model(checkpoint_path)

    def run(self, resume_info_path: str = "") -> None:
        """Run quantization."""
        logger.info("Estimate the original model's size and accuracy")
        # print_datatypes(self.model, "original model")
        # self.trainer.warmup_one_iter()
        estimate_acc_size(self.model, self.trainer)

        # fuse the model
        self._prepare()
        # print_datatypes(self.model, "Fused model")

        # post training static quantization
        if self.static:
            logger.info("Post Training Static Quantization: Run calibration")
            self.trainer.warmup_one_iter()
        # quantization-aware training
        else:
            logger.info("Quantization Aware Training: Run training")
            self.trainer.run(resume_info_path)
            self.model.apply(torch.quantization.disable_observer)
            self.model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)
            # load the best model
            self._load_best_model()

        # quantize the model
        quantized_model = self._quantize(self.model)
        # print_datatypes(quantized_model, "Quantized model")
        logger.info("Estimate the quantized model's size and accuracy")
        estimate_acc_size(quantized_model, self.trainer)
        torch.save(
            quantized_model.state_dict(),
            os.path.join(self.dir_prefix, "quantized_model.pth"),
        )

        # script the model
        scripted_model = torch.jit.script(quantized_model)
        # print_datatypes(scripted_model, "Scripted model")
        logger.info("Estimate the scripted model's size and accuracy")
        estimate_acc_size(scripted_model, self.trainer)

        # save and load the scripted model
        scripted_model_path = os.path.join(self.dir_prefix, "scripted_model.pth")
        torch.jit.save(scripted_model, scripted_model_path)
        logger.info(f"Saved the scripted model as {scripted_model_path}")
        logger.info(f"Loaded the scripted model as {scripted_model_path}")
        loaded_model = torch.jit.load(scripted_model_path)
        logger.info("Estimate the loaded scripted model's size and accuracy")
        estimate_acc_size(loaded_model, self.trainer)

    def _init_model(self, checkpoint_path: str) -> None:
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
            model_utils.dummy_pruning(self.params_all)

        # initialize weights
        logger.info("Initialize weights")
        model_utils.initialize_params(self.model, state_dict)

        if is_pruned:
            logger.info(
                "Get masks and remove prunning reparameterization for prepare_qat"
            )
            self.mask = model_utils.get_masks(self.model)
            model_utils.remove_pruning_reparameterization(self.params_all)

    def _prepare(self) -> None:
        """Quantize the model."""
        self.model.fuse_model()

        # configuration
        self.model.qconfig = torch.quantization.get_default_qat_qconfig(self.backend)

        # prepare
        if self.static:
            torch.quantization.prepare(self.model, inplace=True)
        else:
            torch.quantization.prepare_qat(self.model, inplace=True)

        # load masks
        self._load_masks()

    def _load_masks(self) -> None:
        """Load masks."""
        if not self.mask:
            return

        model_utils.dummy_pruning(self.params_all)
        for name, _ in self.model.named_buffers():
            if name in self.mask:
                module_name, mask_name = name.rsplit(".", 1)
                module = eval("self.model." + module_name)
                module._buffers[mask_name] = self.mask[name]

    def _load_best_model(self) -> None:
        """Load the trained model with the best accuracy."""
        self.trainer.resume()

    def _quantize(self, model: nn.Module) -> nn.Module:
        """Quantize the trained model."""
        if self.mask:
            model_utils.remove_pruning_reparameterization(self.params_all)

        # check the accuracy after each epoch
        quantized_model = torch.quantization.convert(model.eval(), inplace=False)
        quantized_model.eval()

        # set masks again
        if self.mask:
            self._load_masks()

        return quantized_model
