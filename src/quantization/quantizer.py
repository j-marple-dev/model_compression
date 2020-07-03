# -*- coding: utf-8 -*-
"""Quantizer for trained models.

- Author: Curt-Park
- Email: jwpark@jmarple.ai
"""


import os
import time
from typing import Any, Dict, Tuple

from progressbar import progressbar
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch.quantization

from src.models import utils as model_utils
import src.utils as utils

DEBUG_RUN_MODEL_GPU = True  # the original model inference on GPU
logger = utils.get_logger()


class Quantizer:
    """Quantizer for trained models."""

    def __init__(
        self, config: Dict[str, Any], checkpoint_path: str, dir_prefix: str
    ) -> None:
        """Initialize."""
        self.config = config
        self.dir_prefix = dir_prefix

        # create a model
        logger.info(f"Create {self.config['MODEL_NAME']} and load weights")
        self.model = self._create_model(checkpoint_path)

        # get dataloaders
        logger.info("Prepare dataloaders")
        trainset, testset = utils.get_dataset(
            config["DATASET"], config["AUG_TRAIN"], config["AUG_TEST"],
        )
        self.trainloader, self.testloader = utils.get_dataloader(
            trainset, testset, config["BATCH_SIZE"], config["N_WORKERS"],
        )

        # get input shape to create a scripted model
        self.input_shape = [config["BATCH_SIZE"]] + list(trainset[0][0].size())

    def run(self) -> None:
        """Run quantization."""
        # get quantized models
        quantized_model = self._quantize(self.model)
        scripted_model = self._script(quantized_model)

        # print datatypes
        self._print_datatypes("quantized_model", quantized_model)

        # estimation
        logger.info("Estimate the original model")
        if DEBUG_RUN_MODEL_GPU:
            self._estimate_acc_size(self.model, "cuda")
        else:
            self._estimate_acc_size(self.model)

        logger.info("Estimate the quantized model")
        self._estimate_acc_size(quantized_model)

        logger.info("Estimate the scripted model")
        self._estimate_acc_size(scripted_model)

    def _print_datatypes(
        self, model_name: str, model: nn.Module, sep: str = "\n"
    ) -> None:
        """Print all datatypes in the model."""
        log = model_name + "'s datatypes:" + sep
        log += sep.join(str(t) for t in model_utils.get_model_tensor_datatype(model))
        logger.info(log)

    def _quantize(self, model: nn.Module) -> nn.Module:
        """Quantize the model."""
        model.cpu().eval()
        quantized_model = torch.quantization.quantize_dynamic(model, dtype=torch.qint8)
        return quantized_model

    def _script(self, model: nn.Module) -> nn.Module:
        """Get torch scripted model."""
        input_data = torch.randn(self.input_shape)
        scripted_model = torch.jit.trace(model, input_data)
        return scripted_model

    def _estimate_acc_size(self, model: nn.Module, device="cpu") -> None:
        """Estimate the model's performance."""
        acc, inf_time = self._test(model, device)
        size = self._get_model_size_mb(model)
        logger.info(
            f"Acc: {acc:.2f} %\t"
            f"Size: {size:.6f} MB\t"
            f"Avg. Inference Time: {inf_time:.6f} Sec"
        )

    def _create_model(self, checkpoint_path: str) -> nn.Module:
        """Create a model instance and load weights."""
        # create a model
        logger.info(f"Create a(n) {self.config['MODEL_NAME']} instance")
        model_name = self.config["MODEL_NAME"]
        model_config = self.config["MODEL_PARAMS"]
        model = model_utils.get_model(model_name, model_config).cpu()

        # load weights
        logger.info(f"Load weights from the checkpoint {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location=torch.device("cpu"))[
            "state_dict"
        ]

        # check the trained model is pruned
        is_pruned = (
            next((name for name in state_dict if "mask" in name), None) is not None
        )

        # dummy pruning
        if is_pruned:
            logger.info("Dummy prunning to load pruned weights")
            # the following pruning steps must be same as the one conducted for pruning
            params_to_prune = model_utils.get_weight_tuple(model, bias=False)
            prune.global_unstructured(
                params_to_prune, pruning_method=prune.L1Unstructured, amount=0.0,
            )

        # initialize weights
        logger.info("Initialize weights")
        model_utils.initialize_params(model, state_dict)

        # remove pruning reparameterization
        # this combines (weight_orig, weight_mask) and reduce the model size
        if is_pruned:
            logger.info("Remove pruning reparameterazation")
            for module, weight_type in params_to_prune:
                prune.remove(module, weight_type)

        return model

    @torch.no_grad()
    def _test(self, model: nn.Module, device: str) -> Tuple[float, float]:
        """Test one epoch."""
        model.to(device).eval()
        n_correct = n_iter = 0

        s = time.time()
        for data in progressbar(self.testloader, prefix="[Test]\t"):
            images, labels = data[0].to(device), data[1].to(device)
            output = model(images)
            pred = output.argmax(dim=1, keepdim=True)
            n_correct += pred.eq(labels.view_as(pred)).sum().item()
            n_iter += 1
        avg_time = (time.time() - s) / n_iter

        n_total = len(self.testloader.dataset)
        acc = 100 * n_correct / n_total
        return acc, avg_time

    def _get_model_size_mb(self, model: nn.Module) -> float:
        """Get the model file size."""
        torch.save(model.state_dict(), "temp.p")
        size = os.path.getsize("temp.p") / 1e6
        os.remove("temp.p")
        return size
