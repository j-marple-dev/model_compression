# -*- coding: utf-8 -*-
"""Collection of losses.

- Author: Junghoon Kim
- Email: jhkim@jmarple.ai
"""

import os
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import src.utils as utils

logger = utils.get_logger()


class Loss(nn.Module):
    """Base class for loss."""

    def __init__(
        self, model: nn.Module, config: Dict[str, Any], device: torch.device
    ) -> None:
        """Initialize.

        Args:
            model (nn.Module): model to be trained.
            config (Dict[str, Any]): config containing model.
            device (torch.device): device type(GPU, CPU).
        """
        super().__init__()
        self.model = model
        self.config = config
        self.device = device


class HintonKLD(Loss):
    """Hinton KLD Loss accepting soft labels.

    Reference:
        Distilling the Knowledge in a Neural Network(https://arxiv.org/pdf/1503.02531.pdf)

    Attributes:
        T (float): Hinton loss param, temperature(>0).
        alpha (float): Hinton loss param, alpha(0~1).
        cross_entropy (CrossEntropy): cross entropy loss.
        teacher (nn.Module): teacher model.
    """

    def __init__(
        self, model: nn.Module, config: Dict[str, Any], device: torch.device
    ) -> None:
        """Initialize cross entropy loss."""
        super().__init__(model, config, device)
        self.T = (
            self.config["CRITERION_PARAMS"]["T"]
            if "T" in self.config["CRITERION_PARAMS"].keys()
            else 4.0
        )
        assert 0.0 < self.T, (
            f"Hinton loss param, temperature({self.T}) "
            "is invalid value(not a positive)"
        )
        self.alpha = (
            self.config["CRITERION_PARAMS"]["alpha"]
            if "alpha" in self.config["CRITERION_PARAMS"].keys()
            else 0.9
        )
        assert 0.0 <= self.alpha <= 1.0, (
            f"Hinton loss param, alpha({self.alpha}) "
            "is invalid value(not in range 0~1)"
        )

        self.cross_entropy = CrossEntropy(model, config, device)
        self.teacher = self._create_teacher()

    def _create_teacher(self) -> nn.Module:
        """Create teacher network."""
        # exsitense of teacher config
        assert (
            "TEACHER_MODEL_NAME" in self.config
            and "TEACHER_MODEL_PARAMS" in self.config
        ) and (
            self.config["TEACHER_MODEL_NAME"] and self.config["TEACHER_MODEL_PARAMS"]
        ), "No teacher model specified while criterion requires teacher"

        # create teacher instance
        teacher = utils.get_model(
            self.config["TEACHER_MODEL_NAME"], self.config["TEACHER_MODEL_PARAMS"]
        ).to(self.device)

        # teacher path info
        prefix = os.path.join("save", "pretrained")
        model_info = utils.get_pretrained_model_info(teacher)
        model_name, file_name = model_info["dir_name"], model_info["file_name"]
        file_path = os.path.join(prefix, model_name, file_name)

        # load teacher model params:
        if not os.path.isfile(file_path):
            utils.download_pretrained_model(file_path, model_info["link"])
            logger.info(
                f"Pretrained teacher model({model_name}) doesn't exist in the path.\t"
                f"Download teacher model as {file_path}"
            )

        logger.info(f"Load teacher model: {file_path}")
        utils.initialize_params(
            model=teacher, state_dict=torch.load(file_path)["state_dict"],
        )
        teacher = teacher.to(self.device)
        teacher.eval()
        return teacher

    def forward(
        self, images: torch.Tensor, labels: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward model, calculate loss.

        Args:
            image (torch.Tensor): input images.
            labels (torch.Tensor): labels for input images.

        Returns:
            loss (torch.Tensor): calculated loss.
            logit (Dict[str, torch.Tensor]): model output.
        """
        with torch.no_grad():
            logit_t = self.teacher(images)
        logit_s = self.model(images)

        return (
            self.calculate_loss(logit_s=logit_s, logit_t=logit_t, labels=labels),
            {"model": logit_s, "teacher": logit_t},
        )

    def calculate_loss(
        self, logit_s: torch.Tensor, logit_t: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """Pure part of calculate loss, does not contain model forward procedure, \
        so that it can be combined with other loss.

        Args:
            logit_s (torch.Tensor): student model output,
                (https://developers.google.com/machine-learning/glossary/#logits).
            logit_t (torch.Tensor): teacher model ouptut.
            labels (torch.Tensor): labels for input images.

        Returns:
            loss (torch.Tensor): calculated loss.
        """
        log_p_s = F.log_softmax(logit_s / self.T, dim=1)
        p_t = F.softmax(logit_t / self.T, dim=1)
        hinton_kld = F.kl_div(log_p_s, p_t, reduction="batchmean") * (self.T ** 2)
        ce = self.cross_entropy.calculate_loss(logit_s, labels)
        return (1.0 - self.alpha) * ce + self.alpha * hinton_kld


class CrossEntropy(Loss):
    """Crossentropy loss accepting soft labels.

    Attributes:
        log_softmax (nn.Module): log softmax function.
        size_average (bool): param in cross entropy loss,
            if true, losses are averaged over the minibatch size,
            else, losses are summed only.
    """

    def __init__(
        self, model: nn.Module, config: Dict[str, Any], device: torch.device
    ) -> None:
        """Initialize cross entropy loss."""
        super().__init__(model, config, device)
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.size_average = (
            config["CRITERION_PARAMS"]["size_average"]
            if "size_average" in config["CRITERION_PARAMS"]
            else True
        )

    def forward(
        self, images: torch.Tensor, labels: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward model, calculate loss.

        Args:
            image (torch.Tensor): input images.
            labels (torch.Tensor): labels for input images.
        Returns:
            loss (torch.Tensor): calculated loss.
            logit (Dict[str, torch.Tensor]): model output.
        """
        logit = self.model(images)
        return self.calculate_loss(logit=logit, labels=labels), {"model": logit}

    def calculate_loss(self, logit: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Pure part of calculate loss, does not contain model forward procedure, \
        so that it can be combined with other loss.

        Args:
            logit (torch.Tensor): model output,
                (https://developers.google.com/machine-learning/glossary/#logits).
            labels (torch.Tensor): labels for input images.
        Returns:
            loss (torch.Tensor): calculated loss.
        """
        # if labels are index values -> expand to onehot for compatability
        onehot_labels = self._to_onehot(labels)
        log_y = self.log_softmax(logit)
        loss_total = torch.sum(-onehot_labels * log_y, dim=1)
        return torch.mean(loss_total) if self.size_average else torch.sum(loss_total)

    def _to_onehot(self, labels: torch.Tensor) -> torch.Tensor:
        """Convert index based labels into one-hot based labels.

           If labels are one-hot based already(e.g. [0.9, 0.01, 0.03,...]),do nothing.
        """
        if len(labels.size()) == 1:
            labels = nn.functional.one_hot(
                labels, num_classes=self.config["MODEL_PARAMS"]["num_classes"]
            )
        return labels.float().to(self.device)


def get_loss(
    model: nn.Module, config: Dict[str, Any], device: torch.device
) -> nn.Module:
    """Create loss class."""
    return eval(config["CRITERION"])(model, config, device)
