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

from src.models import utils as model_utils
import src.utils as utils

logger = utils.get_logger()


class Criterion(nn.Module):
    """Base class for criterion."""

    def __init__(self, device: torch.device) -> None:
        """Initialize.

        Args:
            device (torch.device): device type(GPU, CPU)
        """
        super().__init__()
        self.device = device


class HintonKLD(Criterion):
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
        self,
        device: torch.device,
        T: float,
        alpha: float,
        teacher_model_name: str,
        teacher_model_params: Dict[str, Any],
        crossentropy_params: Dict[str, Any],
    ) -> None:
        """Initialize cross entropy loss."""
        super().__init__(device)
        self.cross_entropy = CrossEntropy(device, **crossentropy_params)
        self.T = T
        self.alpha = alpha
        self.teacher = self._create_teacher(teacher_model_name, teacher_model_params)

    def _create_teacher(
        self, teacher_model_name: str, teacher_model_params: Dict[str, Any]
    ) -> nn.Module:
        """Create teacher network."""
        # create teacher instance
        teacher = model_utils.get_model(teacher_model_name, teacher_model_params).to(
            self.device
        )

        # teacher path info
        prefix = os.path.join("save", "pretrained")
        model_info = model_utils.get_pretrained_model_info(teacher)
        model_name, file_name = model_info["dir_name"], model_info["file_name"]
        file_path = os.path.join(prefix, model_name, file_name)

        # load teacher model params:
        if not os.path.isfile(file_path):
            model_utils.download_pretrained_model(file_path, model_info["link"])
            logger.info(
                f"Pretrained teacher model({model_name}) doesn't exist in the path.\t"
                f"Download teacher model as {file_path}"
            )

        logger.info(f"Load teacher model: {file_path}")
        state_dict = torch.load(file_path, map_location=self.device)["state_dict"]
        model_utils.initialize_params(model=teacher, state_dict=state_dict)
        teacher = teacher.to(self.device)
        teacher.eval()
        return teacher

    def forward(
        self, model: nn.Module, images: torch.Tensor, labels: torch.Tensor
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
        logit_s = model(images)

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


class CrossEntropy(Criterion):
    """Crossentropy loss accepting soft labels.

    Attributes:
        log_softmax (nn.Module): log softmax function.
        num_classes (int): number of classes in dataset, to get onehot labels
    """

    def __init__(
        self, device: torch.device, num_classes: int, label_smoothing: float
    ) -> None:
        """Initialize cross entropy loss."""
        super().__init__(device)
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.num_classes = num_classes
        self.label_smoothing = label_smoothing

    def forward(
        self, model: nn.Module, images: torch.Tensor, labels: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward model, calculate loss.

        Args:
            image (torch.Tensor): input images.
            labels (torch.Tensor): labels for input images.
        Returns:
            loss (torch.Tensor): calculated loss.
            logit (Dict[str, torch.Tensor]): model output.
        """
        logit = model(images)
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
        target = utils.to_onehot(labels=labels, num_classes=self.num_classes).to(
            self.device
        )
        pred = self.log_softmax(logit)

        # get smooth labels
        if self.label_smoothing > 0.0:
            target = self.add_label_smoothing(target)

        return torch.mean(torch.sum(-target * pred, dim=1))

    @torch.no_grad()
    def add_label_smoothing(self, target: torch.Tensor) -> torch.Tensor:
        """Add smoothness in labels."""
        nonzero_idxs = target != 0.0
        nonzero_cnt = nonzero_idxs.sum(dim=1, keepdim=True).float()

        target *= 1 - self.label_smoothing
        smooth_target = torch.ones_like(target).to(self.device)
        smooth_target *= self.label_smoothing / (self.num_classes - nonzero_cnt)
        smooth_target[nonzero_idxs] = target[nonzero_idxs]
        return smooth_target


def get_criterion(
    criterion_name: str,
    criterion_params: Dict[str, Any],
    device: torch.device,
) -> nn.Module:
    """Create loss class."""
    return eval(criterion_name)(device, **criterion_params)
