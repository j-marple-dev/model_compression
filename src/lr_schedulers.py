# -*- coding: utf-8 -*-
"""Learning rate schedulers.

- Author: Curt-Park
- Email: jwpark@jmarple.ai
"""


from abc import ABC, abstractmethod
import math
from typing import Any, Dict, List

from torch.optim.optimizer import Optimizer


class LrScheduler(ABC):
    """Abstract class for learning rate schedulers."""

    @abstractmethod
    def __call__(self, optimizer: Optimizer, epoch: int) -> None:
        """Set optimizer's learning rate."""
        raise NotImplementedError


class Identity(LrScheduler):
    """Keep learning rate as config["LR"]."""

    def __call__(self, optimizer: Optimizer, epoch: int) -> None:
        """Set optimizer's learning rate."""
        return None


class MultiStepLR(LrScheduler):
    """Multi Step LR scheduler."""

    def __init__(self, milestones: List[int], gamma: float) -> None:
        """Initialize."""
        self.milestones = set(milestones)
        self.gamma = gamma

    def __call__(self, optimizer: Optimizer, epoch: int) -> None:
        """Set optimizer's learning rate."""
        if epoch not in self.milestones:
            return None

        for param_group in optimizer.param_groups:
            param_group["lr"] *= self.gamma


class WarmupCosineLR(LrScheduler):
    """Cosine learning rate scheduler with warm-up."""

    # epochs and target_lr are automatically set in config validator
    def __init__(
        self,
        warmup_epochs: int,
        epochs: int,
        start_lr: float,
        target_lr: float,
        min_lr: float,
        n_rewinding: int,
        decay: float,
    ) -> None:
        """Initialize."""
        self.warmup_epochs = warmup_epochs
        self.base_lr = start_lr
        self.target_lr = target_lr
        self.min_lr = min_lr
        self.period = epochs // n_rewinding
        self.decay = decay
        self.coies = [
            math.cos((i - warmup_epochs) * math.pi / (self.period - warmup_epochs))
            for i in range(self.period)
        ]

    def lr(self, epoch: int) -> float:
        """Get learning rate."""
        n_iter, epoch = divmod(epoch, self.period)
        if epoch < self.warmup_epochs:
            lr = (
                self.base_lr
                + (self.target_lr - self.base_lr) / self.warmup_epochs * epoch
            )
        else:
            lr = 0.5 * (1 + self.coies[epoch]) * self.target_lr
        lr *= (1.0 - self.decay) ** n_iter
        return max(lr, self.min_lr)

    def __call__(self, optimizer: Optimizer, epoch: int) -> None:
        """Set optimizer's learning rate."""
        lr = self.lr(epoch)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr


def get_lr_scheduler(name: str, lr_scheduler_params: Dict[str, Any]) -> LrScheduler:
    """LR scheduler getter."""
    return eval(name)(**lr_scheduler_params)
