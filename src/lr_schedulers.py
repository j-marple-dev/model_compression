# -*- coding: utf-8 -*-
"""Learning rate schedulers.

- Author: Curt-Park
- Email: jwpark@jmarple.ai
"""


import math

from torch.optim.optimizer import Optimizer


class WarmupCosineLR:
    """Cosine learning rate scheduler with warm-up."""

    def __init__(
        self, warmup_epochs: int, epochs: int, base_lr: float, target_lr: float,
    ) -> None:
        """Initialize."""
        self.warmup_epochs = warmup_epochs
        self.epochs = epochs
        self.base_lr = base_lr
        self.target_lr = target_lr
        self.coies = [
            math.cos((i - warmup_epochs) * math.pi / (epochs - warmup_epochs))
            for i in range(epochs)
        ]

    def lr(self, epoch: int) -> float:
        """Get learning rate."""
        if epoch < self.warmup_epochs:
            return (
                self.base_lr
                + (self.target_lr - self.base_lr) / self.warmup_epochs * epoch
            )
        else:
            return 0.5 * (1 + self.coies[epoch]) * self.target_lr

    def __call__(self, optimizer: Optimizer, epoch: int) -> None:
        """Set optimizer's learning rate."""
        lr = self.lr(epoch)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
