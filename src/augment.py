# -*- coding: utf-8 -*-
"""PyTorch transforms for data augmentation.

- Author: Curt-Park
- Email: jwpark@jmarple.ai
"""

import torchvision.transforms as transforms


def simple_augment_train_cifar100() -> transforms.Compose:
    """Simple data augmentation rule for training CIFAR100."""
    return transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )


def simple_augment_test_cifar100() -> transforms.Compose:
    """Simple data augmentation rule for testing CIFAR100."""
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
