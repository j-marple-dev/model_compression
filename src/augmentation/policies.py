# -*- coding: utf-8 -*-
"""PyTorch transforms for data augmentation.

- Author: Curt-Park
- Email: jwpark@jmarple.ai
"""

import torchvision.transforms as transforms

from src.augmentation.methods import (
    AutoAugmentation,
    RandAugmentation,
    SequentialAugmentation,
)
from src.augmentation.transforms import FILLCOLOR

CIFAR100_INFO = {"MEAN": (0.5071, 0.4865, 0.4409), "STD": (0.2673, 0.2564, 0.2762)}
IMAGENET_INFO = {"MEAN": (0.485, 0.456, 0.406), "STD": (0.229, 0.224, 0.225)}
# CHALLENGE_INFO = {"MEAN": (0.0678, 0.2116, 0.0865), "STD": (0.2154, 0.1879, 0.2017)}
CHALLENGE_INFO = CIFAR100_INFO


def simple_augment_train_cifar100() -> transforms.Compose:
    """Simple data augmentation rule for training CIFAR100."""
    return transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_INFO["MEAN"], CIFAR100_INFO["STD"]),
        ]
    )


def simple_augment_test_cifar100() -> transforms.Compose:
    """Simple data augmentation rule for testing CIFAR100."""
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_INFO["MEAN"], CIFAR100_INFO["STD"]),
        ]
    )


def simple_augment_train_ai_challenge(image_size: int = 224) -> transforms.Compose:
    """Simple data augmentation rule for training AI_CHALLENGE dataset."""
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CHALLENGE_INFO["MEAN"], CHALLENGE_INFO["STD"]),
        ]
    )


def simple_augment_test_ai_challenge(image_size: int = 224) -> transforms.Compose:
    """Simple data augmentation rule for testing AI_CHALLENGE dataset."""
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(CHALLENGE_INFO["MEAN"], CHALLENGE_INFO["STD"]),
        ]
    )


def autoaugment_train_cifar100() -> transforms.Compose:
    """Auto augmentation policy for training CIFAR100."""
    policies = [
        [("Invert", 0.1, 7), ("Contrast", 0.2, 6)],
        [("Rotate", 0.7, 2), ("TranslateX", 0.3, 9)],
        [("Sharpness", 0.8, 1), ("Sharpness", 0.9, 3)],
        [("ShearY", 0.5, 8), ("TranslateY", 0.7, 9)],
        [("AutoContrast", 0.5, 8), ("Equalize", 0.9, 2)],
        [("ShearY", 0.2, 7), ("Posterize", 0.3, 7)],
        [("Color", 0.4, 3), ("Brightness", 0.6, 7)],
        [("Sharpness", 0.3, 9), ("Brightness", 0.7, 9)],
        [("Equalize", 0.6, 5), ("Equalize", 0.5, 1)],
        [("Contrast", 0.6, 7), ("Sharpness", 0.6, 5)],
        [("Color", 0.7, 7), ("TranslateX", 0.5, 8)],
        [("Equalize", 0.3, 7), ("AutoContrast", 0.4, 8)],
        [("TranslateY", 0.4, 3), ("Sharpness", 0.2, 6)],
        [("Brightness", 0.9, 6), ("Color", 0.2, 8)],
        [("Solarize", 0.5, 2), ("Invert", 0.0, 3)],
        [("Equalize", 0.2, 0), ("AutoContrast", 0.6, 0)],
        [("Equalize", 0.2, 8), ("Equalize", 0.6, 4)],
        [("Color", 0.9, 9), ("Equalize", 0.6, 6)],
        [("AutoContrast", 0.8, 4), ("Solarize", 0.2, 8)],
        [("Brightness", 0.1, 3), ("Color", 0.7, 0)],
        [("Solarize", 0.4, 5), ("AutoContrast", 0.9, 3)],
        [("TranslateY", 0.9, 9), ("TranslateY", 0.7, 9)],
        [("AutoContrast", 0.9, 2), ("Solarize", 0.8, 3)],
        [("Equalize", 0.8, 8), ("Invert", 0.1, 3)],
        [("TranslateY", 0.7, 9), ("AutoContrast", 0.9, 1)],
    ]
    return transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4, fill=FILLCOLOR),
            transforms.RandomHorizontalFlip(),
            AutoAugmentation(policies),
            SequentialAugmentation([("Cutout", 1.0, 9)]),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_INFO["MEAN"], CIFAR100_INFO["STD"]),
        ]
    )


def autoaugment_train_cifar100_riair() -> transforms.Compose:
    """RIAIR's Auto augmentation policy for training CIFAR100."""
    policies = [
        [("Invert", 0.2, 2)],
        [("Contrast", 0.4, 4)],
        [("Rotate", 0.5, 1)],
        [("TranslateX", 0.4, 3)],
        [("Sharpness", 0.5, 3)],
        [("ShearY", 0.3, 4)],
        [("TranslateY", 0.6, 8)],
        [("AutoContrast", 0.6, 3)],
        [("Equalize", 0.5, 5)],
        [("Solarize", 0.4, 4)],
        [("Color", 0.5, 5)],
        [("Posterize", 0.2, 2)],
        [("Brightness", 0.4, 5)],
        [("Cutout", 0.3, 3)],
        [("ShearX", 0.1, 3)],
    ]
    return transforms.Compose(
        [
            AutoAugmentation(policies, n_select=2),
            transforms.RandomCrop(32, padding=4, fill=FILLCOLOR),
            transforms.RandomHorizontalFlip(),
            SequentialAugmentation([("Cutout", 1.0, 9)]),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_INFO["MEAN"], CIFAR100_INFO["STD"]),
        ]
    )


def randaugment_train_cifar100(
    n_select: int = 2, level: int = 14, n_level: int = 31,
) -> transforms.Compose:
    """Random augmentation policy for training CIFAR100."""
    operators = [
        "Identity",
        "AutoContrast",
        "Equalize",
        "Rotate",
        "Solarize",
        "Color",
        "Posterize",
        "Contrast",
        "Brightness",
        "Sharpness",
        "ShearX",
        "ShearY",
        "TranslateX",
        "TranslateY",
    ]
    return transforms.Compose(
        [
            RandAugmentation(operators, n_select, level, n_level),
            transforms.RandomCrop(32, padding=4, fill=FILLCOLOR),
            transforms.RandomHorizontalFlip(),
            SequentialAugmentation([("Cutout", 1.0, 9)]),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_INFO["MEAN"], CIFAR100_INFO["STD"]),
        ]
    )


def randaugment_train_cifar100_224(
    n_select: int = 2, level: int = 14, n_level: int = 31,
) -> transforms.Compose:
    operators = [
        "Identity",
        "AutoContrast",
        "Equalize",
        "Rotate",
        "Solarize",
        "Color",
        "Posterize",
        "Contrast",
        "Brightness",
        "Sharpness",
        "ShearX",
        "ShearY",
        "TranslateX",
        "TranslateY",
    ]
    return transforms.Compose(
        [
            transforms.Resize(224),
            RandAugmentation(operators, n_select, level, n_level),
            transforms.RandomCrop(224, padding=4, fill=FILLCOLOR),
            transforms.RandomHorizontalFlip(),
            SequentialAugmentation([("Cutout", 1.0, 9)]),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_INFO["MEAN"], CIFAR100_INFO["STD"]),
        ]
    )


def randaugment_train_ai_challenge(
    n_select: int = 2, level: int = 14, n_level: int = 31, image_size: int = 224,
) -> transforms.Compose:
    operators = [
        "Identity",
        "AutoContrast",
        "Equalize",
        "Rotate",
        "Solarize",
        "Color",
        "Posterize",
        "Contrast",
        "Brightness",
        "Sharpness",
        "ShearX",
        "ShearY",
        "TranslateX",
        "TranslateY",
    ]
    return transforms.Compose(
        [
            transforms.Resize((int(image_size * 1.33), int(image_size * 1.33))),
            RandAugmentation(operators, n_select, level, n_level),
            transforms.RandomCrop(image_size, padding=4, fill=FILLCOLOR),
            transforms.RandomHorizontalFlip(),
            SequentialAugmentation([("Cutout", 1.0, 9)]),
            transforms.ToTensor(),
            transforms.Normalize(CHALLENGE_INFO["MEAN"], CHALLENGE_INFO["STD"]),
        ]
    )
