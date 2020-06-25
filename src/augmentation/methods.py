# -*- coding: utf-8 -*-
"""Augmentation methods.

- Author: Curt-Park
- Email: jwpark@jmarple.ai
- Reference:
    https://arxiv.org/pdf/1805.09501.pdf
    https://github.com/kakaobrain/fast-autoaugment/
"""

from abc import ABC
from itertools import chain
import random
from typing import List, Tuple

from PIL.Image import Image

from src.augmentation.transforms import transforms_info


class Augmentation(ABC):
    """Abstract class used by all augmentation methods."""

    def __init__(self, n_level: int = 10) -> None:
        """Initialize."""
        self.transforms_info = transforms_info()
        self.n_level = n_level

    def _apply_augment(self, img: Image, name: str, level: int) -> Image:
        """Apply and get the augmented image.

        Args:
            img (Image): an image to augment
            level (int): magnitude of augmentation in [0, n_level)

        returns:
            Image: an augmented image
        """
        assert 0 <= level < self.n_level
        augment_fn, low, high = self.transforms_info[name]
        return augment_fn(img.copy(), level * (high - low) / self.n_level + low)


class SequentialAugmentation(Augmentation):
    """Sequential augmentation class."""

    def __init__(
        self, policies: List[Tuple[str, float, int]], n_level: int = 10,
    ) -> None:
        """Initialize."""
        super(SequentialAugmentation, self).__init__(n_level)
        self.policies = policies

    def __call__(self, img: Image) -> Image:
        """Run augmentations."""
        for name, pr, level in self.policies:
            if random.random() > pr:
                continue
            img = self._apply_augment(img, name, level)
        return img


class AutoAugmentation(Augmentation):
    """Auto augmentation class.

    References:
        https://arxiv.org/pdf/1805.09501.pdf

    """

    def __init__(
        self,
        policies: List[List[Tuple[str, float, int]]],
        n_select: int = 1,
        n_level: int = 10,
    ) -> None:
        """Initialize."""
        super(AutoAugmentation, self).__init__(n_level)
        self.policies = policies
        self.n_select = n_select

    def __call__(self, img: Image) -> Image:
        """Run augmentations."""
        chosen_policies = random.sample(self.policies, k=self.n_select)
        for name, pr, level in chain.from_iterable(chosen_policies):
            if random.random() > pr:
                continue
            img = self._apply_augment(img, name, level)
        return img


class RandAugmentation(Augmentation):
    """Random augmentation class.

    References:
        RandAugment: Practical automated data augmentation with a reduced search space
        (https://arxiv.org/abs/1909.13719)

    """

    def __init__(
        self,
        transforms: List[str],
        n_select: int = 2,
        level: int = 3,
        n_level: int = 10,
    ) -> None:
        """Initialize."""
        super(RandAugmentation, self).__init__(n_level)
        self.n_select = n_select
        self.level = level
        self.transforms = transforms

    def __call__(self, img: Image) -> Image:
        """Run augmentations."""
        chosen_transforms = random.sample(self.transforms, k=self.n_select)
        for transf in chosen_transforms:
            img = self._apply_augment(img, transf, self.level)
        return img
