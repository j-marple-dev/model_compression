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
import numpy as np
import torch
from torch.utils.data import Dataset

from src.augmentation.transforms import transforms_info
from src.utils import get_rand_bbox_coord, to_onehot


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
        self,
        policies: List[Tuple[str, float, int]],
        n_level: int = 10,
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
        level: int = 14,
        n_level: int = 31,
    ) -> None:
        """Initialize."""
        super(RandAugmentation, self).__init__(n_level)
        self.n_select = n_select
        self.level = level if type(level) is int and 0 <= level < n_level else None
        self.transforms = transforms

    def __call__(self, img: Image) -> Image:
        """Run augmentations."""
        chosen_transforms = random.sample(self.transforms, k=self.n_select)
        for transf in chosen_transforms:
            level = self.level if self.level else random.randint(0, self.n_level - 1)
            img = self._apply_augment(img, transf, level)
        return img


class CutMix(Dataset):
    """A Dataset class for CutMix.

    References:
        https://github.com/ildoonet/cutmix

    """

    def __init__(
        self, dataset: Dataset, num_classes: int, beta: float = 1.0, prob: float = 0.5
    ) -> None:
        self.dataset = dataset
        self.num_classes = num_classes
        self.beta = beta
        self.prob = prob

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert image and label to a cutmix image and label.

        Combine two training samples by cutting and pasting two images along a random box.
        The ground truth label is also "mixed" via the combination ratio.
        The combination ratio is sampled from a beta distribution.
        """
        img, label = self.dataset[index]  # label: int
        label = torch.tensor([label], dtype=torch.long)
        label_onehot = to_onehot(label, self.num_classes)
        # sampling the length ratio of random box to the image
        len_ratio = np.sqrt(np.random.beta(self.beta, self.beta))

        if random.random() > self.prob or len_ratio < 1e-3:
            return img, label_onehot.squeeze_(0)

        w, h = img.size()[-2], img.size()[-1]
        (x0, y0), (x1, y1) = get_rand_bbox_coord(w, h, len_ratio)
        # compute the combination ratio
        comb_ratio = (x1 - x0) * (y1 - y0) / (w * h)

        rand_ind = np.random.randint(len(self))
        rand_img, rand_label = self.dataset[rand_ind]
        rand_label = torch.tensor([rand_label], dtype=torch.long)
        img[:, x0:x1, y0:y1] = rand_img[:, x0:x1, y0:y1]
        label_onehot = (1 - comb_ratio) * label_onehot + comb_ratio * to_onehot(
            rand_label, self.num_classes
        )
        return img, label_onehot.squeeze_(0)

    def __len__(self) -> int:
        return len(self.dataset)
