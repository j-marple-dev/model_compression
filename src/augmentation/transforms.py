# -*- coding: utf-8 -*-
"""Image transformations for augmentation.

- Author: Curt-Park
- Email: jwpark@jmarple.ai
- Reference:
    https://github.com/kakaobrain/fast-autoaugment/
    https://github.com/DeepVoltaire/AutoAugment
"""

import random
from typing import Callable, Dict, Tuple

import PIL
from PIL.Image import Image
import PIL.ImageDraw
import PIL.ImageEnhance
import PIL.ImageOps

FILLCOLOR = (128, 128, 128)
FILLCOLOR_RGBA = (128, 128, 128, 128)


def transforms_info() -> Dict[
    str, Tuple[Callable[[Image, float], Image], float, float]
]:
    """Return augmentation functions and their ranges."""
    transforms_list = [
        (Invert, 0.0, 0.0),
        (Contrast, 0.0, 0.9),
        (AutoContrast, 0.0, 0.0),
        (Rotate, 0.0, 30.0),
        (TranslateX, 0.0, 150 / 331),
        (TranslateY, 0.0, 150 / 331),
        (Sharpness, 0.0, 0.9),
        (ShearX, 0.0, 0.3),
        (ShearY, 0.0, 0.3),
        (Color, 0.0, 0.9),
        (Brightness, 0.0, 0.9),
        (Equalize, 0.0, 0.0),
        (Solarize, 256.0, 0.0),
        (Posterize, 8, 4),
        (Cutout, 0, 0.5),
    ]
    return {f.__name__: (f, low, high) for f, low, high in transforms_list}


def Invert(img: Image, _: float) -> Image:
    """Invert the image."""
    return PIL.ImageOps.invert(img)


def Contrast(img: Image, magnitude: float) -> Image:
    """Put contrast effect on the image."""
    return PIL.ImageEnhance.Contrast(img).enhance(
        1 + magnitude * random.choice([-1, 1])
    )


def AutoContrast(img: Image, _: float) -> Image:
    """Put contrast effect on the image."""
    return PIL.ImageOps.autocontrast(img)


def Rotate(img: Image, magnitude: float) -> Image:
    """Rotate the image (degree)."""
    rot = img.convert("RGBA").rotate(magnitude)
    return PIL.Image.composite(
        rot, PIL.Image.new("RGBA", rot.size, FILLCOLOR_RGBA), rot
    ).convert(img.mode)


def TranslateX(img: Image, magnitude: float) -> Image:
    """Translate the image on x-axis."""
    return img.transform(
        img.size,
        PIL.Image.AFFINE,
        (1, 0, magnitude * img.size[0] * random.choice([-1, 1]), 0, 1, 0),
        fillcolor=FILLCOLOR,
    )


def TranslateY(img: Image, magnitude: float) -> Image:
    """Translate the image on y-axis."""
    return img.transform(
        img.size,
        PIL.Image.AFFINE,
        (1, 0, 0, 0, 1, magnitude * img.size[1] * random.choice([-1, 1])),
        fillcolor=FILLCOLOR,
    )


def Sharpness(img: Image, magnitude: float) -> Image:
    """Adjust the sharpness of the image."""
    return PIL.ImageEnhance.Sharpness(img).enhance(
        1 + magnitude * random.choice([-1, 1])
    )


def ShearX(img: Image, magnitude: float) -> Image:
    """Shear the image on x-axis."""
    return img.transform(
        img.size,
        PIL.Image.AFFINE,
        (1, magnitude * random.choice([-1, 1]), 0, 0, 1, 0),
        PIL.Image.BICUBIC,
        fillcolor=FILLCOLOR,
    )


def ShearY(img: Image, magnitude: float) -> Image:
    """Shear the image on y-axis."""
    return img.transform(
        img.size,
        PIL.Image.AFFINE,
        (1, 0, 0, magnitude * random.choice([-1, 1]), 1, 0),
        PIL.Image.BICUBIC,
        fillcolor=FILLCOLOR,
    )


def Color(img: Image, magnitude: float) -> Image:
    """Adjust the color balance of the image."""
    return PIL.ImageEnhance.Color(img).enhance(1 + magnitude * random.choice([-1, 1]))


def Brightness(img: Image, magnitude: float) -> Image:
    """Adjust brightness of the image."""
    return PIL.ImageEnhance.Brightness(img).enhance(
        1 + magnitude * random.choice([-1, 1])
    )


def Equalize(img: Image, _: float) -> Image:
    """Equalize the image."""
    return PIL.ImageOps.equalize(img)


def Solarize(img: Image, magnitude: float) -> Image:
    """Solarize the image."""
    return PIL.ImageOps.solarize(img, magnitude)


def Posterize(img: Image, magnitude: float) -> Image:
    """Posterize the image."""
    magnitude = int(magnitude)
    return PIL.ImageOps.posterize(img, magnitude)


def Cutout(img: Image, magnitude: float) -> Image:
    """Cutout some region of the image."""
    if magnitude == 0.0:
        return img
    w, h = img.size
    size_hole_w = int(magnitude * w)
    size_hole_h = int(magnitude * h)
    x = random.randint(0, w - 1)
    y = random.randint(0, h - 1)

    x0 = max(0, x - size_hole_w // 2)
    y0 = max(0, y - size_hole_h // 2)
    x1 = min(w, x + size_hole_w // 2)
    y1 = min(h, y + size_hole_h // 2)
    xy = [(x0, y0), (x1, y1)]

    img = img.copy()
    PIL.ImageDraw.Draw(img).rectangle(xy, fill=FILLCOLOR)
    return img
