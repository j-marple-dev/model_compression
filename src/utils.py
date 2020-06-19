# -*- coding: utf-8 -*-
"""Utils for model compression.

- Author: Curt-Park
- Email: jwpark@jmarple.ai
"""

import glob
import logging
import logging.handlers
import os
import random
import re
import sys

import numpy as np
import torch


def set_random_seed(seed: int):
    """Set random seed."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    # for CuDNN backend
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True  # type: ignore
        torch.backends.cudnn.benchmark = False  # type: ignore


def set_logger(
    filename: str,
    mode: str = "a",
    level: int = logging.DEBUG,
    maxbytes: int = 1024 * 1024 * 10,  # default: 10Mbyte
    backupcnt: int = 100,
) -> None:
    """Create and get the logger for the console and files."""
    logger = logging.getLogger("model_compression")
    logger.setLevel(level)

    chdlr = logging.StreamHandler(sys.stdout)
    chdlr.setLevel(logging.DEBUG)
    cfmts = "%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s"
    chdlr.setFormatter(logging.Formatter(cfmts))
    logger.addHandler(chdlr)

    fhdlr = logging.handlers.RotatingFileHandler(
        filename, mode=mode, maxBytes=maxbytes, backupCount=backupcnt
    )
    fhdlr.setLevel(logging.DEBUG)
    ffmts = "%(asctime)s - "
    ffmts += "%(processName)s - %(threadName)s - "
    ffmts += "%(filename)s:%(lineno)d - %(levelname)s - %(message)s"
    fhdlr.setFormatter(logging.Formatter(ffmts))
    logger.addHandler(fhdlr)


def get_logger() -> logging.Logger:
    """Get logger instance."""
    return logging.getLogger("model_compression")


def get_latest_file(filepath: str, pattern: str = "*") -> str:
    """Get the latest file from the input filepath."""
    filelist = glob.glob(os.path.join(filepath, pattern))
    return max(filelist, key=os.path.getctime) if filelist else ""


def dot2bracket(s: str) -> str:
    """Replace layer names with valid names for pruning.

    Test:
       >>> dot2bracket("dense2.1.bn1.bias")
       'dense2[1].bn1.bias'
       >>> dot2bracket("dense2.13.bn1.bias")
       'dense2[13].bn1.bias'
       >>> dot2bracket("conv2.123.bn1.bias")
       'conv2[123].bn1.bias'
       >>> dot2bracket("dense2.6.conv2.5.bn1.bias")
       'dense2[6].conv2[5].bn1.bias'
       >>> dot2bracket("model.6")
       'model[6]'
       >>> dot2bracket("vgg.2.conv2.bn.2")
       'vgg[2].conv2.bn[2]'
       >>> dot2bracket("features.11")
       'features[11]'
    """
    pattern = r"\.[0-9]+"
    s_list = list(s)
    for m in re.finditer(pattern, s):
        start, end = m.span()
        s_list[start] = "["
        if end < len(s) and s_list[end] == ".":
            s_list[end] = "]."
        else:
            s_list.insert(end, "]")
    return "".join(s_list)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
