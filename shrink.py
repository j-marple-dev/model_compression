# -*- coding: utf-8 -*-
"""Shrink model, and save and run.

- Author: Junghoon Kim
- Email: jhkim@jmarple.ai
"""


import argparse
import os
import shutil

from src.runners import initialize
from src.runners.shrinker import Shrinker

# arguments
parser = argparse.ArgumentParser(description="Model shrinker.")
parser.add_argument("--gpu", default=0, type=int, help="GPU id to use")
parser.add_argument("--checkpoint", type=str, help="input checkpoint path to quantize")
parser.add_argument("--config", type=str, help="Pruning configuration path")
args = parser.parse_args()

# get config and directory path prefix for logging
config, dir_prefix, device = initialize(
    mode="shrink", config_path=args.config, gpu_id=args.gpu
)

assert args.checkpoint and os.path.exists(args.checkpoint), "--checkpoint required"
shutil.copyfile(args.checkpoint, os.path.join(dir_prefix, "orig_model.pth.tar"))

# run quantization
shrinker = Shrinker(
    config=config,
    checkpoint_path=args.checkpoint,
    dir_prefix=dir_prefix,
    device=device,
)
shrinker.run()
