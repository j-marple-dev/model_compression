# -*- coding: utf-8 -*-
"""Quantization Runner.

- Author: Curt-Park
- Email: jwpark@jmarple.ai
"""


import argparse
import os

from src.quantization import initialize
from src.quantization.quantizer import Quantizer

# arguments
parser = argparse.ArgumentParser(description="Model quantizer.")
parser.add_argument("--config", type=str, required=True, help="Configuration path")
parser.add_argument(
    "--checkpoint", type=str, required=True, help="Input checkpoint path to quantize"
)
args = parser.parse_args()

assert os.path.exists(args.config)
assert os.path.exists(args.checkpoint)

# get config and directory path prefix for logging
config, dir_prefix = initialize(args.config)


# run quantization
quantizer = Quantizer(
    config=config, checkpoint_path=args.checkpoint, dir_prefix=dir_prefix
)
quantizer.run()
