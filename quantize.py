# -*- coding: utf-8 -*-
"""Quantization Runner.

- Author: Curt-Park
- Email: jwpark@jmarple.ai
"""


import argparse
import os
import shutil

from src.runners import curr_time, initialize
from src.runners.quantizer import Quantizer

# arguments
parser = argparse.ArgumentParser(description="Model quantizer.")
parser.add_argument(
    "--resume", type=str, default="", help="Input log directory name to resume"
)
parser.add_argument(
    "--check-acc",
    dest="check_acc",
    action="store_true",
    help="Check inference accuracy",
)
parser.add_argument(
    "--wlog", dest="wlog", action="store_true", help="Turns on wandb logging"
)
parser.add_argument(
    "--static",
    dest="static",
    action="store_true",
    help="Post-training static quantization",
)
parser.add_argument(
    "--backend", type=str, default="fbgemm", help="pytorch quantization backend"
)
parser.add_argument("--config", type=str, help="Configuration path")
parser.add_argument("--checkpoint", type=str, help="input checkpoint path to quantize")
parser.set_defaults(check_acc=False)
parser.set_defaults(wlog=False)
parser.set_defaults(static=False)
args = parser.parse_args()

# get config and directory path prefix for logging
config, dir_prefix, _ = initialize("quantize", args.config, args.resume)

if not args.resume:
    assert args.checkpoint and os.path.exists(args.checkpoint), "--checkpoint required"
    checkpoint_path = args.checkpoint
    shutil.copyfile(args.checkpoint, os.path.join(dir_prefix, "orig_model.pth.tar"))
else:
    checkpoint_path = os.path.join(dir_prefix, "orig_model.pth.tar")

# wandb
wandb_name = curr_time
wandb_init_params = dict(config=config, name=wandb_name, group=args.config)

# run quantization
quantizer = Quantizer(
    config=config,
    checkpoint_path=checkpoint_path,
    dir_prefix=dir_prefix,
    static=args.static,
    check_acc=args.check_acc,
    backend=args.backend,
    wandb_log=args.wlog,
    wandb_init_params=wandb_init_params,
)
quantizer.run(args.resume)
