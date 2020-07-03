# -*- coding: utf-8 -*-
"""Pruning Runner.

- Author: Curt-Park
- Email: jwpark@jmarple.ai
"""


import argparse

from src.pruning import curr_time, initialize
from src.pruning.pruner import Pruner

# arguments
parser = argparse.ArgumentParser(description="Model pruner.")
parser.add_argument("--gpu", default=0, type=int, help="GPU id to use")
parser.add_argument(
    "--resume", type=str, default="", help="Input log directory name to resume"
)
parser.add_argument(
    "--wlog", dest="wlog", action="store_true", help="Turns on wandb logging"
)
parser.add_argument(
    "--config",
    type=str,
    default="config/prune/simplenet_kd.py",
    help="Configuration path",
)
parser.set_defaults(log=False)
args = parser.parse_args()

# initialize
config, dir_prefix, device = initialize(args.config, args.resume, args.gpu)

# run pruning
wandb_name = args.resume if args.resume else curr_time
wandb_init_params = dict(config=config, name=wandb_name, group=args.config)
pruner = Pruner(
    config=config,
    dir_prefix=dir_prefix,
    wandb_log=args.wlog,
    wandb_init_params=wandb_init_params,
    device=device,
)
pruner.run(args.resume)
