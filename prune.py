# -*- coding: utf-8 -*-
"""Pruning Runner.

- Author: Curt-Park
- Email: jwpark@jmarple.ai
"""


import argparse

from src.runners import curr_time, initialize

# arguments
parser = argparse.ArgumentParser(description="Model pruner.")
parser.add_argument("--multi-gpu", action="store_true", help="Multi-GPU use")
parser.add_argument("--gpu", default=0, type=int, help="GPU id to use")
parser.add_argument(
    "--resume",
    type=str,
    default="",
    help="Input checkpoint directory name",
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
parser.add_argument("--test-weight", default="", help="Weight filepath to test")
parser.set_defaults(multi_gpu=False)
parser.set_defaults(log=False)
args = parser.parse_args()

# initialize
config, dir_prefix, device = initialize(
    "prune", args.config, args.resume, args.multi_gpu, args.gpu
)

# run pruning
wandb_name = args.resume if args.resume else curr_time
wandb_init_params = dict(config=config, name=wandb_name, group=args.config)
Pruner = getattr(
    __import__("src.runners.pruner", fromlist=[""]), config["PRUNE_METHOD"]
)

pruner = Pruner(
    config=config,
    dir_prefix=dir_prefix,
    wandb_log=args.wlog,
    wandb_init_params=wandb_init_params,
    device=device,
)
if args.test_weight:
    if not args.test_weight.startswith(args.resume):
        raise Exception(f"{args.test_weight} from {args.resume} ?")
    pruner.test(args.test_weight)
else:
    pruner.run(args.resume)
