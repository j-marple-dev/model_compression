# -*- coding: utf-8 -*-
"""Training Runner.

- Author: Curt-Park
- Email: jwpark@jmarple.ai
"""


import argparse

from src.runners import curr_time, initialize
from src.runners.trainer import Trainer

# arguments
parser = argparse.ArgumentParser(description="Model trainer.")
parser.add_argument("--multi-gpu", action="store_true", help="Multi-GPU use")
parser.add_argument("--gpu", default=0, type=int, help="GPU id to use")
parser.add_argument(
    "--resume", type=str, default="", help="Input log directory name to resume"
)
parser.add_argument(
    "--wlog", dest="wlog", action="store_true", help="Turns on wandb logging"
)
parser.add_argument(
    "--config", type=str, default="config/train/simplenet.py", help="Configuration path"
)
parser.set_defaults(multi_gpu=False)
parser.set_defaults(wlog=False)
args = parser.parse_args()

# initialize
config, dir_prefix, device = initialize(
    "train", args.config, args.resume, args.multi_gpu, args.gpu
)

# run training
wandb_name = args.resume if args.resume else curr_time
wandb_init_params = dict(config=config, name=wandb_name, group=args.config)
trainer = Trainer(
    config=config,
    dir_prefix=dir_prefix,
    checkpt_dir="train",
    wandb_log=args.wlog,
    wandb_init_params=wandb_init_params,
    device=device,
)
trainer.run(args.resume)
