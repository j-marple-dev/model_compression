# -*- coding: utf-8 -*-
"""Runner for model compression.

- Author: Curt-Park
- Email: jwpark@jmarple.ai
"""


import argparse
import datetime
import glob
import os
import shutil

import torch

import src.utils as utils

# arguments
parser = argparse.ArgumentParser(description="Model compression.")
parser.add_argument(
    "--module", type=str, default="pruner", help="Module to run (without .py)"
)
parser.add_argument(
    "--config", type=str, default="naive_lth_simple", help="Configuration name"
)
parser.add_argument(
    "--resume", type=str, default="", help="Input log directory name to resume"
)
parser.add_argument(
    "--wlog", dest="wlog", action="store_true", help="Turns on wandb logging"
)
parser.add_argument("--gpu", default=0, type=int, help="GPU id to use.")
parser.set_defaults(log=False)
args = parser.parse_args()

# setup device
device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

# create directories
curr_time = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
dir_to_create = ["", "data", "checkpoint"]
dir_to_create += [os.path.join("checkpoint", curr_time)] if not args.resume else []
for name in dir_to_create:
    path = os.path.join("save", name)
    if not os.path.exists(path):
        os.mkdir(path)

# resume or load existing configurations
checkpt_dir = os.path.join("save", "checkpoint")
if args.resume:
    dir_prefix = os.path.join(checkpt_dir, args.resume)
    assert os.path.exists(dir_prefix), f"{dir_prefix} does not exist"
    config_path = glob.glob(os.path.join(dir_prefix, "*.py"))[0]
    config_from = config_path.rsplit(".", 1)[0].replace(os.path.sep, ".")
else:
    dir_prefix = os.path.join(checkpt_dir, curr_time)
    config_name = f"{args.config}.py"
    shutil.copyfile(
        os.path.join("config", config_name), os.path.join(dir_prefix, f"{config_name}")
    )
    config_from = "config." + args.config
config = __import__(config_from, fromlist=["config"]).config

# set logger
utils.set_logger(filename=os.path.join(dir_prefix, f"{args.config}.log"))

# set random seed
utils.set_random_seed(config["SEED"])

# create a module to run
wandb_name = args.resume if args.resume else curr_time
wandb_init_params = dict(config=config, name=wandb_name, group=args.config)
module = __import__("src.runners." + args.module, fromlist=[args.module])

# run the module
module.run(
    config=config,
    dir_prefix=dir_prefix,
    device=device,
    wandb_init_params=wandb_init_params,
    wandb_log=args.wlog,
    resume_info_path=args.resume,
)
