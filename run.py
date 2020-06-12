# -*- coding: utf-8 -*-
"""Runner for model compression.

- Author: Curt-Park
- Email: jwpark@jmarple.ai
"""


import argparse
import datetime
import os
import shutil

import torch

from src.prune import Pruner
from src.train import Trainer
import src.utils as utils

# arguments
parser = argparse.ArgumentParser(description="Model compression.")
parser.add_argument(
    "--config", type=str, default="naive_lth_simple", help="Configuration name"
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
for name in ["", "data", "model", f"model/{curr_time}"]:
    path = os.path.join("save", name)
    if not os.path.exists(path):
        os.mkdir(path)
dir_prefix = f"save/model/{curr_time}"

# get logger
logger = utils.get_logger(filename=os.path.join(dir_prefix, f"{args.config}.log"))

# load and copy configurations
config = __import__("src.config." + args.config, fromlist=["config"]).config
shutil.copyfile(
    f"src/config/{args.config}.py", os.path.join(dir_prefix, f"{args.config}.config")
)

# set random seed
utils.set_random_seed(config["SEED"])

# create dataloaders
trainloader, testloader = utils.get_dataset(
    config["BATCH_SIZE"],
    config["N_WORKERS"],
    config["DATASET"],
    config["AUG_TRAIN"],
    config["AUG_TEST"],
)

# create an initial model
model = utils.get_model(config["MODEL_NAME"], config["MODEL_PARAMS"]).to(device)

# run pretraining
logger.info("Run pretraining")
wandb_init_params = dict(config=config, name=curr_time, group=args.config)
trainer = Trainer(
    model=model,
    trainloader=trainloader,
    testloader=testloader,
    config=config,
    dir_prefix=dir_prefix,
    model_dir="pretrain",
    device=device,
    logger=logger,
    wandb_log=args.wlog,
    wandb_init_params=wandb_init_params,
)
trainer.run({"sparsity": 0.0})

# run pruning module
logger.info("Run pruning")
pruner = Pruner(trainer=trainer, config=config, dir_prefix=dir_prefix, logger=logger)
pruner.run()
