# -*- coding: utf-8 -*-
"""Trainer for models.

- Author: Curt-Park
- Email: jwpark@jmarple.ai
"""

import logging
import os
from typing import Any, Callable, Dict, List, Tuple

from progressbar import progressbar
import torch
import torch.nn as nn
import torch.optim as optim
import wandb

from src.format import default_format, percent_format
from src.lr_schedulers import WarmupCosineLR
import src.utils as utils

logger = logging.getLogger()


class Trainer:
    """Trainer for models."""

    def __init__(
        self,
        config: Dict[str, Any],
        dir_prefix: str,
        model_dir: str,
        device: torch.device,
        wandb_log: bool,
        wandb_init_params: Dict[str, Any],
    ) -> None:
        """Initialize."""
        self.config = config
        self.device = device
        self.wandb_log = wandb_log
        self.reset(dir_prefix, model_dir)

        # create a model
        self.model = utils.get_model(config["MODEL_NAME"], config["MODEL_PARAMS"]).to(
            device
        )

        # create dataloaders
        self.trainloader, self.testloader = utils.get_dataset(
            config["BATCH_SIZE"],
            config["N_WORKERS"],
            config["DATASET"],
            config["AUG_TRAIN"],
            config["AUG_TEST"],
        )

        # define criterion and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.config["LR"],
            momentum=self.config["MOMENTUM"],
            weight_decay=self.config["WEIGHT_DECAY"],
        )

        # learning rate scheduler
        self.lr_scheduler = WarmupCosineLR(
            self.config["WARMUP_EPOCHS"],
            self.config["EPOCHS"],
            self.config["START_LR"],
            self.config["LR"],
        )

        # create logger
        if wandb_log:
            wandb.init(**wandb_init_params)

    def reset(self, dir_prefix: str, model_dir: str) -> None:
        """Reset the configurations."""
        self.dir_prefix = dir_prefix
        self.model_dir = model_dir
        self.best_acc = 0.0

        # best model path
        self.best_model_path = os.path.join(dir_prefix, model_dir)
        if not os.path.exists(self.best_model_path):
            os.mkdir(self.best_model_path)

    def run(self) -> None:
        """Train the model."""
        for epoch in range(self.config["EPOCHS"]):
            self.run_one_epoch(epoch)

    def run_one_epoch(
        self,
        epoch: int,
        extra_log_info: List[Tuple[str, float, Callable[[float], str]]] = None,
    ) -> None:
        """Train one epoch and run testing and logging."""
        self.lr_scheduler(self.optimizer, epoch)

        # train
        train_loss, train_acc = self.train_one_epoch()

        # test
        test_loss, test_acc = self.test_one_epoch()

        # save model
        if test_acc > self.best_acc:
            self.save_params(self.best_model_path, str(epoch), epoch, test_acc)
            self.best_acc = test_acc

        # log
        if not extra_log_info:
            extra_log_info = []
        log_info = []
        log_info.append(
            ("train/lr", self.optimizer.param_groups[0]["lr"], default_format)
        )
        log_info.append(("train/loss", train_loss, default_format))
        log_info.append(("train/acc", train_acc, percent_format))
        log_info.append(("test/loss", test_loss, default_format))
        log_info.append(("test/acc", test_acc, percent_format))
        log_info.append(("test/best_acc", self.best_acc, percent_format))
        self.log_one_epoch(epoch, log_info + extra_log_info)

    def log_one_epoch(
        self, epoch: int, log_info: List[Tuple[str, float, Callable[[float], str]]]
    ) -> None:
        """Log information after running one epoch."""
        log_str = f"Epoch: [{epoch} | {self.config['EPOCHS']-1}]\t"
        log_str += "\t".join([f"{name}: " + f(val) for name, val, f in log_info])
        logger.info(log_str)

        # logging
        if self.wandb_log:
            utils.wlog_weight(self.model)
            wandb.log(dict((name, val) for name, val, _ in log_info))

    def train_one_epoch(self) -> Tuple[float, float]:
        """Train one epoch."""
        correct = total = 0
        losses = []
        self.model.train()
        for data in progressbar(self.trainloader):
            # get the inputs; data is a list of [inputs, labels]
            images, labels = data[0].to(self.device), data[1].to(self.device)

            # zero the parameter gradients
            self.optimizer.zero_grad()

            # forward + backward + optimize
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())

            # compute accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        train_acc = 100.0 * correct / total

        avg_loss = sum(losses) / len(losses)
        return avg_loss, train_acc

    def test_one_epoch(self) -> Tuple[float, float]:
        """Test one epoch."""
        correct = total = 0
        losses = []
        self.model.eval()
        for data in progressbar(self.testloader):
            images, labels = data[0].to(self.device), data[1].to(self.device)
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            losses.append(loss.item())

            # compute accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        test_acc = 100.0 * correct / total

        avg_loss = sum(losses) / len(losses)
        return avg_loss, test_acc

    def save_params(
        self, model_path: str, filename: str, epoch: int, test_acc: float = 0.0
    ) -> None:
        """Save model."""
        utils.save_checkpoint(
            {
                "state_dict": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "test_acc": test_acc,
            },
            path=model_path,
            filename=f"{filename}.pth.tar",
        )
        logger.info(f"Saved model in {model_path}/{filename}.pth.tar")

    def load_params(self, model_path: str, with_mask=True) -> None:
        """Load weights and masks."""
        checkpt = torch.load(model_path)
        self.initialize_params(self.model, checkpt["state_dict"], with_mask=with_mask)
        self.initialize_params(self.optimizer, checkpt["optimizer"], with_mask=False)
        logger.info(f"Loaded parameters from {model_path}")

    def initialize_params(
        self, model: Any, state_dict: Dict[str, Any], with_mask=True
    ) -> None:
        """Initialize weights and masks."""
        model_dict = model.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {}
        for k, v in state_dict.items():
            if k in model_dict and (with_mask or "weight_mask" not in k):
                pretrained_dict[k] = v
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        model.load_state_dict(model_dict)


def run(
    config: Dict[str, Any],
    dir_prefix: str,
    device: torch.device,
    wandb_log: bool,
    wandb_init_params: Dict[str, Any],
) -> None:
    """Run training process."""
    trainer = Trainer(
        config=config,
        dir_prefix=dir_prefix,
        model_dir="train",
        device=device,
        wandb_log=wandb_log,
        wandb_init_params=wandb_init_params,
    )
    trainer.run()
