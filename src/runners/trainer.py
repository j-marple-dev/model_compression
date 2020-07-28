# -*- coding: utf-8 -*-
"""Trainer for models.

- Author: Curt-Park
- Email: jwpark@jmarple.ai
"""

from collections import defaultdict
import os
from typing import Any, Callable, DefaultDict, Dict, List, Tuple

import pandas as pd
from progressbar import progressbar
from sklearn.metrics import f1_score, precision_score, recall_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb

from src.augmentation.methods import CutMix
from src.criterions import get_criterion
from src.format import default_format, percent_format
from src.lr_schedulers import get_lr_scheduler
from src.models import utils as model_utils
from src.regularizers import get_regularizer
from src.runners.runner import Runner
import src.utils as utils

logger = utils.get_logger()


class Trainer(Runner):
    """Trainer for models."""

    def __init__(
        self,
        config: Dict[str, Any],
        dir_prefix: str,
        checkpt_dir: str,
        wandb_log: bool,
        wandb_init_params: Dict[str, Any],
        device: torch.device,
        half: bool = False,
        test_preprocess_hook: Callable[[nn.Module], nn.Module] = None,
    ) -> None:
        """Initialize."""
        super(Trainer, self).__init__(config, dir_prefix)
        self.half = half
        self.device = device
        self.wandb_log = wandb_log
        self.reset(checkpt_dir)
        self.test_preprocess_hook = test_preprocess_hook

        # create a model
        model_name = self.config["MODEL_NAME"]
        model_config = self.config["MODEL_PARAMS"]
        self.model = model_utils.get_model(model_name, model_config).to(self.device)
        if device == torch.device("cuda"):  # multi-gpu
            self.model = torch.nn.DataParallel(self.model).to(self.device)
        if self.half:
            self.model.half()
        n_params = model_utils.count_model_params(self.model)
        torch.save(self.model.state_dict(), os.path.join(self.dir_prefix, "init.pth"))
        logger.info(
            f"Created a model {self.config['MODEL_NAME']} with {(n_params / 10**6):.2f}M params"
        )

        logger.info("Setup train configuration")
        self.setup_train_configuration(self.config)

        # change this if you want another acc metric for the best
        # {"precision", "recall", "f1"}
        self.acc_metric = self.config["BEST_ACC_METRIC"]

        # create logger
        if wandb_log:
            wandb.init(**wandb_init_params)

        self.n_correct_epoch: DefaultDict[str, int] = defaultdict(lambda: 0)
        self.labels_stat: DefaultDict[str, List[int]] = defaultdict(lambda: [])
        self.preds_stat: DefaultDict[str, List[int]] = defaultdict(lambda: [])

    def setup_train_configuration(self, config: Dict[str, Any]) -> None:
        """Setup train configuration."""
        self.config = config
        self.total_epochs = self.config["EPOCHS"]

        # get datasets
        trainset, testset = utils.get_dataset(
            config["DATASET"],
            config["AUG_TRAIN"],
            config["AUG_TEST"],
            config["AUG_TRAIN_PARAMS"],
            config["AUG_TEST_PARAMS"],
        )
        self.input_size = trainset[0][0].size()
        logger.info("Datasets prepared")

        # transform the training dataset for CutMix augmentation
        if "CUTMIX" in config:
            trainset = CutMix(
                trainset, config["MODEL_PARAMS"]["num_classes"], **config["CUTMIX"],
            )

        # get dataloaders
        self.trainloader, self.testloader = utils.get_dataloader(
            trainset, testset, config["BATCH_SIZE"], config["N_WORKERS"],
        )
        logger.info("Dataloader prepared")

        self.test_img_paths = dict()
        # get image file path (ImageFolder only)
        if hasattr(testset, "samples"):
            n_testset = len(testset)
            self.test_img_paths = {
                "paths": [testset.samples[idx][0] for idx in range(n_testset)]
            }

        # define criterion and optimizer
        self.criterion = get_criterion(
            criterion_name=config["CRITERION"],
            criterion_params=config["CRITERION_PARAMS"],
            device=self.device,
        )

        self.regularizer = None
        if "REGULARIZER" in config:
            self.regularizer = get_regularizer(
                config["REGULARIZER"], config["REGULARIZER_PARAMS"]
            )

        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=config["LR"],
            momentum=config["MOMENTUM"],
            weight_decay=config["WEIGHT_DECAY"],
            nesterov=config["NESTEROV"],
        )

        # learning rate scheduler
        self.lr_scheduler = get_lr_scheduler(
            config["LR_SCHEDULER"], config["LR_SCHEDULER_PARAMS"],
        )

    def reset(self, checkpt_dir: str) -> None:
        """Reset the configurations."""
        self.checkpt_dir = checkpt_dir
        self.best_acc = 0.0
        self.epoch = 0

        # best model path
        self.model_save_dir = os.path.join(self.dir_prefix, checkpt_dir)
        if not os.path.exists(self.model_save_dir):
            os.mkdir(self.model_save_dir)

    def get_model_save_dir(self) -> str:
        """Get model save directory."""
        return self.model_save_dir

    def resume(self) -> int:
        """Setting to resume the training."""
        last_epoch = -1
        latest_file_path = self._fetch_latest_checkpt()
        if latest_file_path and os.path.exists(latest_file_path):
            self.load_params(latest_file_path)
            _, self.checkpt_dir, filename = latest_file_path.rsplit(os.path.sep, 2)
            # fetch the last epoch from the filename
            last_epoch = int(filename.split("_", 1)[0])
        return last_epoch + 1

    def load_best_model(self) -> None:
        """Load current best model."""
        self.resume()

    def run(self, resume_info_path: str = "") -> None:
        """Train the model."""
        # resume trainer if needed
        start_epoch = 0
        if resume_info_path:
            start_epoch = self.resume()

        for self.epoch in range(start_epoch, self.config["EPOCHS"]):
            self.run_one_epoch(self.epoch)

    def run_one_epoch(
        self,
        epoch: int,
        extra_log_info: List[Tuple[str, float, Callable[[float], str]]] = None,
    ) -> None:
        """Train one epoch and run testing and logging."""
        self.lr_scheduler(self.optimizer, epoch)

        # train
        train_loss, train_stat = self.train_one_epoch()

        # test
        test_loss, test_stat = self.test_one_epoch()

        # save all params that showed the best acc
        test_acc = test_stat["model_" + self.acc_metric]
        if test_acc > self.best_acc:
            self.best_acc = test_acc
            filename = str(epoch) + "_" + f"{self.best_acc:.2f}".replace(".", "_")
            self.save_params(self.model_save_dir, filename, epoch, test_acc)

        # save the model every epoch
        model_path = os.path.join(self.model_save_dir, f"{epoch}.pth")
        logger.info(f"Saved model as {model_path}")
        torch.save(self.model.state_dict(), model_path)

        # log
        if not extra_log_info:
            extra_log_info = []
        lr = self.optimizer.param_groups[0]["lr"]
        log_info: List[Tuple[str, float, Callable[[float], str]]] = []
        log_info.append(("train/lr", lr, default_format))
        log_info.append(("train/loss", train_loss, default_format))
        log_info += [("train/" + k, v, percent_format) for k, v in train_stat.items()]
        log_info.append(("test/loss", test_loss, default_format))
        log_info += [("test/" + k, v, percent_format) for k, v in test_stat.items()]
        log_info.append(("test/best_" + self.acc_metric, self.best_acc, percent_format))
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
            model_utils.wlog_weight(self.model)
            wandb.log(dict((name, val) for name, val, _ in log_info))

    def count_correct_prediction(
        self, logits: Dict[str, torch.Tensor], labels: torch.Tensor
    ) -> None:
        """Count correct prediction in one iteration."""
        if len(labels.size()) != 1:  # For e.g., CutMix labels
            return
        for module_name, logit in logits.items():
            _, predicted = torch.max(F.softmax(logit, dim=1).data, 1)
            n_correct = int((predicted == labels).sum().cpu())
            self.n_correct_epoch[module_name] += n_correct
            self.labels_stat[module_name] += labels.clone().detach().tolist()
            self.preds_stat[module_name] += predicted.clone().detach().tolist()

    def get_epoch_statistics(self, is_test: bool = False) -> Dict[str, float]:
        """Get accuracy, f1_score, cofusion matrix, and then reset statistics."""
        stat = dict()
        for module_name in self.n_correct_epoch:
            precision = precision_score(
                self.labels_stat[module_name],
                self.preds_stat[module_name],
                average="macro",
            ).mean()
            precision *= 100
            recall = recall_score(
                self.labels_stat[module_name],
                self.preds_stat[module_name],
                average="macro",
            ).mean()
            recall *= 100
            f1 = f1_score(
                self.labels_stat[module_name],
                self.preds_stat[module_name],
                average="macro",
            ).mean()
            f1 *= 100
            stat.update({module_name + "_recall": recall})
            stat.update({module_name + "_precision": precision})
            stat.update({module_name + "_f1": f1})
        self.n_correct_epoch.clear()
        self.labels_stat.clear()
        self.preds_stat.clear()

        return stat

    def save_inference_results_csv(self) -> None:
        """Save inference results at test."""
        for module_name in self.preds_stat:
            # columns: img_paths / labels / predictions
            if self.test_img_paths:
                test_results = self.test_img_paths
            else:
                test_results = dict()
            test_results.update(
                dict(
                    labels=self.labels_stat[module_name],
                    preds=self.preds_stat[module_name],
                )
            )

            # create a csv file
            df = pd.DataFrame(test_results)
            filename = os.path.join(
                self.model_save_dir, f"{self.epoch}_" + module_name + "_pred"
            )
            filename += ".csv"
            df.to_csv(filename)

    def train_one_epoch(self) -> Tuple[float, Dict[str, float]]:
        """Train one epoch."""
        losses = []
        self.model.train()
        # trainloaders contain same length(iteration) of batch dataset
        for data in progressbar(self.trainloader, prefix="[Train]\t"):
            # get the inputs; data is a list of [inputs, labels]
            images, labels = data[0].to(self.device), data[1].to(self.device)
            if self.half:
                images = images.half()

            # zero the parameter gradients
            self.optimizer.zero_grad()

            # forward + backward + optimize
            loss, outputs = self.criterion(self.model, images=images, labels=labels)
            if self.regularizer:
                loss += self.regularizer(self.model)
            self.count_correct_prediction(outputs, labels)
            loss.backward()
            self.optimizer.step()

            losses.append(loss.item())

        avg_loss = sum(losses) / len(losses)
        acc = self.get_epoch_statistics()
        return avg_loss, acc

    def test_one_epoch(self) -> Tuple[float, Dict[str, float]]:
        """Test one epoch."""
        model = self.model
        if self.test_preprocess_hook:
            model = self.test_preprocess_hook(self.model)
        return self.test_one_epoch_model(model)

    @torch.no_grad()
    def test_one_epoch_model(self, model: nn.Module) -> Tuple[float, Dict[str, float]]:
        """Test the input model."""
        losses = []
        model.eval()
        # testloaders contain same length(iteration) of batch dataset
        for data in progressbar(self.testloader, prefix="[Test]\t"):
            images, labels = data[0].to(self.device), data[1].to(self.device)

            if self.half:
                images = images.half()

            # forward + backward + optimize
            loss, outputs = self.criterion(model, images=images, labels=labels)

            self.count_correct_prediction(outputs, labels)

            losses.append(loss.item())

        avg_loss = sum(losses) / len(losses)
        self.save_inference_results_csv()
        acc = self.get_epoch_statistics(is_test=True)
        return avg_loss, acc

    @torch.no_grad()
    def warmup_one_iter(self) -> None:
        """Run one iter for wramup."""
        self.model.eval()
        for batch_data in self.testloaders[0]:
            images, labels = (
                batch_data[0].to(self.device),
                batch_data[1].to(self.device),
            )

            # forward + backward + optimize
            loss, outputs = self.criterion(
                model=self.model, images=images, labels=labels
            )
            return None

    def save_params(
        self,
        model_path: str,
        filename: str,
        epoch: int,
        test_acc: float = 0.0,
        record_path: bool = True,
    ) -> None:
        """Save model."""
        params = {
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "test_acc": test_acc,
        }

        filepath = os.path.join(model_path, f"{filename}.{self.fileext}")
        torch.save(params, filepath)
        logger.info(
            f"Saved all params in {model_path}{os.path.sep}{filename}.{self.fileext}"
        )

        if record_path:
            with open(
                os.path.join(self.dir_prefix, self.checkpt_paths), "a"
            ) as checkpts:
                checkpts.write(filepath + "\n")

    def load_params(self, model_path: str, with_mask=True) -> None:
        """Load weights and masks."""
        checkpt = torch.load(model_path, map_location=self.device)
        model_utils.initialize_params(
            self.model, checkpt["state_dict"], with_mask=with_mask
        )
        model_utils.initialize_params(
            self.optimizer, checkpt["optimizer"], with_mask=False
        )
        self.best_acc = checkpt["test_acc"]
        logger.info(f"Loaded parameters from {model_path}")
