"""Validator for models.

- Author: Haneol Kim.
- Contact: hekim@jmarple.ai
"""

from collections import defaultdict
import os
from typing import Any, DefaultDict, Dict, Optional, Tuple, Union

from progressbar import progressbar
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from src.criterions import get_criterion
from src.logger import get_logger
from src.models import utils as model_utils
from src.regularizers import get_regularizer
from src.runners.runner import Runner
from src.utils import count_param, get_dataloader, get_dataset, select_device

LOGGER = get_logger(__name__)


class Validator(Runner):
    """Validator for models."""

    def __init__(
        self,
        config: Dict[str, Any],
        dir_prefix: str,
        checkpt_dir: str,
        device: Union[str, torch.device] = "cpu",
        half: bool = False,
        decomposed: bool = False,
        weight_path: Optional[str] = None,
    ) -> None:
        """Initialize vaildator."""
        if decomposed and weight_path is None:
            raise ValueError("If decomposed, the weight_path should be given.")
        elif not decomposed and weight_path:
            decomposed = True

        super(Validator, self).__init__(config, dir_prefix)
        if isinstance(device, torch.device):
            self.device = device
        else:
            self.device = select_device(device)
        self.half = half

        self.decomposed = decomposed
        self.weight_path = weight_path

        # create a model
        if self.decomposed and self.weight_path:
            self.model = model_utils.load_decomposed_model(self.weight_path)
            self.model.to(self.device)
            if device == torch.device("cuda"):
                self.model = torch.nn.DataParallel(self.model).to(self.device)

        else:
            model_name = self.config["MODEL_NAME"]
            model_config = self.config["MODEL_PARAMS"]
            self.model = model_utils.get_model(model_name, model_config).to(self.device)
            if device == torch.device("cuda"):  # multi-gpu
                self.model = torch.nn.DataParallel(self.model).to(self.device)

        if self.half:
            self.model.half()

        self.setup_val_configuration()
        self.n_correct_epoch: DefaultDict[str, int] = defaultdict(lambda: 0)

        LOGGER.info(f"Model parameters: {count_param(self.model)}")

    def setup_val_configuration(self) -> None:
        """Set up validation configuration."""
        # get datasets
        trainset, testset = get_dataset(
            self.config["DATASET"],
            self.config["AUG_TRAIN"],
            self.config["AUG_TEST"],
            self.config["AUG_TRAIN_PARAMS"],
            self.config["AUG_TEST_PARAMS"],
        )

        self.input_size = trainset[0][0].size()
        LOGGER.info("Datasets prepared")

        _, self.testloader = get_dataloader(
            trainset,
            testset,
            self.config["BATCH_SIZE"],
            self.config["N_WORKERS"],
        )
        LOGGER.info("Dataloader prepared")

        # define criterion and optimizer
        self.criterion = get_criterion(
            criterion_name=self.config["CRITERION"],
            criterion_params=self.config["CRITERION_PARAMS"],
            device=self.device,
        )

        self.regularizer = None
        if "REGULARIZER" in self.config:
            self.regularizer = get_regularizer(
                self.config["REGULARIZER"], self.config["REGULARIZER_PARAMS"]
            )

        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.config["LR"],
            momentum=self.config["MOMENTUM"],
            weight_decay=self.config["WEIGHT_DECAY"],
            nesterov=self.config["NESTEROV"],
        )
        if not self.decomposed:
            self.resume()

    def run(self) -> Tuple[float, dict]:
        """Train the model."""
        # resume trainer if needed
        test_loss, acc = self.test_one_epoch()
        # LOGGER.info(f"loss : {test_loss}, accuracy : {acc['model_acc']}%")
        return test_loss, acc

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
            self._count_correct_prediction(outputs, labels)

            losses.append(loss.item())

        avg_loss = sum(losses) / len(losses)
        acc = self._get_epoch_acc(is_test=True)
        return avg_loss, acc

    def test_one_epoch(self) -> Tuple[float, Dict[str, float]]:
        """Test one epoch."""
        model = self.model
        return self.test_one_epoch_model(model)

    def _count_correct_prediction(
        self, logits: Dict[str, torch.Tensor], labels: torch.Tensor
    ) -> None:
        """Count correct prediction in one iteration."""
        if len(labels.size()) != 1:  # For e.g., CutMix labels
            return
        for module_name, logit in logits.items():
            _, predicted = torch.max(F.softmax(logit, dim=1).data, 1)
            n_correct = int((predicted == labels).sum().cpu())
            self.n_correct_epoch[module_name] += n_correct

    def load_model(self, model_path: str, with_mask: bool = True) -> None:
        """Load weights and masks."""
        checkpt = torch.load(model_path, map_location=self.device)

        model_utils.initialize_params(
            self.model, checkpt["state_dict"], with_mask=with_mask
        )
        LOGGER.info(f"Loaded the model from {model_path}")

    def _get_epoch_acc(self, is_test: bool = False) -> Dict[str, float]:
        """Get accuracy and reset statistics."""
        n_total = (
            len(self.testloader.dataset) if is_test else len(self.trainloader.dataset)
        )
        acc = dict()
        for module_name in self.n_correct_epoch:
            accuracy = 100 * self.n_correct_epoch[module_name] / n_total
            acc.update({module_name + "_acc": accuracy})
        self.n_correct_epoch.clear()

        return acc

    def resume(self) -> int:
        """Set to resume the training."""
        last_epoch = -1
        latest_file_path = self._fetch_latest_checkpt()
        if latest_file_path and os.path.exists(latest_file_path):
            self.load_params(latest_file_path)
            _, self.checkpt_dir, filename = latest_file_path.rsplit(os.path.sep, 2)
            # fetch the last epoch from the filename
            last_epoch = int(filename.split("_", 1)[0])
        return last_epoch + 1

    def load_params(self, model_path: str, with_mask: bool = True) -> None:
        """Load weights and masks."""
        checkpt = torch.load(model_path, map_location=self.device)
        model_utils.initialize_params(
            self.model, checkpt["state_dict"], with_mask=with_mask
        )
        model_utils.initialize_params(
            self.optimizer, checkpt["optimizer"], with_mask=False
        )
        self.best_acc = checkpt["test_acc"]
        LOGGER.info(f"Loaded parameters from {model_path}")
