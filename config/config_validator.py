# -*- coding: utf-8 -*-
"""Config validatior.

- Author: Junghoon Kim
- Email: jhkim@jmarple.ai
"""

from abc import ABC, abstractmethod
import ast
import os
from typing import Any, Dict, List, Set

import src.utils as utils

logger = utils.get_logger()


class ConfigValidator(ABC):
    """Abstract class for config validation."""

    def __init__(self, config: Dict[str, Any], log: bool = True) -> None:
        """Initialize."""
        super().__init__()
        self.config = config
        self.necessary_config_names: Set[str] = set()
        if log:
            self.log()

    @abstractmethod
    def check(self) -> None:
        """Check configs are specified correctly."""

        raise NotImplementedError

    def check_key_exists(self) -> None:
        """Check existence of keys."""
        omitted_configs = self.necessary_config_names - set(self.config.keys())
        assert len(omitted_configs) == 0, omitted_configs

    def log(self) -> None:
        """Log config data."""
        log_str = "[Config]\t"
        log_str += "\t".join([f"{k}: {v}" for k, v in self.config.items()])
        logger.info(log_str)


class TrainConfigValidator(ConfigValidator):
    """Config validation for train config."""

    def __init__(self, config: Dict[str, Any], log: bool = True) -> None:
        """Initialize."""
        super().__init__(config, log)

        self.necessary_config_names = {
            "AUG_TRAIN",
            "AUG_TEST",
            "DATASET",
            "MODEL_NAME",
            "MODEL_PARAMS",
            "MOMENTUM",
            "WEIGHT_DECAY",
            "SEED",
            "BATCH_SIZE",
            "EPOCHS",
            "LR",
            "CRITERION",
            "CRITERION_PARAMS",
            "N_WORKERS",
        }

    def check(self) -> None:
        """Check configs are specified correctly."""
        # check existence
        self.check_key_exists()

        # check valid range and type
        assert 0 <= self.config["MOMENTUM"] <= 1
        assert isinstance(self.config["MOMENTUM"], float)

        assert self.config["WEIGHT_DECAY"] >= 0
        assert isinstance(self.config["WEIGHT_DECAY"], float)

        assert self.config["SEED"] >= 0
        assert isinstance(self.config["SEED"], int)

        assert self.config["BATCH_SIZE"] > 0
        assert isinstance(self.config["BATCH_SIZE"], int)

        assert self.config["EPOCHS"] > 0
        assert isinstance(self.config["EPOCHS"], int)

        assert self.config["LR"] > 0
        assert isinstance(self.config["LR"], float)

        if "CUTMIX" in self.config:
            cutmix_config = self.config["CUTMIX"]
            assert "beta" in cutmix_config
            assert cutmix_config["beta"] > 0
            assert "prob" in cutmix_config
            assert 0 < cutmix_config["prob"] <= 1

        self.check_criterion()
        self.check_lr_schedulers()
        self.check_regularizer()

    def check_criterion(self) -> None:
        """Check criterion config validity."""
        # get criterion class lists
        criterion_names = get_class_names_in_files(
            "src" + os.path.sep + "criterions.py"
        )
        criterion_names.remove("Criterion")

        # Check config criterion exists
        assert self.config["CRITERION"] in criterion_names

        # Run criterion config check
        params: Dict[str, Any] = self.config["CRITERION_PARAMS"]

        if self.config["CRITERION"] == "HintonKLD":
            assert "T" in params
            assert params["T"] > 0.0
            assert isinstance(params["T"], float)

            assert "alpha" in params
            assert 0.0 <= params["alpha"] <= 1.0
            assert isinstance(params["alpha"], float)

            # check additional params(teacher) exist
            assert "teacher_model_name" in params
            assert isinstance("teacher_model_name", str)
            assert "teacher_model_params" in params

            assert "crossentropy_params" in params
            # Hintonloss contains crossentropy
            ce_params = params["crossentropy_params"]

        elif self.config["CRITERION"] == "CrossEntropy":
            ce_params = self.config["CRITERION_PARAMS"]

        if ce_params:
            assert "num_classes" in ce_params
            assert ce_params["num_classes"] > 0
            assert isinstance(ce_params["num_classes"], int)

    def check_regularizer(self) -> None:
        """Check regularizer config validity."""
        if "REGULARIZER" not in self.config:
            return None

        regularizer_names = get_class_names_in_files(
            "src" + os.path.sep + "regularizers.py"
        )

        # Check config regularizer exists
        assert self.config["REGULARIZER"] in regularizer_names

        # Run regularizer config check
        params: Dict[str, Any] = self.config["REGULARIZER_PARAMS"]

        if self.config["REGULARIZER"] == "BnWeight":
            assert "coeff" in params
            assert params["coeff"] > 0.0
            assert isinstance(params["coeff"], float)

    def check_lr_schedulers(self) -> None:
        """Check learning rate scheduler is valid."""
        # set default scheduler
        if (
            "LR_SCHEDULER" not in self.config
            or self.config["LR_SCHEDULER"] == "Identity"
        ):
            self.config["LR_SCHEDULER"] = "Identity"
            self.config["LR_SCHEDULER_PARAMS"] = dict()

        assert self.config["LR_SCHEDULER"] in {
            "Identity",
            "MultiStepLR",
            "WarmupCosineLR",
        }
        assert "LR_SCHEDULER_PARAMS" in self.config
        assert isinstance(self.config["LR_SCHEDULER_PARAMS"], dict)

        if self.config["LR_SCHEDULER"] == "MultiStepLR":
            # milestones: list[int]
            assert "milestones" in self.config["LR_SCHEDULER_PARAMS"]
            for n in self.config["LR_SCHEDULER_PARAMS"]["milestones"]:
                assert isinstance(n, int)

            # gamma: float
            assert "gamma" in self.config["LR_SCHEDULER_PARAMS"]
            assert 0 < self.config["LR_SCHEDULER_PARAMS"]["gamma"] <= 1.0
            assert isinstance(self.config["LR_SCHEDULER_PARAMS"]["gamma"], float)

        elif self.config["LR_SCHEDULER"] == "WarmupCosineLR":
            # set epochs: int
            self.config["LR_SCHEDULER_PARAMS"]["epochs"] = self.config["EPOCHS"]

            # set target_lr: float
            self.config["LR_SCHEDULER_PARAMS"]["target_lr"] = self.config["LR"]

            # warmp_epochs: int
            assert "warmup_epochs" in self.config["LR_SCHEDULER_PARAMS"]
            assert (
                0
                <= self.config["LR_SCHEDULER_PARAMS"]["warmup_epochs"]
                <= self.config["EPOCHS"]
            )
            assert isinstance(self.config["LR_SCHEDULER_PARAMS"]["warmup_epochs"], int)

            # start_lr: float
            if self.config["LR_SCHEDULER_PARAMS"]["warmup_epochs"] != 0:
                assert "start_lr" in self.config["LR_SCHEDULER_PARAMS"]
                assert (
                    0
                    < self.config["LR_SCHEDULER_PARAMS"]["start_lr"]
                    <= self.config["LR"]
                )
                assert isinstance(self.config["LR_SCHEDULER_PARAMS"]["start_lr"], float)


class PruneConfigValidator(ConfigValidator):
    """Config validation for prune config."""

    def __init__(self, config: Dict[str, Any], log: bool = True) -> None:
        """Initialize."""
        super().__init__(config, log=True)

        self.necessary_config_names = {
            "TRAIN_CONFIG",
            "N_PRUNING_ITER",
            "PRUNE_AMOUNT",
            "STORE_PARAM_BEFORE",
            "PRUNE_START_FROM",
        }

    def check(self) -> None:
        """Check configs are specified correctly."""
        # check existence
        self.check_key_exists()

        # validate training config
        TrainConfigValidator(self.config["TRAIN_CONFIG"], log=False).check()

        # if SEED is not specified, set it same as training config's SEED
        if "SEED" not in self.config:
            self.config["SEED"] = self.config["TRAIN_CONFIG"]["SEED"]

        # if EPOCHS is not specified, set it same as training config's EPOCHS
        if "EPOCHS" not in self.config:
            self.config["EPOCHS"] = self.config["TRAIN_CONFIG"]["EPOCHS"]
        # training config should contain the same epoch number in pruning config
        else:
            self.config["TRAIN_CONFIG"]["EPOCHS"] = self.config["EPOCHS"]

        # check valid range and type
        assert self.config["N_PRUNING_ITER"] > 0
        assert isinstance(self.config["N_PRUNING_ITER"], int)

        assert self.config["EPOCHS"] > 0
        assert isinstance(self.config["EPOCHS"], int)

        assert 0 < self.config["PRUNE_AMOUNT"] < 1.0
        assert isinstance(self.config["PRUNE_AMOUNT"], float)

        assert 0 <= self.config["STORE_PARAM_BEFORE"] <= self.config["EPOCHS"]
        assert isinstance(self.config["STORE_PARAM_BEFORE"], int)

        assert 0 <= self.config["PRUNE_START_FROM"] <= self.config["EPOCHS"]
        assert isinstance(self.config["PRUNE_START_FROM"], int)


class QuantConfigValidator(ConfigValidator):
    """Config validation for quantization config."""

    def __init__(self, config: Dict[str, Any], log: bool = True) -> None:
        """Initialize."""
        super().__init__(config, log)

        self.necessary_config_names = {
            "TRAIN_CONFIG",
            "EPOCHS",
            "QUANT_BACKEND",
        }

    def check(self) -> None:
        """Check configs are specified correctly."""
        # check existence
        self.check_key_exists()

        # validate training config
        TrainConfigValidator(self.config["TRAIN_CONFIG"], log=False).check()

        # if SEED is not specified, set it same as training config's SEED
        if "SEED" not in self.config:
            self.config["SEED"] = self.config["TRAIN_CONFIG"]["SEED"]

        # if EPOCHS is not specified, set it same as training config's EPOCHS
        if "EPOCHS" not in self.config:
            self.config["EPOCHS"] = self.config["TRAIN_CONFIG"]["EPOCHS"]
        # training config should contain the same epoch number in pruning config
        else:
            self.config["TRAIN_CONFIG"]["EPOCHS"] = self.config["EPOCHS"]

        assert self.config["QUANT_BACKEND"] in {"fbgemm", "qnnpack"}
        assert isinstance(self.config["QUANT_BACKEND"], str)


def get_class_names_in_files(path: str) -> List[str]:
    """Read all class names in file."""
    with open(path) as file:
        module = ast.parse(file.read())
        return [node.name for node in module.body if isinstance(node, ast.ClassDef)]
