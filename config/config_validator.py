# -*- coding: utf-8 -*-
"""Config validatior.

- Author: Junghoon Kim
- Email: jhkim@jmarple.ai
"""

from abc import ABC, abstractmethod
import ast
from typing import Any, Dict, Set

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
            "WEIGHT_DECAY",
            "LR",
            "CRITERION",
            "CRITERION_PARAMS",
            "WARMUP_EPOCHS",
            "N_WORKERS",
            "START_LR",
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

        assert 0 <= self.config["WARMUP_EPOCHS"] <= self.config["EPOCHS"]
        assert isinstance(self.config["WARMUP_EPOCHS"], int)

        assert self.config["LR"] > 0
        assert isinstance(self.config["LR"], float)

        assert 0 < self.config["START_LR"] <= self.config["LR"]
        assert isinstance(self.config["START_LR"], float)

        if "CUTMIX" in self.config:
            cutmix_config = self.config["CUTMIX"]
            assert "beta" in cutmix_config
            assert cutmix_config["beta"] > 0
            assert "prob" in cutmix_config
            assert 0 < cutmix_config["prob"] <= 1

        self.check_criterion()

    def check_criterion(self) -> None:
        """Check criterion config validity"""
        # get criterion class lists
        with open("src/losses.py") as file:
            node = ast.parse(file.read())
        criterion_names = [n.name for n in node.body if isinstance(n, ast.ClassDef)]
        # remove parent class(Loss)
        criterion_names.remove("Loss")

        # Check config criterion exists
        assert self.config["CRITERION"] in criterion_names
        check_criterion = self.config["CRITERION"]

        # Run criterion config check
        # Some criterion doesnt need any params
        if hasattr(self, check_criterion):
            """To run config check for criterion, make corresponding function name
            (e.g. HintonKLD -> HintonKLD)"""
            getattr(self, check_criterion)(self.config["CRITERION_PARAMS"])

    def HintonKLD(self, config: Dict[str, Any]) -> None:
        """Check HintonKLD Criterion params."""
        assert "T" in config
        assert config["T"] > 0.0
        assert isinstance(config["T"], float)

        assert "alpha" in config
        assert 0.0 <= config["alpha"] <= 1.0
        assert isinstance(config["alpha"], float)

        # check additional params(teacher) exist
        assert "teacher_model_name" in config
        assert isinstance("teacher_model_name", str)
        assert "teacher_model_params" in config

        # Hintonloss contains crossentropy
        self.CrossEntropy(config["crossentropy_params"])

    def CrossEntropy(self, config: Dict[str, Any]) -> None:
        """Check CrossEntropy Criterion params."""
        assert "num_classes" in config
        assert config["num_classes"] > 0
        assert isinstance(config["num_classes"], int)


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
