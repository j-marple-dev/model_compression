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

        if "NESTEROV" in self.config:
            assert type(self.config["NESTEROV"]) is bool
        else:
            self.config["NESTEROV"] = False  # default

        if "CUTMIX" in self.config:
            cutmix_config = self.config["CUTMIX"]
            assert "beta" in cutmix_config
            assert cutmix_config["beta"] > 0
            assert "prob" in cutmix_config
            assert 0 < cutmix_config["prob"] <= 1

        if "AUG_TRAIN_PARAMS" in self.config:
            assert isinstance(self.config["AUG_TRAIN_PARAMS"], dict)
        else:
            self.config["AUG_TRAIN_PARAMS"] = dict()

        if "AUG_TEST_PARAMS" in self.config:
            assert isinstance(self.config["AUG_TEST_PARAMS"], dict)
        else:
            self.config["AUG_TEST_PARAMS"] = dict()

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

        ce_params = None
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

            # if HintonLoss contains crossentropy
            assert "crossentropy_params" in params
            ce_params = params["crossentropy_params"]

        elif self.config["CRITERION"] == "CrossEntropy":
            ce_params = self.config["CRITERION_PARAMS"]

        if ce_params:
            assert "num_classes" in ce_params
            assert ce_params["num_classes"] > 0
            assert isinstance(ce_params["num_classes"], int)

            if "label_smoothing" not in ce_params:
                ce_params["label_smoothing"] = 0.0
            else:
                assert 0.0 <= ce_params["label_smoothing"] < 1.0
                assert type(ce_params["label_smoothing"]) is float

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

        lr_scheduler_names = get_class_names_in_files(
            "src" + os.path.sep + "lr_schedulers.py"
        )
        lr_scheduler_names.remove("LrScheduler")

        # Check config regularizer exists
        assert self.config["LR_SCHEDULER"] in lr_scheduler_names
        assert "LR_SCHEDULER_PARAMS" in self.config
        assert isinstance(self.config["LR_SCHEDULER_PARAMS"], dict)

        if self.config["LR_SCHEDULER"] == "MultiStepLR":
            # milestones: list[int]
            assert "milestones" in self.config["LR_SCHEDULER_PARAMS"]
            for n in self.config["LR_SCHEDULER_PARAMS"]["milestones"]:
                assert isinstance(n, int)

            assert "gamma" in self.config["LR_SCHEDULER_PARAMS"]
            assert 0 < self.config["LR_SCHEDULER_PARAMS"]["gamma"] <= 1.0
            assert isinstance(self.config["LR_SCHEDULER_PARAMS"]["gamma"], float)

        elif self.config["LR_SCHEDULER"] == "WarmupCosineLR":
            # set epochs: int
            self.config["LR_SCHEDULER_PARAMS"]["epochs"] = self.config["EPOCHS"]

            # set target_lr: float
            self.config["LR_SCHEDULER_PARAMS"]["target_lr"] = self.config["LR"]

            # warmp_epochs
            assert "warmup_epochs" in self.config["LR_SCHEDULER_PARAMS"]
            assert (
                0
                <= self.config["LR_SCHEDULER_PARAMS"]["warmup_epochs"]
                <= self.config["EPOCHS"]
            )
            assert isinstance(self.config["LR_SCHEDULER_PARAMS"]["warmup_epochs"], int)

            # start_lr
            if self.config["LR_SCHEDULER_PARAMS"]["warmup_epochs"] != 0:
                assert "start_lr" in self.config["LR_SCHEDULER_PARAMS"]
                assert (
                    0
                    < self.config["LR_SCHEDULER_PARAMS"]["start_lr"]
                    <= self.config["LR"]
                )
                assert isinstance(self.config["LR_SCHEDULER_PARAMS"]["start_lr"], float)

            # n_rewinding
            if "n_rewinding" not in self.config["LR_SCHEDULER_PARAMS"]:
                self.config["LR_SCHEDULER_PARAMS"]["n_rewinding"] = 1
            else:
                assert type(self.config["LR_SCHEDULER_PARAMS"]["n_rewinding"]) is int
                assert self.config["LR_SCHEDULER_PARAMS"]["n_rewinding"] > 0
                assert (
                    self.config["EPOCHS"]
                    % self.config["LR_SCHEDULER_PARAMS"]["n_rewinding"]
                    == 0
                )

            # Check zero division in lr scheduling
            assert (
                self.config["EPOCHS"]
                // self.config["LR_SCHEDULER_PARAMS"]["n_rewinding"]
                > self.config["LR_SCHEDULER_PARAMS"]["warmup_epochs"]
            )

            # min_lr
            if "min_lr" not in self.config["LR_SCHEDULER_PARAMS"]:
                self.config["LR_SCHEDULER_PARAMS"]["min_lr"] = 0.0
            else:
                assert type(self.config["LR_SCHEDULER_PARAMS"]["min_lr"]) is float
                assert self.config["LR_SCHEDULER_PARAMS"]["min_lr"] >= 0.0

            # decay
            if "decay" not in self.config["LR_SCHEDULER_PARAMS"]:
                self.config["LR_SCHEDULER_PARAMS"]["decay"] = 0.0
            else:
                assert type(self.config["LR_SCHEDULER_PARAMS"]["decay"]) is float
                assert 0.0 <= self.config["LR_SCHEDULER_PARAMS"]["decay"] < 1.0


class PruneConfigValidator(ConfigValidator):
    """Config validation for prune config."""

    def __init__(self, config: Dict[str, Any], log: bool = True) -> None:
        """Initialize."""
        super().__init__(config, log)

        self.necessary_config_names = {
            "TRAIN_CONFIG",
            "N_PRUNING_ITER",
            "PRUNE_METHOD",
            "PRUNE_PARAMS",
        }

    def check(self) -> None:
        """Check configs are specified correctly."""
        # check existence
        self.check_key_exists()

        # validate training config
        TrainConfigValidator(self.config["TRAIN_CONFIG"], log=False).check()
        # if different training policy at prune is not specified
        if "TRAIN_CONFIG_AT_PRUNE" not in self.config:
            self.config["TRAIN_CONFIG_AT_PRUNE"] = self.config["TRAIN_CONFIG"]
        TrainConfigValidator(self.config["TRAIN_CONFIG_AT_PRUNE"], log=False).check()

        # validate prune config
        self.check_prune_methods()

        # if SEED is not specified, set it same as training config's SEED
        if "SEED" not in self.config:
            self.config["SEED"] = self.config["TRAIN_CONFIG"]["SEED"]

        assert 0 < self.config["N_PRUNING_ITER"]
        assert isinstance(self.config["N_PRUNING_ITER"], int)

    def check_prune_methods(self) -> None:
        """Check prune methods config validity."""
        # get criterion class lists
        pruner_names = get_class_names_in_files(
            "src" + os.path.sep + "runners" + os.path.sep + "pruner.py"
        )
        # Remove abstract class name
        pruner_names.remove("Pruner")
        pruner_names.remove("ChannelwisePruning")

        # Check pruner method in config exists
        assert self.config["PRUNE_METHOD"] in pruner_names

        # Common config
        assert "PRUNE_AMOUNT" in self.config["PRUNE_PARAMS"]
        assert 0.0 < self.config["PRUNE_PARAMS"]["PRUNE_AMOUNT"] < 1.0
        assert isinstance(self.config["PRUNE_PARAMS"]["PRUNE_AMOUNT"], float)

        assert "STORE_PARAM_BEFORE" in self.config["PRUNE_PARAMS"]
        assert (
            0
            <= self.config["PRUNE_PARAMS"]["STORE_PARAM_BEFORE"]
            <= self.config["TRAIN_CONFIG_AT_PRUNE"]["EPOCHS"]
        )
        assert isinstance(self.config["PRUNE_PARAMS"]["STORE_PARAM_BEFORE"], int)

        assert "TRAIN_START_FROM" in self.config["PRUNE_PARAMS"]
        assert (
            0
            <= self.config["PRUNE_PARAMS"]["TRAIN_START_FROM"]
            <= self.config["TRAIN_CONFIG_AT_PRUNE"]["EPOCHS"]
        )
        assert isinstance(self.config["PRUNE_PARAMS"]["TRAIN_START_FROM"], int)

        assert "PRUNE_AT_BEST" in self.config["PRUNE_PARAMS"]
        assert isinstance(self.config["PRUNE_PARAMS"]["PRUNE_AT_BEST"], bool)

        # Config for methods
        if (
            self.config["PRUNE_METHOD"] == "Magnitude"
            or self.config["PRUNE_METHOD"] == "SlimMagnitude"
        ):
            assert "NORM" in self.config["PRUNE_PARAMS"]
            # https://pytorch.org/docs/master/generated/torch.norm.html
            assert isinstance(self.config["PRUNE_PARAMS"]["NORM"], int) or self.config[
                "PRUNE_PARAMS"
            ]["NORM"] in ("fro", "nuc")


class QuantizeConfigValidator(TrainConfigValidator):
    """Config validation for quantization config."""

    def __init__(self, config: Dict[str, Any], log: bool = True) -> None:
        """Initialize."""
        super().__init__(config, log)

    def check(self) -> None:
        """Check configs are specified correctly."""
        # validate training config
        super().check()


class ShrinkConfigValidator(PruneConfigValidator):
    """Config validation for shrink config."""

    def __init__(self, config: Dict[str, Any], log: bool = True) -> None:
        """Initialize."""
        super().__init__(config, log)

    def check(self) -> None:
        """Check configs are specified correctly."""
        # validate pruning config
        super().check()

        assert self.config["TRAIN_CONFIG"]["MODEL_NAME"] in {
            "densenet",
            "quant_densenet",
            "simplenet",
            "quant_simplenet",
        }, f"{self.config['TRAIN_CONFIG']['MODEL_NAME']} is not supported"


def get_class_names_in_files(path: str) -> List[str]:
    """Read all class names in file."""
    with open(path) as file:
        module = ast.parse(file.read())
        return [node.name for node in module.body if isinstance(node, ast.ClassDef)]
