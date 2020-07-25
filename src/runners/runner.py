# -*- coding: utf-8 -*-
"""Abstract Runner class which contains methods to implement.

- Author: Curt-Park
- Email: jwpark@jmarple.ai
"""

from abc import ABC, abstractmethod
import os
from typing import Any, Dict


class Runner(ABC):
    """Abstract class used by runners (e.g. trainer, pruner)."""

    def __init__(self, config: Dict[str, Any], dir_prefix: str) -> None:
        """Initialize."""
        self.config = config
        self.dir_prefix = dir_prefix
        self.fileext = "pth.tar"
        self.checkpt_paths = "checkpt_paths.log"

    @abstractmethod
    def run(self, resume_info_path: str = "") -> None:
        """Run the module."""
        pass

    def _fetch_latest_checkpt(self) -> str:
        """Fetch the latest checkpoint file path from the log file."""
        checkpt_paths = os.path.join(self.dir_prefix, self.checkpt_paths)
        if not os.path.exists(checkpt_paths):
            return ""
        latest_file_path = ""
        with open(checkpt_paths, "r") as checkpts:
            checkpts_list = checkpts.readlines()
            if checkpts_list:
                latest_file_path = checkpts_list[-1][:-1]  # w/o '\n'
        return latest_file_path
