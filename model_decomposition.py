"""Tensor decomposition.

- Author: Haneol Kim
- Contact: hekim@jmarple.ai
"""

import argparse
from copy import deepcopy
import os
from pathlib import Path
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn

from src.logger import colorstr, get_logger
from src.runners import initialize
from src.runners.validator import Validator
from src.tensor_decomposition.decomposition import decompose_model
from src.utils import count_param

LOGGER = get_logger(__name__)


def get_parser() -> argparse.Namespace:
    """Get argument parser."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/train/cifar100/densenet_201.py",
        help="Configuration path (.py)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="Input log directory name to resume in save/checkpoint",
    )
    parser.add_argument("--gpu", default=0, type=int, help="GPU id to use")
    parser.add_argument("--multi-gpu", action="store_true", help="Multi-GPU use.")
    parser.add_argument(
        "--dst",
        type=str,
        default=os.path.join("exp", "decompose"),
        help="Export directory. Directory will be {dst}/decompose/{DATE}_runs1, ...",
    )
    parser.add_argument(
        "--prune-step",
        default=0.01,
        type=float,
        help="Prunning trial max step. Maximum step while searching prunning ratio with binary search. Pruning will be applied priro to decomposition. If prune-step is equal or smaller than 0.0, prunning will not be applied.",
    )
    parser.add_argument(
        "--loss-thr",
        default=0.1,
        type=float,
        help="Loss value to compare original model output and decomposed model output to judge to switch to decomposed conv.",
    )
    parser.add_argument(
        "--half", dest="half", action="store_true", help="Use half precision"
    )
    parser.add_argument(
        "--log",
        dest="log",
        action="store_true",
        help="Logging the tensor decomposition results.",
    )
    parser.add_argument(
        "--file_name",
        type=str,
        default="decomposed_model.pt",
        help="Decomposed model's file name",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.set_defaults(multi_gpu=False)
    parser.set_defaults(half=False)
    parser.set_defaults(log=False)
    return parser.parse_args()


def log_result(
    ori_param: int,
    decomp_param: int,
    ori_time: float,
    decomp_time: float,
    ori_result: float,
    decomp_result: float,
) -> Tuple[Dict[str, str], List[str]]:
    """Generate string for logging."""
    log_dict = {}
    dict_keys = [
        "ori_param",
        "decomp_param",
        "ori_time",
        "decomp_time",
        "ori_result",
        "decomp_result",
    ]
    log_dict.update({dict_keys[0]: f"  Original # of param    : {ori_param}"})
    log_dict.update({dict_keys[1]: f"Decomposed # of param    : {decomp_param}"})
    log_dict.update({dict_keys[2]: f"Time took (Original)     : {ori_time:.5f}s"})
    log_dict.update({dict_keys[3]: f"Time took (Decomposed)   : {decomp_time:.5f}s"})
    log_dict.update({dict_keys[4]: f"Original model accuray   : {ori_result}"})
    log_dict.update({dict_keys[5]: f"Decomposed model accuray : {decomp_result}"})

    return log_dict, dict_keys


def run_decompose(
    args: argparse.Namespace,
    validator: Validator,
    device: torch.device,
) -> Tuple[nn.Module, Tuple[Tuple[list, ...], np.ndarray, tuple]]:
    """Run tensor decomposition on given model.

    Args:
        args: arguments for the tensor decomposition.
            args.prune_step(float): prune step.
            args.loss_thr(float): Loss threshold for decomposition.
        model: Original model.
        validator: validation runner.
        device: device to run validation.

    Return:
        decomposed_model,
        (
            (mP, mR, mAP0.5, mAP0.5:0.95, 0, 0, 0),
            mAP0.5 by classes,
            time measured (pre-processing, inference, NMS)
        )
    """
    t0 = time.monotonic()
    ori_result = validator.run()[1]["model_acc"]
    origin_time_took = time.monotonic() - t0
    model = validator.model
    decomposed_model = deepcopy(validator.model.cpu())
    decompose_model(
        decomposed_model, loss_thr=args.loss_thr, prune_step=args.prune_step
    )

    LOGGER.info(
        f"Decomposed with prunning step: {args.prune_step}, decomposition loss threshold: {args.loss_thr}"
    )

    decomposed_model.to(device)
    decomposed_model.eval()

    validator.model = decomposed_model
    t0 = time.monotonic()
    decomposed_result = validator.run()[1]["model_acc"]
    decomposed_time_took = time.monotonic() - t0

    log_dict, log_keys = log_result(
        ori_param=count_param(model),
        decomp_param=count_param(decomposed_model),
        ori_time=origin_time_took,
        decomp_time=decomposed_time_took,
        ori_result=ori_result,
        decomp_result=decomposed_result,
    )

    for key in log_keys:
        LOGGER.info(log_dict[key])

    if args.log:
        log_file = os.path.join(args.resume, "decompose_log.txt")
        with open(log_file, "w") as f:
            f.writelines([log_dict[key] for key in log_keys])

    return decomposed_model, decomposed_result


if __name__ == "__main__":
    args = get_parser()

    torch.manual_seed(args.seed)
    LOGGER.info(f"Random Seed: {args.seed}")

    # initialize
    config, dir_prefix, device = initialize(
        "train", args.config, args.resume, args.multi_gpu, args.gpu
    )

    validator = Validator(
        config=config,
        dir_prefix=dir_prefix,
        device=device,
        half=args.half,
        checkpt_dir="train",
    )
    decomp_model, _ = run_decompose(args, validator, device)

    resume_dir = args.resume.split("/")[2]

    weight_dir = Path(os.path.join("decompose", resume_dir))
    weight_dir.mkdir(parents=True, exist_ok=True)
    filename = Path(args.file_name)

    weight_path = weight_dir / filename
    LOGGER.info(f"Decomposed model saved in {colorstr('cyan', 'bold', weight_path)}")

    torch.save({"model": decomp_model.cpu().half(), "decomposed": True}, weight_path)

    os.popen(f'cp {os.path.join(args.resume, "*.py")} {weight_dir}')
