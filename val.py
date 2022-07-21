"""Validation runner.

- Author: Haneol Kim
- Contact: hekim@jmarple.ai
"""

import argparse
import os

from src.logger import get_logger
from src.runners import initialize
from src.runners.validator import Validator

LOGGER = get_logger(__name__)


def get_parser() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--multi-gpu", action="store_true", help="Multi-GPU use")
    parser.add_argument("--gpu", default=0, type=int, help="GPU id to use")
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="Input log directory name to resume in save/checkpoint",
    )
    parser.add_argument(
        "--half", dest="half", action="store_true", help="Use half precision"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/train/cifar100/densenet_201.py",
        help="Configuration path (.py)",
    )
    parser.add_argument(
        "--decomp", dest="decomposed", action="store_true", help="Use decomposed model."
    )
    parser.add_argument(
        "--decomp_dir",
        type=str,
        default="",
        help="Decomposed model weight file path (e.g. decompose/220714_180306/weight.pt).",
    )
    parser.set_defaults(half=False)
    parser.set_defaults(multi_gpu=False)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_parser()

    if args.decomposed and args.decomp_dir:
        if args.decomp_dir.endswith(".pt"):
            decomp_dir = args.decomp_dir.split("/")
            weight = decomp_dir[-1]
            decomp_dir = os.path.join(decomp_dir[0], decomp_dir[1])
            config, dir_prefix, device = initialize(
                "val", args.config, decomp_dir, args.multi_gpu, args.gpu
            )
            weight_path = args.decomp_dir
        else:
            raise ValueError("The decomposed dir should be end with pt")
    else:
        config, dir_prefix, device = initialize(
            "val", args.config, args.resume, args.multi_gpu, args.gpu
        )
        weight_path = args.decomp_dir

    print(config)
    validator = Validator(
        config=config,
        dir_prefix=dir_prefix,
        checkpt_dir="train",
        device=device,
        half=args.half,
        decomposed=args.decomposed,
        weight_path=weight_path,
    )

    _, acc = validator.run()
    LOGGER.info(f"accuracy : {acc['model_acc']}%")
