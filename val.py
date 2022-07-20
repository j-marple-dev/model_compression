"""Validation runner.

- Author: Haneol Kim
- Contact: hekim@jmarple.ai
"""

import argparse
import os

from src.runners import initialize
from src.runners.validator import Validator


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
        help="Decomposed model weight directory (e.g. decompose/220714_180306).",
    )
    parser.set_defaults(half=False)
    parser.set_defaults(multi_gpu=False)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_parser()

    if args.decomposed and args.decomp_dir:
        config, dir_prefix, device = initialize(
            "val", args.config, args.decomp_dir, args.multi_gpu, args.gpu
        )
    else:
        config, dir_prefix, device = initialize(
            "val", args.config, args.resume, args.multi_gpu, args.gpu
        )

    print(config)
    validator = Validator(
        config=config,
        dir_prefix=dir_prefix,
        checkpt_dir="train",
        device=device,
        half=args.half,
        decomposed=args.decomposed,
        weight_path=os.path.join(args.decomp_dir, "decomposed_model.pt"),
    )

    validator.run()
