# -*- coding: utf-8 -*-
"""Utils for handling models.

- Author: Curt-Park
- Email: jwpark@jmarple.ai
"""

from collections import OrderedDict
import hashlib
import os
import re
import tarfile
from typing import Any, Dict, List, Optional, Set, Tuple

import gdown
import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import wandb
import yaml  # type: ignore

from src.logger import colorstr, get_logger

LOGGER = get_logger(__name__)


def get_model(model_name: str, model_config: Dict[str, Any]) -> nn.Module:
    """Get PyTorch model."""
    # get model constructor
    return __import__("src.models." + model_name, fromlist=[model_name]).get_model(
        **model_config
    )


def load_decomposed_model(
    weight_path: str, model_cfg_path: str = "", load_ema: bool = True
) -> Optional[nn.Module]:
    """Load PyTorch model.

    Args:
        weight_path: weight path which ends with .pt
        model_cfg_path: if provided, the model will first construct by the model_cfg,
                        and transfer weights to the constructed model.
                        In case of model_cfg_path was provided but not weight_path,
                        the model weights will be randomly initialized
                        (for experiment purpose).
        load_ema: load EMA weights if possible.
    Return:
        PyTorch model,
        None if loading PyTorch model has failed.
    """
    if weight_path == "":
        LOGGER.warning(
            "Providing "
            + colorstr("bold", "no weights path")
            + " will validate a randomly initialized model. Please use only for a experiment purpose."
        )
    else:
        ckpt = torch.load(weight_path)
        if isinstance(ckpt, dict):
            model_key = (
                "ema"
                if load_ema and "ema" in ckpt.keys() and ckpt["ema"] is not None
                else "model"
            )
            ckpt_model = ckpt[model_key]
        elif isinstance(ckpt, nn.Module):
            ckpt_model = ckpt

        ckpt_model = ckpt_model.cpu().float()

    if ckpt_model is None and model_cfg_path == "":
        LOGGER.warning("No weights and no model_cfg has been found.")
        return None

    model = ckpt_model

    return model


def initialize_params(
    model: Any, state_dict: Dict[str, Any], with_mask: bool = True
) -> None:
    """Initialize weights and masks."""
    model_dict = model.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict = OrderedDict()
    for key_ori, key_pre in zip(model_dict.keys(), state_dict.keys()):
        if with_mask or ("weight_mask" not in key_ori and "bias_mask" not in key_ori):
            pretrained_dict[key_ori] = state_dict[key_pre]
    # 3. load the new state dict
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)


def get_model_hash(model: nn.Module) -> str:
    """Get model info as hash."""
    return hashlib.sha224(str(model).encode("UTF-8")).hexdigest()


def get_pretrained_model_info(model: nn.Module) -> Dict[str, str]:
    """Read yaml file and get pretrained model.

    Read yaml file, get pretrained model information(model_dir, gdrive_link) given hash.
    """
    model_hash = str(get_model_hash(model))
    with open("config/pretrained_model_url.yaml", mode="r") as f:
        model_info = yaml.load(f, Loader=yaml.FullLoader)[model_hash]
    return model_info


def get_model_tensor_datatype(model: nn.Module) -> List[Tuple[str, torch.dtype]]:
    """Print all tensors data types."""
    return [
        (name, tensor.dtype)
        for name, tensor in model.state_dict().items()
        if hasattr(tensor, "dtype")
    ]


def get_params(
    model: nn.Module, extract_conditions: Tuple[Tuple[Any, str], ...]
) -> Tuple[Tuple[nn.Module, str], ...]:
    """Get parameters(weight and bias) tuples for pruning."""
    t = []
    for module in model.modules():
        for module_type, param_name in extract_conditions:
            # it returns true when we try hasattr(even though it returns None)
            if (
                isinstance(module, module_type)
                and getattr(module, param_name) is not None
            ):
                t += [(module, param_name)]
    return tuple(t)


def get_layernames(model: nn.Module) -> Set[str]:
    """Get parameters(weight and bias) layer name.

    Notes:
       No usage now, can be deprecated.
    """
    t = set()
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        layer_name = name.rsplit(".", 1)[0]
        t.add(layer_name)
    return t


def get_model_size_mb(model: nn.Module) -> float:
    """Get the model file size."""
    torch.save(model.state_dict(), "temp.p")
    size = os.path.getsize("temp.p") / 1e6
    os.remove("temp.p")
    return size


def remove_pruning_reparameterization(
    params_to_prune: Tuple[Tuple[nn.Module, str], ...]
) -> None:
    """Combine (weight_orig, weight_mask) and reduce the model size."""
    for module, weight_type in params_to_prune:
        prune.remove(module, weight_type)


def get_masks(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Get masks from the model."""
    mask = dict()
    for k, v in model.state_dict().items():
        if "mask" in k:
            mask[k] = v.detach().cpu().clone()
    return mask


def dummy_pruning(params_all: Tuple[Tuple[nn.Module, str], ...]) -> None:
    """Conduct fake pruning."""
    prune.global_unstructured(
        params_all,
        pruning_method=prune.L1Unstructured,
        amount=0.0,
    )


def sparsity(
    params_all: Tuple[Tuple[nn.Module, str], ...],
    module_types: Tuple[Any, ...] = (
        nn.Conv2d,
        nn.Linear,
        nn.BatchNorm1d,
        nn.BatchNorm2d,
    ),
) -> float:
    """Get the proportion of zeros in weights (default: model's sparsity)."""
    n_zero = n_total = 0

    for module, param_name in params_all:
        match = next((m for m in module_types if type(module) is m), None)
        if not match:
            continue
        n_zero += int(torch.sum(getattr(module, param_name) == 0.0).item())
        n_total += getattr(module, param_name).nelement()

    return (100.0 * n_zero / n_total) if n_total != 0 else 0.0


def mask_sparsity(
    params_all: Tuple[Tuple[nn.Module, str], ...],
    module_types: Tuple[Any, ...] = (
        nn.Conv2d,
        nn.Linear,
        nn.BatchNorm1d,
        nn.BatchNorm2d,
    ),
) -> float:
    """Get the ratio of zeros in weight masks."""
    n_zero = n_total = 0
    for module, param_name in params_all:
        match = next((m for m in module_types if type(module) is m), None)
        if not match:
            continue
        param_mask_name = param_name + "_mask"
        if hasattr(module, param_mask_name):
            param = getattr(module, param_mask_name)
            n_zero += int(torch.sum(param == 0.0).item())
            n_total += param.nelement()

    return (100.0 * n_zero / n_total) if n_total != 0 else 0.0


def download_pretrained_model(file_path: str, download_link: str) -> None:
    """Get pretrained model from google drive."""
    model_folder, model_name, file_name = file_path.rsplit(os.path.sep, 2)
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    # Download, unzip
    zip_file_path = os.path.join(model_folder, model_name + ".tar.xz")
    gdown.download(download_link, zip_file_path)
    with tarfile.open(zip_file_path, "r:*") as f:
        f.extractall(model_folder)


def dot2bracket(s: str) -> str:
    """Replace layer names with valid names for pruning.

    Test:
       >>> dot2bracket("dense2.1.bn1.bias")
       'dense2[1].bn1.bias'
       >>> dot2bracket("dense2.13.bn1.bias")
       'dense2[13].bn1.bias'
       >>> dot2bracket("conv2.123.bn1.bias")
       'conv2[123].bn1.bias'
       >>> dot2bracket("dense2.6.conv2.5.bn1.bias")
       'dense2[6].conv2[5].bn1.bias'
       >>> dot2bracket("model.6")
       'model[6]'
       >>> dot2bracket("vgg.2.conv2.bn.2")
       'vgg[2].conv2.bn[2]'
       >>> dot2bracket("features.11")
       'features[11]'
       >>> dot2bracket("dense_blocks.0.0.conv1")
       'dense_blocks[0][0].conv1'
    """
    pattern = r"\.[0-9]+"
    s_list = list(s)
    for m in re.finditer(pattern, s):
        start, end = m.span()
        # e.g s_list == [..., ".", "0", ".", "0", ".", ...]
        # step1: [..., "[", "0", "].", "0", ".", ...]
        # step2: [..., "[", "0", "][", "0", "].", ...]
        s_list[start] = s_list[start][:-1] + "["
        if end < len(s) and s_list[end] == ".":
            s_list[end] = "]."
        else:
            s_list.insert(end, "]")
    return "".join(s_list)


def wlog_weight(model: nn.Module) -> None:
    """Log weights on wandb."""
    wlog = dict()
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        layer_name, weight_type = name.rsplit(".", 1)

        # get params(weight, bias, weight_orig)
        if weight_type in ("weight", "bias", "weight_orig"):
            w_name = "params/" + layer_name + "." + weight_type
            weight = eval("model." + dot2bracket(layer_name) + "." + weight_type)
            weight = weight.cpu().data.numpy()
            wlog.update({w_name: wandb.Histogram(weight)})
        else:
            continue

        # get masked weights
        if weight_type == "weight_orig":
            w_name = "params/" + layer_name + ".weight"
            named_buffers = eval(
                "model." + dot2bracket(layer_name) + ".named_buffers()"
            )
            mask: Tuple[str, torch.Tensor] = (
                next(x for x in list(named_buffers) if x[0] == "weight_mask")[1]
                .cpu()
                .data.numpy()
            )
            masked_weight = weight[np.where(mask == 1.0)]
            wlog.update({w_name: wandb.Histogram(masked_weight)})
    wandb.log(wlog, commit=False)


def split_channels(n_channels: int, n_chunks: int) -> List[int]:
    """Get splitted channel numbers.

    It adds up all the remainders to the first chunck.
    """
    split = [n_channels // n_chunks for _ in range(n_chunks)]
    split[0] += n_channels - sum(split)
    return split


def count_model_params(model: nn.Module) -> int:
    """Count and return the total number of model params."""
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    import doctest

    doctest.testmod()
