# -*- coding: utf-8 -*-
"""Utils for handling models.

- Author: Curt-Park
- Email: jwpark@jmarple.ai
"""

import hashlib
import os
import re
import tarfile
from typing import Any, Dict, List, Set, Tuple

import gdown
import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import wandb
import yaml


def get_model(model_name: str, model_config: Dict[str, Any]) -> nn.Module:
    """Get PyTorch model."""
    # get model constructor
    return __import__("src.models." + model_name, fromlist=[model_name]).get_model(
        **model_config
    )


def initialize_params(model: Any, state_dict: Dict[str, Any], with_mask=True) -> None:
    """Initialize weights and masks."""
    model_dict = model.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict = {}
    for k, v in state_dict.items():
        if k in model_dict and (with_mask or "weight_mask" not in k):
            pretrained_dict[k] = v
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)


def get_model_hash(model: nn.Module) -> str:
    """Get model info as hash."""
    return hashlib.sha224(str(model).encode("UTF-8")).hexdigest()


def get_pretrained_model_info(model: nn.Module) -> Dict[str, str]:
    """Read yaml file, get pretrained model information(model_dir, gdrive_link) \
        given hash."""
    model_hash = str(get_model_hash(model))
    with open("config/pretrained_model_url.yaml", mode="r") as f:
        model_info = yaml.load(f, Loader=yaml.FullLoader)[model_hash]
    return model_info


def get_param_names(model: nn.Module) -> Set[str]:
    """Get param names in the model."""
    layer_names = set()
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        layer_name, weight_type = name.rsplit(".", 1)
        layer_names.add(layer_name)
    return layer_names


def get_model_tensor_datatype(model: nn.Module) -> List[Tuple[str, torch.dtype]]:
    """Print all tensors data types."""
    return [
        (name, tensor.dtype)
        for name, tensor in model.state_dict().items()
        if hasattr(tensor, "dtype")
    ]


def get_model_size_mb(model: nn.Module) -> float:
    """Get the model file size."""
    torch.save(model.state_dict(), "temp.p")
    size = os.path.getsize("temp.p") / 1e6
    os.remove("temp.p")
    return size


def dummy_pruning(model: nn.Module) -> Tuple[Tuple[nn.Module, str], ...]:
    """Conduct fake pruning."""
    params_to_prune = get_weight_tuple(model, bias=False)
    prune.global_unstructured(
        params_to_prune, pruning_method=prune.L1Unstructured, amount=0.0,
    )
    return params_to_prune


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


def get_weight_tuple(
    model: nn.Module, bias: bool = False
) -> Tuple[Tuple[nn.Module, str], ...]:
    """Get weight and bias tuples for pruning."""
    t = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        layer_name, weight_type = name.rsplit(".", 1)
        if weight_type == "weight" or (bias and weight_type == "bias"):
            t.append((eval("model." + dot2bracket(layer_name)), weight_type,))
    return tuple(t)


def sparsity(
    params_all: Tuple[Tuple[nn.Module, str], ...], module_name: str = ""
) -> float:
    """Get the proportion of zeros in weights (default: model's sparsity)."""
    n_zero = n_total = 0

    for module, wtype in params_all:
        if module_name not in str(module):
            continue
        n_zero += int(torch.sum(getattr(module, wtype) == 0.0).item())
        n_total += getattr(module, wtype).nelement()

    return (100.0 * n_zero / n_total) if n_total != 0 else 0.0


def mask_sparsity(model: nn.Module) -> float:
    """Get the ratio of zeros in weight masks."""
    n_zero = n_total = 0
    param_names = get_param_names(model)

    for w in param_names:
        param_instance = eval("model." + dot2bracket(w) + ".weight_mask")
        n_zero += int(torch.sum(param_instance == 0.0).item())
        n_total += param_instance.nelement()

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
    """
    pattern = r"\.[0-9]+"
    s_list = list(s)
    for m in re.finditer(pattern, s):
        start, end = m.span()
        s_list[start] = "["
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
            mask: Tuple[str, torch.Tensor] = next(
                x for x in list(named_buffers) if x[0] == "weight_mask"
            )[1].cpu().data.numpy()
            masked_weight = weight[np.where(mask == 1.0)]
            wlog.update({w_name: wandb.Histogram(masked_weight)})
    wandb.log(wlog, commit=False)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
