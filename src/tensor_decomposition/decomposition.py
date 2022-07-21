"""Tensor Decomposition for YOLO model.

- Author: Jongkuk Lim
- Contact: limjk@jmarple.ai
"""

from copy import deepcopy
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.optimize import minimize_scalar
import tensorly as tl
from tensorly.decomposition import partial_tucker
import torch
from torch import nn
import torch.nn.utils.prune as prune

from src.logger import get_logger

LOGGER = get_logger(__name__)

tl.set_backend("pytorch")


def tau(x: np.ndarray, alpha: float) -> np.ndarray:
    """Compute tau value for EVBsigma2.

    Args:
        x: value to compute tau.
        alpha: alpha blending parameter.

    Return:
        tau value from x
    """
    return 0.5 * (x - (1 + alpha) + np.sqrt((x - (1 + alpha)) ** 2 - 4 * alpha))


def EVBsigma2(
    sigma2: float, L: int, M: int, s: np.ndarray, residual: float, xubar: float
) -> float:
    """Compute sigma value for EVBMF.

    Args:
        sigma2: sigma value
        L: matrix shape L
        M: matrix shape M
        s: matrix of singular values
        residual: residual value.
        xubar: bar{x_u}

    Return:
        sigma value for EVBMF
    """
    H = len(s)

    alpha = L / M
    x = s**2 / (M * sigma2)

    z1 = x[x > xubar]
    z2 = x[x <= xubar]
    tau_z1 = tau(z1, alpha)

    term1 = np.sum(z2 - np.log(z2))
    term2 = np.sum(z1 - tau_z1)
    term3 = np.sum(np.log(np.divide(tau_z1 + 1, z1)))
    term4 = alpha * np.sum(np.log(tau_z1 / alpha + 1))

    obj = (
        term1
        + term2
        + term3
        + term4
        + residual / (M * sigma2)
        + (L - H) * np.log(sigma2)
    )

    return obj


def EVBMF(
    Y: torch.Tensor, sigma2: Optional[int] = None, H: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """Compute EVBMF.

    Implementation of the analytical solution to EVBMF.

    (Empirical Variational Bayes Matrix Factorization)

    This function can be used to calculate the analytical solution to empirical VBMF.
    This is based on the paper and MatLab code by Nakajima et al.:
    "Global analytic solution of fully-observed variational Bayesian matrix factorization."

    Notes
    -----
        If sigma2 is unspecified, it is estimated by minimizing the free energy.
        If H is unspecified, it is set to the smallest of the sides of the input Y.

    Args:
        Y : numpy-array
            Input matrix that is to be factorized. Y has shape (L,M), where L<=M.

        sigma2 : int or None (default=None)
            Variance of the noise on Y.

        H : int or None (default = None)
            Maximum rank of the factorized matrices.

    Returns:
        U : numpy-array
            Left-singular vectors.

        S : numpy-array
            Diagonal matrix of singular values.

        V : numpy-array
            Right-singular vectors.

        post : dictionary
            Dictionary containing the computed posterior values.


    References
    ----------
    .. [1] Nakajima, Shinichi, et al. "Global analytic solution of fully-observed variational Bayesian matrix factorization." Journal of Machine Learning Research 14.Jan (2013): 1-37.

    .. [2] Nakajima, Shinichi, et al. "Perfect dimensionality recovery by variational Bayesian PCA." Advances in Neural Information Processing Systems. 2012.
    """
    L, M = Y.shape  # has to be L<=M

    if H is None:
        H = L

    alpha = L / M
    tauubar = 2.5129 * np.sqrt(alpha)

    # SVD of the input matrix, max rank of H
    U, s, V = np.linalg.svd(Y)
    U = U[:, :H]
    s = s[:H]
    V = V[:H].T

    # Calculate residual
    residual = 0.0
    if H < L:
        residual = np.sum(np.sum(Y**2) - np.sum(s**2))

    # Estimation of the variance when sigma2 is unspecified
    if sigma2 is None:
        xubar = (1 + tauubar) * (1 + alpha / tauubar)
        eH_ub = int(np.min([np.ceil(L / (1 + alpha)) - 1, H]))
        upper_bound = (np.sum(s**2) + residual) / (L * M)
        lower_bound = np.max([s[eH_ub] ** 2 / (M * xubar), np.mean(s[eH_ub:] ** 2) / M])

        scale = 1.0  # /lower_bound
        s = s * np.sqrt(scale)
        residual = residual * scale
        lower_bound = lower_bound * scale
        upper_bound = upper_bound * scale

        sigma2_opt = minimize_scalar(
            EVBsigma2,
            args=(L, M, s, residual, xubar),
            bounds=[lower_bound, upper_bound],
            method="Bounded",
        )
        sigma2 = sigma2_opt.x

    # Threshold gamma term
    threshold = np.sqrt(M * sigma2 * (1 + tauubar) * (1 + alpha / tauubar))
    pos = np.sum(s > threshold)

    # Formula (15) from [2]
    d = np.multiply(
        s[:pos] / 2,
        1
        - np.divide((L + M) * sigma2, s[:pos] ** 2)
        + np.sqrt(
            (1 - np.divide((L + M) * sigma2, s[:pos] ** 2)) ** 2
            - 4 * L * M * sigma2**2 / s[:pos] ** 4
        ),
    )

    # Computation of the posterior
    post = {}
    post["ma"] = np.zeros(H)
    post["mb"] = np.zeros(H)
    post["sa2"] = np.zeros(H)
    post["sb2"] = np.zeros(H)
    post["cacb"] = np.zeros(H)

    tau = np.multiply(d, s[:pos]) / (M * sigma2)
    delta = np.multiply(np.sqrt(np.divide(M * d, L * s[:pos])), 1 + alpha / tau)

    post["ma"][:pos] = np.sqrt(np.multiply(d, delta))
    post["mb"][:pos] = np.sqrt(np.divide(d, delta))
    post["sa2"][:pos] = np.divide(sigma2 * delta, s[:pos])
    post["sb2"][:pos] = np.divide(sigma2, np.multiply(delta, s[:pos]))
    post["cacb"][:pos] = np.sqrt(np.multiply(d, s[:pos]) / (L * M))
    post["sigma2"] = sigma2  # type: ignore
    post["F"] = 0.5 * (
        L * M * np.log(2 * np.pi * sigma2)
        + (residual + np.sum(s**2)) / sigma2
        + np.sum(M * np.log(tau + 1) + L * np.log(tau / alpha + 1) - M * tau)
    )

    return U[:, :pos], np.diag(d), V[:, :pos], post


def decompose_layer_evaluation(
    layer: nn.Conv2d, test_input: torch.Tensor, origin_out: torch.Tensor
) -> Tuple[Optional[nn.Sequential], Union[torch.Tensor, float]]:
    """Decompose layer and evaluate loss.

    Args:
        layer: layer to apply tensor decomposition.
        test_input: test input tensor to feedforward layer.
        origin_out: original output tensor: layer(test_input).

    Return:
        (decomposed_layer, loss)
        (None, inf) if failed to decompose the layer.
    """
    original_layer = deepcopy(layer)
    decomposed_layer = None

    try:
        decomposed_layer = tucker_decomposition_conv_layer(original_layer)
    except ValueError:
        LOGGER.info("Decompose tensor failed.")
        return None, float("inf")

    decomposed_out = decomposed_layer(test_input)
    loss = torch.abs(origin_out - decomposed_out).sum() / origin_out.numel()

    return decomposed_layer, loss


def decompose_model(
    model: nn.Module, loss_thr: float = 0.1, prune_step: float = 0.01
) -> None:
    """Decompose conv in model recursively.

    Decompose all (n, n) conv to (1, 1) -> (n, n) -> (1, 1)
        --> n > 1
    Note that this is in-place operation.

    Args:
        model: PyTorch model.
        loss_thr: loss threshold to compare between original conv with decomposed conv.
            loss = (o1 - o2).abs().sum() / o1.numel()
            o1: original conv out
            o2: decomposed conv out
        prune_step: pruning ratio step size.
            i.e. prune_step=0.1 will try to prune (0.1, 0.2, 0.3, ...) before decomposition until loss is larger than loss_thr.
            if prune_step is equal or smaller than 0.0, prunning will not be applied.
    """
    for i, (name, module) in enumerate(model.named_children()):
        if len(list(module.children())) > 0:
            decompose_model(
                module, loss_thr=loss_thr, prune_step=prune_step
            )  # Call recursively

        if isinstance(module, nn.Conv2d):
            if isinstance(model, nn.ModuleList):
                conv = model[i]
            else:
                # conv = module
                conv = model.conv
                # conv = deepcopy(module)
                # module_param = [x for x in module.parameters()]
                # conv_param = [x for x in conv.parameters()]

            if conv != module:
                # if conv != module and not torch.equal(module_param[0], conv_param[0]):
                continue

            if conv.kernel_size == (1, 1):
                continue

            test_input = torch.rand((1024, *conv.weight.shape[1:]))
            origin_out = conv(test_input)
            decomposed_conv = None

            # Run decomposition before searching prunning ratio to check if it's worthy.
            decomposed_conv_candidate, loss = decompose_layer_evaluation(
                conv, test_input, origin_out
            )
            LOGGER.info(f"{name} (Prune: {0.0:.3f}): Loss(mean): {loss}, ")
            if loss < loss_thr:
                run_b_search = True
                decomposed_conv = decomposed_conv_candidate
            else:
                run_b_search = False

            if prune_step <= 0:
                run_b_search = False

            max_prune_ratio = 1.0
            min_prune_ratio = 0.0
            prune_ratio = (max_prune_ratio + min_prune_ratio) / 2

            while run_b_search:  # Binary search for pruning ratio.
                original_conv = deepcopy(conv)
                if prune_ratio > 0.0:
                    prune.l1_unstructured(
                        original_conv, name="weight", amount=prune_ratio
                    )
                    prune.remove(original_conv, "weight")

                decomposed_conv_candidate, loss = decompose_layer_evaluation(
                    original_conv, test_input, origin_out
                )

                LOGGER.info(f"{name} (Prune: {prune_ratio:.3f}): Loss(mean): {loss}, ")

                if loss < loss_thr:
                    min_prune_ratio = prune_ratio
                    decomposed_conv = decomposed_conv_candidate
                else:
                    max_prune_ratio = prune_ratio

                next_prune_ratio = (max_prune_ratio + min_prune_ratio) / 2
                if (
                    abs(prune_ratio - next_prune_ratio) == 0
                    or abs(prune_ratio - next_prune_ratio) < prune_step
                ):
                    break

                prune_ratio = next_prune_ratio

            if decomposed_conv is not None:
                for attr_name in ["in_channels", "out_channels", "kernel_size"]:
                    setattr(decomposed_conv, attr_name, getattr(conv, attr_name))
                # decomposed_conv.in_channels = conv.in_channels
                # decomposed_conv.out_channels = conv.out_channels
                # decomposed_conv.kernel_size = conv.kernel_size

                if isinstance(model, nn.ModuleList):
                    model[i] = decomposed_conv
                else:
                    model.conv = decomposed_conv

                LOGGER.info("    |---------- Switching conv to decomposed conv")
            else:
                LOGGER.info("    |---------- Skip switching to decomposed conv.")


def estimate_ranks(layer: nn.Conv2d) -> List[int]:
    """Estimate ranks for the given layer.

    Unfold the 2 modes of the Tensor the decomposition will
    be performed on, and estimates the ranks of the matrices using VBMF

    Args:
        layer: Conv2d module.

    Return:
        estimated ranks
    """
    weights = layer.weight.data
    unfold_0 = tl.base.unfold(weights, 0)
    unfold_1 = tl.base.unfold(weights, 1)

    _, diag_0, _, _ = EVBMF(unfold_0)
    _, diag_1, _, _ = EVBMF(unfold_1)
    ranks = [diag_0.shape[0], diag_1.shape[1]]
    return ranks


def tucker_decomposition_conv_layer(layer: nn.Conv2d) -> nn.Sequential:
    """Perform Tucker decomposition on the Conv2d Layer.

    The ranks are estimated with a Python implementation of VBMF
    https://github.com/CasvandenBogaard/VBMF

    Args:
        layer: Conv2d module.

    Return:
        nn.Sequential object with the Tucker decomposition.
        Which consists of (1, 1) conv -> (n, n) conv -> (1, 1) conv
    """
    ranks = estimate_ranks(layer)
    LOGGER.info(f"{layer} : VBMF Estimated ranks:  {ranks}")
    core, [last, first] = partial_tucker(
        layer.weight.data, modes=[0, 1], rank=ranks, init="svd"
    )

    # A pointwise convolution that reduces the channels from S to R3
    first_layer = torch.nn.Conv2d(
        in_channels=first.shape[0],
        out_channels=first.shape[1],
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=layer.dilation,  # type: ignore
        bias=False,
    )

    # A regular 2D convolution layer with R3 input channels
    # and R3 output channels
    core_layer = torch.nn.Conv2d(
        in_channels=core.shape[1],
        out_channels=core.shape[0],
        kernel_size=layer.kernel_size,  # type: ignore
        stride=layer.stride,  # type: ignore
        padding=layer.padding,  # type: ignore
        dilation=layer.dilation,  # type: ignore
        bias=False,
    )

    # A pointwise convolution that increases the channels from R4 to T
    last_layer = torch.nn.Conv2d(
        in_channels=last.shape[1],
        out_channels=last.shape[0],
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=layer.dilation,  # type: ignore
        bias=layer.bias is not None,
    )

    if layer.bias is not None:
        last_layer.bias.data = layer.bias.data  # type: ignore

    first_layer.weight.data = torch.transpose(first, 1, 0).unsqueeze(-1).unsqueeze(-1)
    last_layer.weight.data = last.unsqueeze(-1).unsqueeze(-1)
    core_layer.weight.data = core

    new_layers = [first_layer, core_layer, last_layer]
    return nn.Sequential(*new_layers)
