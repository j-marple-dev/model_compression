# -*- coding: utf-8 -*-
"""Plotter.

- Author: Junghoon Kim
- Email: jhkim@jmarple.ai
"""

import os
from typing import List, NamedTuple, Sequence, Tuple, Union

import PIL.Image
import matplotlib.axes
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator
import numpy as np
import torch
import torch.nn as nn
import wandb

from src.utils import get_logger

# dummy file change
logger = get_logger()


class PruneStat(NamedTuple):
    """NamedTuple to handle prune statisics."""

    pruned: Sequence[Union[int, float]]
    remained: Sequence[Union[int, float]]
    zero: Sequence[Union[int, float]]
    nonzero: Sequence[Union[int, float]]


class Plotter:
    """Plotter for models.

    Currently, it only plots sparsity information of each layer of the model,
    but it can be utilized for plotting all sort of infomration.
    """

    def __init__(self, wandb_log: bool) -> None:
        """Initialize."""
        # params to plot
        self.width = 0.4
        self.leftmargin = 0.2
        self.rightmargin = 0.2
        self.wandb_log = wandb_log
        self.total_sparsity = 0.0

    def plot_conf_mat(self, conf_mat: np.ndarray, save_dir: str, epoch: int) -> None:
        """Save a confusion matrix as an image."""
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(1, 1, 1)
        ax.matshow(conf_mat)
        # Gridlines based on minor ticks
        ax.xaxis.set_major_locator(FixedLocator(np.linspace(0, 41, 1)))
        ax.yaxis.set_major_locator(FixedLocator(np.linspace(0, 41, 1)))

        ax.grid(which="minor", color="w", linestyle="-", linewidth=2)
        fig.savefig(save_dir + os.path.sep + str(epoch))
        if self.wandb_log:
            pil_image = PIL.Image.frombytes(
                "RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb()
            )
            wandb.log(
                {
                    "Pruned/"
                    + "confusion_matrix": [
                        wandb.Image(pil_image, caption="Confusion matrix")
                    ]
                },
                commit=False,
            )
        plt.close(fig)

    def plot(self, model: nn.Module, path: str) -> None:
        """Plot sparsity information and save into given path(and wandb if enabled)."""
        layer_names, params, ratio = self._get_prune_statistics(model)
        self._plot_pruned_stats(
            layer_names, params, os.path.join(path, "parameters.png")
        )
        self._plot_pruned_stats(layer_names, ratio, os.path.join(path, "ratio.png"))

    def _get_prune_statistics(
        self, model: nn.Module
    ) -> Tuple[List[str], PruneStat, PruneStat]:
        """Get prune statisics for each layer."""
        layer_names = []

        pruned_params, remained_params = [], []
        zero_params, nonzero_params = [], []
        pruned_ratio, remained_ratio = [], []
        zero_ratio, nonzero_ratio = [], []

        for name, module in model.named_modules():
            if type(module) not in (nn.Conv2d, nn.Linear):
                continue
            if not (hasattr(module, "weight_mask") and hasattr(module, "weight")):
                continue
            layer_names.append(name)
            total = getattr(module, "weight_mask").nelement()
            pruned = int(torch.sum(getattr(module, "weight_mask") == 0.0).item())
            zero = int(torch.sum(getattr(module, "weight_mask") == 0.0).item())

            pruned_params.append(pruned)
            remained_params.append(total - pruned)

            pruned_ratio.append(pruned / total)
            remained_ratio.append(1 - pruned / total)

            zero_params.append(zero)
            nonzero_params.append(total - zero)

            zero_ratio.append(zero / total)
            nonzero_ratio.append(1 - zero / total)

        params = PruneStat(pruned_params, remained_params, zero_params, nonzero_params)
        ratio = PruneStat(pruned_ratio, remained_ratio, zero_ratio, nonzero_ratio)

        self.total_sparsity = sum(pruned_params) / (
            sum(pruned_params) + sum(remained_params)
        )

        return layer_names, params, ratio

    def _plot_pruned_stats(
        self, x_names: List[str], stats: PruneStat, save_path: str
    ) -> None:
        """Plot pruned parameters for each layers."""
        # extract type save_path: 'path+type.png'
        stat_type = save_path.rsplit(".", 1)[0].rsplit("/", 1)[1]

        fig, ax = self._get_fig(x_names)
        x = np.arange(len(x_names))

        kargs_base = dict(width=self.width, edgecolor="black")
        kargs_first_bar = {**kargs_base, "x": x - 1 / 2 * self.width}
        kargs_second_bar = {**kargs_base, "x": x + 1 / 2 * self.width}

        # draw first bar(pruned, remained)
        kargs_pruned = dict(
            **kargs_first_bar,
            height=stats.pruned,
            bottom=stats.remained,
            color="w",
            label="Pruned",
        )
        kargs_remained = dict(
            **kargs_first_bar, height=stats.remained, label="Remained"
        )

        # return needed only when we annotate info on bars
        bar_pruned = ax.bar(**kargs_pruned)
        bar_remained = ax.bar(**kargs_remained)

        # draw second bar(zero, nonzero)
        kargs_zero = dict(
            **kargs_second_bar,
            height=stats.zero,
            bottom=stats.nonzero,
            color="w",
            label="Zero",
        )
        kargs_nonzero = dict(**kargs_second_bar, height=stats.nonzero, label="Nonzero")
        ax.bar(**kargs_zero)
        ax.bar(**kargs_nonzero)

        # annotate on top of bars
        self._annotate_on_bar(ax, bars=bar_remained)
        self._annotate_on_stacked_bars(
            ax, bars=bar_pruned, bottom_bars=bar_remained, addup_bottom_bar_data=True
        )

        # draw info on figure
        ax.set_ylabel(stat_type.capitalize())
        ax.set_title(
            f"Model layerwise statistics, total sparsity: {100 * self.total_sparsity:.2f}%"
        )
        ax.set_xticks(x)
        ax.set_xticklabels(x_names, rotation=60, horizontalalignment="right")
        ax.legend()
        ax.autoscale(enable=True)
        _, ymax = ax.get_ylim()
        ax.set_ylim(top=ymax * 1.1)
        # plot pruned on top of remainder(shows remainder only)
        fig.savefig(save_path)
        if self.wandb_log:
            pil_image = PIL.Image.frombytes(
                "RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb()
            )
            wandb.log(
                {
                    "Pruned/"
                    + stat_type: [
                        wandb.Image(pil_image, caption=stat_type.capitalize())
                    ]
                },
                commit=False,
            )

    def _annotate_on_bar(
        self,
        ax: matplotlib.axes.Axes,
        bars: List[matplotlib.axes.Axes.bar],
    ) -> None:
        """Attach a text label above each bar in rects, displaying its height."""
        for _, bar in enumerate(bars):
            height = bar.get_height()
            display_value = height
            self._ax_annotate(ax, bar, display_value, height)

    def _annotate_on_stacked_bars(
        self,
        ax: matplotlib.axes.Axes,
        bars: List[matplotlib.axes.Axes.bar],
        bottom_bars: List[matplotlib.axes.Axes.bar],
        addup_bottom_bar_data: bool = False,
    ) -> None:
        """Same as annotate_on_bar but on top of stacked bars."""
        for i, bar in enumerate(bars):
            bottom_height = bottom_bars[i].get_height()
            height = bar.get_height()
            # value display can be either summed data bar below or only top bar.
            display_value = height + bottom_height if addup_bottom_bar_data else height
            self._ax_annotate(ax, bar, display_value, height + bottom_height)

    def _ax_annotate(
        self,
        ax: matplotlib.axes.Axes,
        bar: matplotlib.axes.Axes.bar,
        display_value: Union[int, float],
        height: float,
    ) -> None:
        """Warpper for ax.annotate."""
        ax.annotate(
            f"{display_value*100:.1f}%"
            if isinstance(display_value, float)
            else f"{display_value}",
            xytext=(0, 50)
            if isinstance(display_value, float)
            else (0, 10),  # 3 points vertical offset
            rotation=90 if isinstance(display_value, float) else 0,
            fontsize="large",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            textcoords="offset points",
            ha="center",
            va="top",
        )

    def _get_fig(
        self, labels: List[str]
    ) -> Tuple[matplotlib.pyplot.figure, matplotlib.axes.Axes]:
        """Get figure, axes."""
        figwidth = self.leftmargin + self.rightmargin + (len(labels) + 1) * self.width
        if figwidth < 8:
            figwidth = 8

        fig, ax = plt.subplots(figsize=(figwidth, 14))
        return fig, ax
