# -*- coding: utf-8 -*-
"""Get adjacent modules from a given model.

- Author: Curt-Park
- Email: jwpark@jmarple.ai
"""

from itertools import chain
from typing import Any, Dict, List, Tuple, Union

import torch
import torch.nn as nn


class AdjModuleGetter:
    """Adjacent module getter used by Shrinker.

    This gets adjacent module information to use for model shrinking.
    Assume the model consists of conv-bn-relu sequence.
    """

    def __init__(
        self, model: nn.Module, input_size: Tuple[int, ...], device: torch.device
    ) -> None:
        """Initialize."""
        self.model = model
        # op: module_ahead[op] = module
        self.module_ahead: Dict[Any, nn.Module] = dict()
        # op_behind[module] = op
        self.op_behind: Dict[nn.Module, Any] = dict()
        self.last_conv_shape = 0

        # register hooks
        hooks = []
        for module in self.model.modules():
            hooks.append(module.register_forward_hook(self._hook_fn))

        # execute hooks and create backward operators graph
        rand_in = torch.randn(input_size, device=device)
        out = self.model(rand_in)
        self.op_backward_graph = self._create_backward_op_graph(out)

        # remove hooks
        for hook in hooks:
            hook.remove()

    def find_modules_ahead_of(
        self, module: nn.Module, target_type: "type"
    ) -> List[Any]:
        """Find all modules ahead of the input backward opterator."""

        def find_modules_ahead_of(op: Any) -> List[Any]:
            if op in self.module_ahead and type(self.module_ahead[op]) is target_type:
                return [self.module_ahead[op]]
            modules: List[Any] = []
            if op in self.op_backward_graph and self.op_backward_graph[op]:
                module_iter = chain.from_iterable(
                    find_modules_ahead_of(prev_op)
                    for prev_op in self.op_backward_graph[op]
                )
                modules = list(module_iter)
            return modules

        return find_modules_ahead_of(self.op_behind[module])

    def find_module_next_to(
        self, module: nn.Module, target_type: "type"
    ) -> Union[nn.Module, None]:
        """Find a single module right next to the input module."""
        layers = [
            v for v in self.model.modules() if type(v) in {type(module), target_type}
        ]
        for i in range(1, len(layers)):
            if layers[i - 1] is module and type(layers[i]) is target_type:
                return layers[i]
        return None

    def _create_backward_op_graph(self, out: torch.Tensor) -> Dict[Any, List[Any]]:
        """Create a graph that contains backward operators' information."""
        graph: Dict[Any, List[Any]] = dict()

        def backward_search(var: Any) -> None:
            if var in graph:
                return
            graph[var] = []
            if hasattr(var, "next_functions"):
                for next_var in var.next_functions:
                    if next_var[0] is None:
                        continue
                    graph[var].append(next_var[0])
                    backward_search(next_var[0])

        backward_search(out.grad_fn)  # type: ignore
        return graph

    def _hook_fn(self, module: nn.Module, inp: torch.Tensor, out: torch.Tensor) -> None:
        self.module_ahead[out.grad_fn] = module  # type: ignore
        self.op_behind[module] = out.grad_fn  # type: ignore

        if type(module) == nn.Flatten:  # type: ignore
            self.last_conv_shape = inp[0].size()[-1]
