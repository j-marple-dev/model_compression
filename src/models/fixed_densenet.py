# -*- coding: utf-8 -*-
"""Fixed DenseNet Model.

All blocks consist of ConvBNReLU for quantization.

- Author: Curt-Park
- Email: jwpark@jmarple.ai
"""

import math
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn

from src.models.common_layers import ConvBNReLU


class Bottleneck(nn.Module):
    """Bottleneck block for DenseNet."""

    def __init__(
        self, inplanes: int, expansion: int = 4, growthRate: int = 12,
    ) -> None:
        """Initialize."""
        super(Bottleneck, self).__init__()
        planes = expansion * growthRate
        self.conv1 = ConvBNReLU(inplanes, planes, kernel_size=1)
        self.conv2 = ConvBNReLU(planes, growthRate, kernel_size=3)

    def _cat(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Concat channels."""
        return torch.cat((x, y), 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward."""
        out = self.conv1(x)
        out = self.conv2(out)
        return self._cat(x, out)


class BasicBlock(nn.Module):
    """BasicBlock for DenseNet."""

    def __init__(
        self, inplanes: int, expansion: int = 1, growthRate: int = 12,
    ) -> None:
        """Initialize."""
        super(BasicBlock, self).__init__()
        self.conv = ConvBNReLU(inplanes, growthRate, kernel_size=3)

    def _cat(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Concat channels."""
        return torch.cat((x, y), 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward."""
        out = self.conv(x)
        return self._cat(x, out)


class Transition(nn.Module):
    """Transition between blocks."""

    def __init__(self, inplanes: int, outplanes: int) -> None:
        """Initialize."""
        super(Transition, self).__init__()
        self.conv = ConvBNReLU(inplanes, outplanes, kernel_size=1)
        self.avg_pool = nn.AvgPool2d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward."""
        out = self.conv(x)
        out = self.avg_pool(out)
        return out


class DenseNet(nn.Module):
    """DenseNet architecture."""

    def __init__(
        self,
        num_classes: int,
        depth: int = 22,
        growthRate: int = 12,
        compressionRate: int = 2,
        block: "type" = Bottleneck,
    ) -> None:
        """Initialize."""
        assert (depth - 4) % 3 == 0, "depth should be 3n+4"

        super(DenseNet, self).__init__()
        n = (depth - 4) // 3 if block is BasicBlock else (depth - 4) // 6

        self.growthRate = growthRate

        # self.inplanes is a global variable used across multiple
        self.inplanes = growthRate * 2
        self.stem = ConvBNReLU(3, self.inplanes, kernel_size=3)

        self.dense1 = self._make_denseblock(block, n)
        self.trans1 = self._make_transition(compressionRate)
        self.dense2 = self._make_denseblock(block, n)
        self.trans2 = self._make_transition(compressionRate)
        self.dense3 = self._make_denseblock(block, n)

        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(self.inplanes, num_classes)

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.last_conv_shape = 1

        # name : ([in], [out])
        # https://github.com/python/mypy/issues/5751
        # Dict
        self.conv_connection: Dict[str, Tuple[List[str], List[str]]] = dict(
            {"stem.conv": ([], ["stem.bn"])}
        )

        # hard coded connection pair
        # [(base, in_channel)]
        name_setup = [
            ("dense1.", "stem.bn"),
            ("dense2.", "trans1.conv.bn"),
            ("dense3.", "trans2.conv.bn"),
        ]
        """
        in_layer1: consist of concatenated layer from earlier layer.
        ex):
        dense2.8.conv1 (['trans1.bn', 'dense2.0.bn2', 'dense2.1.bn2',
                         'dense2.2.bn2', 'dense2.3.bn2', 'dense2.4.bn2',
                         'dense2.5.bn2', 'dense2.6.bn2', 'dense2.7.bn2'],
                         ['dense2.8.bn1'])
        dense2.8.conv2 (['dense2.8.bn1'], ['dense2.8.bn2'])
        """

        # dense layer
        for (base, in_name) in name_setup:
            for f_i in range(16):
                layer1_name = base + str(f_i) + ".conv1.conv"
                layer2_name = base + str(f_i) + ".conv2.conv"

                # layer_1
                in_layers1 = [in_name] + [
                    base + str(idx) + ".conv2.bn" for idx in range(f_i)
                ]
                out_layers1 = [base + str(f_i) + ".conv1.bn"]
                in_layers2 = out_layers1
                out_layers2 = [base + str(f_i) + ".conv2.bn"]
                self.conv_connection.update(
                    {
                        layer1_name: (in_layers1, out_layers1),
                        layer2_name: (in_layers2, out_layers2),
                    }
                )

        # trans layer
        for idx, in_layer in enumerate(["stem.bn", "trans1.conv.bn"], start=1):
            layer_name = "trans" + str(idx) + ".conv.conv"
            in_layers = [in_layer] + [
                "dense" + str(idx) + "." + str(f_i) + "conv2.bn" for f_i in range(16)
            ]
            out_layers = ["trans" + str(idx) + ".conv.bn"]
            self.conv_connection.update({layer_name: (in_layers, out_layers)})

        # name : [in]
        self.fc_connection: Dict[str, List[str]] = {
            "fc": ["trans2.conv.bn"]
            + [base + str(f_i) + ".conv2.bn" for f_i in range(16)]
        }

        # for debug
        """
        print('CHECK')
        for k, v in self.conv_connection.items():
            print(k, v)
        for k, v in self.fc_connection.items():
            print(k, v)
        """

    def _make_denseblock(self, block: "type", blocks: int) -> nn.Module:
        """Make a dense block."""
        layers = []
        for _ in range(blocks):
            # Currently we fix the expansion ratio as the default value
            layers.append(block(self.inplanes, growthRate=self.growthRate))
            self.inplanes += self.growthRate

        return nn.Sequential(*layers)

    def _make_transition(self, compressionRate: int) -> nn.Module:
        """Make a transition."""
        inplanes = self.inplanes
        outplanes = int(math.floor(self.inplanes // compressionRate))
        self.inplanes = outplanes
        return Transition(inplanes, outplanes)

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        """Actual forward procedures."""
        x = self.stem(x)

        x = self.trans1(self.dense1(x))
        x = self.trans2(self.dense2(x))
        x = self.dense3(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward."""
        return self._forward_impl(x)


def get_model(**kwargs: Any) -> nn.Module:
    """Constructs a ResNet model. """
    return DenseNet(**kwargs)
