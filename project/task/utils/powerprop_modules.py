"""
A PyTorch re-implementation of the following notebook:
https://github.com/deepmind/deepmind-research/blob/master/powerpropagation/powerpropagation.ipynb
written by DeepMind.

Adapted from the code available at: https://github.com/mysistinechapel/powerprop by @jjkc33 and @mysistinechapel
"""

from typing import Union

import torch
from torch import nn
import torch.nn.functional as F
from torch.types import _int, _size


class PowerPropLinear(nn.Module):
    """Powerpropagation Linear module."""

    def __init__(
        self,
        alpha: float,
        in_features: int,
        out_features: int,
        bias: bool = True,
    ):
        super(PowerPropLinear, self).__init__()
        self.alpha = alpha
        self.in_features = in_features
        self.out_features = out_features
        self.b = bias
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if self.b else None

    def __repr__(self):
        return (
            f"PowerPropLinear(alpha={self.alpha}, in_features={self.in_features},"
            f" out_features={self.out_features}, bias={self.b})"
        )

    def get_weight(self):
        weight = self.weight.detach()
        return torch.sign(weight) * torch.pow(torch.abs(weight), self.alpha)
        # return self.weight * torch.pow(torch.abs(self.weight), self.alpha - 1.0)

    def forward(self, inputs, mask=None):
        # Apply the re-parametrisation to `self.weight` using `self.alpha`
        # weight = self.weight * torch.pow(torch.abs(self.weight), self.alpha - 1.0)
        weight = torch.sign(self.weight) * torch.pow(torch.abs(self.weight), self.alpha)
        # Apply a mask, if given
        if mask is not None:
            weight *= mask
        # Compute the linear forward pass usign the re-parametrised weight
        return F.linear(input=inputs, weight=weight, bias=self.bias)


class PowerPropConv2D(nn.Module):
    """Powerpropagation Conv2D module."""

    def __init__(
        self,
        alpha: float,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: Union[_size, _int] = 1,
        padding: Union[_size, _int] = 1,
        dilation: Union[_size, _int] = 1,
        groups: _int = 1,
        bias: bool = False,
    ):
        super(PowerPropConv2D, self).__init__()
        self.alpha = alpha
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.b = bias
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size, kernel_size)
        )
        self.bias = nn.Parameter(torch.empty(out_channels)) if bias else None
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

    def __repr__(self):
        return (
            f"PowerPropConv2D(alpha={self.alpha}, in_channels={self.in_channels},"
            f" out_channels={self.out_channels}, kernel_size={self.kernel_size},"
            f" bias={self.b}, stride={self.stride}, padding={self.padding},"
            f" dilation={self.dilation}, groups={self.groups})"
        )

    def get_weights(self):
        weights = self.weight.detach()
        return torch.sign(weights) * torch.pow(torch.abs(weights), self.alpha)
        # return self.weight * torch.pow(torch.abs(self.weight), self.alpha - 1.0)

    def forward(self, inputs, mask=None):
        # Apply the re-parametrisation to `self.weight` using `self.alpha`
        # weight = self.weight * torch.pow(torch.abs(self.weight), self.alpha - 1.0)
        weight = torch.sign(self.weight) * torch.pow(torch.abs(self.weight), self.alpha)
        # Apply a mask, if given
        if mask is not None:
            weight *= mask
        # Compute the conv2d forward pass usign the re-parametrised weight
        return F.conv2d(
            input=inputs,
            weight=weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
