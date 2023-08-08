"""
A PyTorch re-implementation of the following notebook:
https://github.com/deepmind/deepmind-research/blob/master/powerpropagation/powerpropagation.ipynb
written by DeepMind.

Adapted from the code available at: https://github.com/mysistinechapel/powerprop by @jjkc33 and @mysistinechapel
"""

from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.types import _int, _size


def init_weights(module: nn.Module):
    """Initialise PowerPropLinear and PowerPropConv2D layers in the input module."""
    if isinstance(module, (PowerPropLinear, PowerPropConv2D)):
        fan_in = calculate_fan_in(module.w.data)

        # constant from scipy.stats.truncnorm.std(a=-2, b=2, loc=0., scale=1.)
        distribution_stddev = 0.87962566103423978

        std = np.sqrt(1.0 / fan_in) / distribution_stddev
        a, b = -2.0 * std, 2.0 * std

        u = nn.init.trunc_normal_(module.w.data, std=std, a=a, b=b)
        u = torch.sign(u) * torch.pow(torch.abs(u), 1.0 / module.alpha)

        module.w.data = u
        if module.b is not None:
            module.b.data.zero_()


def calculate_fan_in(tensor: torch.Tensor) -> float:
    """Modified from: https://github.com/pytorch/pytorch/blob/master/torch/nn/init.py"""
    dimensions = tensor.dim()
    if dimensions < 2:
        raise ValueError(
            "Fan in can not be computed for tensor with fewer than 2 dimensions"
        )

    num_input_fmaps = tensor.size(1)
    receptive_field_size = 1
    if dimensions > 2:
        for s in tensor.shape[2:]:
            receptive_field_size *= s
    fan_in = num_input_fmaps * receptive_field_size

    return float(fan_in)


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
        self.bias = bias
        self.w = torch.nn.Parameter(torch.empty(out_features, in_features))
        self.b = torch.nn.Parameter(torch.empty(out_features)) if self.bias else None

    def __repr__(self):
        return f"PowerPropLinear(alpha={self.alpha}, in_features={self.in_features}, out_features={self.out_features}, bias={self.bias})"

    def get_weights(self):
        weights = self.w.detach()
        return torch.sign(weights) * torch.pow(torch.abs(weights), self.alpha)

    def forward(self, inputs, mask=None):
        # Apply the re-parametrisation to `self.w` using `self.alpha`
        weights = self.w * torch.pow(torch.abs(self.w), self.alpha - 1.0)
        # Apply a mask, if given
        if mask is not None:
            weights *= mask
        # Compute the linear forward pass usign the re-parametrised weights
        return F.linear(inputs=inputs, weights=weights, bias=self.b)


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
        self.bias = bias
        self.w = torch.nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size, kernel_size)
        )
        self.b = torch.nn.Parameter(torch.empty(out_channels)) if bias else None
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

    def __repr__(self):
        return f"PowerPropConv2D(alpha={self.alpha}, in_channels={self.in_channels}, out_channels={self.out_channels}, kernel_size={self.kernel_size}, bias={self.bias}, stride={self.stride}, padding={self.padding}, dilation={self.dilation}, groups={self.groups})"

    def get_weights(self):
        weights = self.w.detach()
        return torch.sign(weights) * torch.pow(torch.abs(weights), self.alpha)

    def forward(self, inputs, mask=None):
        # Apply the re-parametrisation to `self.w` using `self.alpha`
        weights = self.w * torch.pow(torch.abs(self.w), self.alpha - 1.0)
        # Apply a mask, if given
        if mask is not None:
            weights *= mask
        # Compute the conv2d forward pass usign the re-parametrised weights
        return F.conv2d(
            input=inputs,
            weight=weights,
            bias=self.b,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
