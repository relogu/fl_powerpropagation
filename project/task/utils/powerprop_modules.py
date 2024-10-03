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


def spectral_norm(
    self_weight: torch.Tensor, num_iterations: int = 1, epsilon: float = 1e-12
) -> torch.Tensor:
    """Spectral Normalization with sign handling and stability check."""
    weight = self_weight  # .detach()
    sign_weight = torch.sign(weight)
    weight_abs = weight.abs()

    weight_mat = weight_abs.view(weight_abs.size(0), -1)
    u = torch.randn(weight_mat.size(0), 1, device=weight.device)
    v = torch.randn(weight_mat.size(1), 1, device=weight.device)

    for _ in range(num_iterations):
        v = F.normalize(torch.matmul(weight_mat.t(), u), dim=0)
        u = F.normalize(torch.matmul(weight_mat, v), dim=0)

    sigma = torch.matmul(u.t(), torch.matmul(weight_mat, v))
    sigma = torch.clamp(sigma, min=epsilon)  # Ensure sigma is not too small

    weight_normalized = (
        weight_abs / sigma
    )  # Normalize the weight by the largest singular value
    # weight_updated = weight * weight_normalized.view_as(self_weight)
    # weight_updated = sign_weight * weight_normalized.view_as(self_weight)

    # weight_updated = sign_weight * torch.pow(self_weight, 2 + weight_normalized.view_as(self_weight))
    exponent = 1 + weight_normalized.view_as(self_weight)
    exponent = torch.clamp(exponent, max=10)  # Clamp to prevent overflow

    weight_updated = sign_weight * torch.pow(weight_abs, exponent)

    return weight_updated


class PowerPropLinear(nn.Module):
    """Powerpropagation Linear module."""

    def __init__(
        self,
        alpha: float,
        sparsity: float,
        in_features: int,
        out_features: int,
        bias: bool = True,
    ):
        super(PowerPropLinear, self).__init__()
        self.alpha = alpha
        self.sparsity = sparsity
        self.in_features = in_features
        self.out_features = out_features
        self.b = bias
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if self.b else None

    def __repr__(self):
        return (
            f"PowerPropLinear(alpha={self.alpha}, sparsity={self.sparsity},"
            f" in_features={self.in_features},"
            f" out_features={self.out_features}, bias={self.b})"
        )

    def get_weight(self):
        weight = self.weight.detach()
        if self.alpha == 1.0:
            return weight
        elif self.alpha < 0:
            return spectral_norm(weight)
        return torch.sign(weight) * torch.pow(torch.abs(weight), self.alpha)
        # return self.weight * torch.pow(torch.abs(self.weight), self.alpha - 1.0)

    def forward(self, inputs, mask=None):
        # Apply the re-parametrisation to `self.weight` using `self.alpha`
        # weight = self.weight * torch.pow(torch.abs(self.weight), self.alpha - 1.0)
        if self.alpha == 1.0:
            weight = self.weight
        elif self.alpha < 0:
            weight = spectral_norm(self.weight)
        else:
            weight = torch.sign(self.weight) * torch.pow(
                torch.abs(self.weight), self.alpha
            )
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
        sparsity: float,
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
        self.sparsity = sparsity
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
            f"PowerPropConv2D(alpha={self.alpha}, sparsity={self.sparsity},"
            f" in_channels={self.in_channels},"
            f" out_channels={self.out_channels}, kernel_size={self.kernel_size},"
            f" bias={self.b}, stride={self.stride}, padding={self.padding},"
            f" dilation={self.dilation}, groups={self.groups})"
        )

    def get_weights(self):
        weights = self.weight.detach()
        if self.alpha == 1.0:
            return weights
        elif self.alpha < 0:
            return spectral_norm(weights)
        return torch.sign(weights) * torch.pow(torch.abs(weights), self.alpha)
        # return self.weight * torch.pow(torch.abs(self.weight), self.alpha - 1.0)

    def forward(self, inputs, mask=None):
        # Apply the re-parametrisation to `self.weight` using `self.alpha`
        # weight = self.weight * torch.pow(torch.abs(self.weight), self.alpha - 1.0)
        if self.alpha == 1.0:
            weight = self.weight
        elif self.alpha < 0:
            weight = spectral_norm(self.weight)
        else:
            weight = torch.sign(self.weight) * torch.pow(
                torch.abs(self.weight), self.alpha
            )
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


class PowerPropConv1D(nn.Module):
    """Powerpropagation Conv1D module."""

    def __init__(
        self,
        alpha: float,
        sparsity: float,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: Union[_size, _int] = 1,
        padding: Union[_size, _int] = 1,
        dilation: Union[_size, _int] = 1,
        groups: _int = 1,
        bias: bool = False,
    ):
        super(PowerPropConv1D, self).__init__()
        self.alpha = alpha
        self.sparsity = sparsity
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.b = bias
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size))
        self.bias = nn.Parameter(torch.empty(out_channels)) if bias else None
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

    def __repr__(self):
        return (
            f"PowerPropConv1D(alpha={self.alpha}, sparsity={self.sparsity},"
            f" in_channels={self.in_channels},"
            f" out_channels={self.out_channels}, kernel_size={self.kernel_size},"
            f" bias={self.b}, stride={self.stride}, padding={self.padding},"
            f" dilation={self.dilation}, groups={self.groups})"
        )

    def get_weights(self):
        weights = self.weight.detach()
        if self.alpha == 1.0:
            return weights
        return torch.sign(weights) * torch.pow(torch.abs(weights), self.alpha)
        # return self.weight * torch.pow(torch.abs(self.weight), self.alpha - 1.0)

    def forward(self, inputs, mask=None):
        # Apply the re-parametrisation to `self.weight` using `self.alpha`
        # weight = self.weight * torch.pow(torch.abs(self.weight), self.alpha - 1.0)
        if self.alpha == 1.0:
            weight = self.weight
        else:
            weight = torch.sign(self.weight) * torch.pow(
                torch.abs(self.weight), self.alpha
            )
        # Apply a mask, if given
        if mask is not None:
            weight *= mask

        return F.conv1d(
            input=inputs,
            weight=weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
