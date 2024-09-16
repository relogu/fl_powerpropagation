"""
TODO:
"""

from copy import deepcopy
from logging import log
import logging
from typing import Union
from matplotlib import pyplot as plt
import numpy as np


import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.grad import conv2d_input, conv2d_weight
from torch.nn.modules.utils import _pair
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.types import _int, _size
from project.fed.utils.utils import (
    get_tensor_sparsity,
    nonzeros_tensor,
    print_nonzeros_tensor,
)

from project.task.utils.drop import (
    drop_nhwc_send_th,
    drop_structured,
    drop_structured_filter,
    drop_threshold,
    matrix_drop,
)

torch.autograd.set_detect_anomaly(True)

top_k_threshold = 0.01


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

    return self_weight * weight_normalized.view_as(self_weight)

    # weight_updated = weight * weight_normalized.view_as(self_weight)
    # weight_updated = sign_weight * weight_normalized.view_as(self_weight)

    # weight_updated = sign_weight * torch.pow(self_weight, 2 + weight_normalized.view_as(self_weight))
    exponent = 1 + weight_normalized.view_as(self_weight)
    exponent = torch.clamp(exponent, max=10)  # Clamp to prevent overflow

    weight_updated = sign_weight * torch.pow(weight_abs, exponent)

    return weight_updated


def convolution_backward(
    ctx,
    grad_output,
):
    sparse_input, sparse_weight, bias = ctx.saved_tensors
    conf = ctx.conf
    input_grad = (
        weight_grad
    ) = (
        bias_grad
    ) = (
        sparsity_grad
    ) = (
        grad_in_th
    ) = grad_wt_th = stride_grad = padding_grad = dilation_grad = groups_grad = None

    # Compute gradient w.r.t. input
    if ctx.needs_input_grad[0]:
        input_grad = conv2d_input(
            sparse_input.shape,
            sparse_weight,
            grad_output,
            conf["stride"],
            conf["padding"],
            conf["dilation"],
            conf["groups"],
        )

    # Compute gradient w.r.t. weight
    if ctx.needs_input_grad[1]:
        weight_grad = conv2d_weight(
            sparse_input,
            sparse_weight.shape,
            grad_output,
            conf["stride"],
            conf["padding"],
            conf["dilation"],
            conf["groups"],
        )

    # Compute gradient w.r.t. bias (works for every Conv2d shape)
    if bias is not None and ctx.needs_input_grad[2]:
        bias_grad = grad_output.sum(dim=(0, 2, 3))

    return (
        input_grad,
        weight_grad,
        bias_grad,
        sparsity_grad,
        grad_in_th,
        grad_wt_th,
        stride_grad,
        padding_grad,
        dilation_grad,
        groups_grad,
    )


class swat_linear(Function):
    @staticmethod
    def forward(ctx, input, weight, bias, sparsity):

        if input.dim() == 2 and bias is not None:
            # The fused op is marginally faster
            output = torch.addmm(bias, input, weight.t())
        else:
            output = input.matmul(weight.t())
            if bias is not None:
                output += bias

        topk = 1 - sparsity
        if (
            topk < top_k_threshold
        ):  # necassary since too small values create problem to drop !?
            topk = top_k_threshold
        # Must be decided a new treshold

        sparse_input = matrix_drop(input, topk)

        ctx.save_for_backward(sparse_input, weight, bias)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        sparse_input, sparse_weight, bias = ctx.saved_tensors

        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(sparse_weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(sparse_input)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)

        return grad_input, grad_weight, grad_bias, None


class SWATLinear(nn.Module):
    """Powerpropagation Linear module."""

    def __init__(
        self,
        alpha: float,
        in_features: int,
        out_features: int,
        bias: bool = True,
        sparsity: float = 0.3,
    ):
        super(SWATLinear, self).__init__()
        self.alpha = alpha
        self.in_features = in_features
        self.out_features = out_features
        self.b = bias
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if self.b else None
        self.sparsity = sparsity

    def __repr__(self):
        return (
            f"SWATLinear(alpha={self.alpha}, in_features={self.in_features},"
            f" out_features={self.out_features}, bias={self.b},"
            f" sparsity={self.sparsity})"
        )

    def get_weights(self):
        weights = self.weight.detach()
        if self.alpha == 1.0:
            return weights
        elif self.alpha < 0:
            return spectral_norm(weights)
        return torch.sign(weights) * torch.pow(torch.abs(weights), self.alpha)

    def _call_swat_linear(self, input, weight) -> torch.Tensor:
        if self.training:
            sparsity = get_tensor_sparsity(weight)
        else:
            # Avoid to sparsify during the evaluation
            sparsity = 0.0
        return swat_linear.apply(input, weight, self.bias, sparsity)

    def forward(self, input):
        # Apply the re-parametrisation to `self.weight` using `self.alpha`
        if self.alpha == 1.0:
            powerprop_weight = self.weight
        elif self.alpha < 0:
            powerprop_weight = spectral_norm(self.weight)
        else:
            # powerprop_weight = self.weight * torch.pow(torch.abs(self.weight), self.alpha - 1.0)
            powerprop_weight = torch.sign(self.weight) * torch.pow(
                torch.abs(self.weight), self.alpha
            )

        # Perform SWAT forward pass
        output = self._call_swat_linear(input, powerprop_weight)

        # Return the output
        return output


class swat_conv2d(Function):
    @staticmethod
    def forward(
        ctx,
        input,
        weight,
        bias,
        sparsity,
        in_threshold,
        stride,
        padding,
        dilation,
        groups,
    ):
        # Ensure input tensor is contiguous
        input = input.contiguous()
        # weight = weight.contiguous()
        # if bias is not None:
        #     bias = bias.contiguous()

        output = F.conv2d(
            input=input,
            weight=weight,
            bias=bias,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )

        # the output is not sparsified, it make sense?

        # Sparsifying activations
        # /SWAT-code/cifar10-100-code/custom_layers/custom_conv.py, sparsify_activations
        # The threshold is computed in the forward pass, only the first time
        # In the original paper is calcolated aftear a warmup period every tot epochs
        # if self.epoch >= self.warmup:
        #     if self.batch_idx % self.period == 0:

        topk = 1 - sparsity
        # necassary since too small values create problem to drop !???
        if topk < top_k_threshold:
            topk = top_k_threshold

        sparse_input = matrix_drop(input, topk)
        if in_threshold < 0.0:
            sparse_input, in_threshold_tensor = drop_nhwc_send_th(input, topk)
            in_threshold = in_threshold_tensor.item()
        else:
            sparse_input = drop_threshold(input, in_threshold)

        ctx.conf = {
            "stride": stride,
            "padding": padding,
            "dilation": dilation,
            "groups": groups,
        }

        ctx.save_for_backward(sparse_input, weight, bias)

        return output, in_threshold

    # Use @once_differentiable by default unless we intend to double backward
    @staticmethod
    @once_differentiable
    # def backward(ctx, grad_output, grad_wt_th, grad_in_th):
    def backward(ctx, grad_output, grad_in_th):
        grad_output = grad_output.contiguous()
        return convolution_backward(ctx, grad_output)


class SWATConv2D(nn.Module):
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
        sparsity: float = 0.3,
        pruning_type: str = "unstructured",
        warm_up: int = 0,
        period: int = 1,
    ):
        super(SWATConv2D, self).__init__()
        self.alpha = alpha
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.b = bias
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size, kernel_size)
        )
        self.bias = nn.Parameter(torch.empty(out_channels)) if bias else None
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.sparsity = sparsity
        self.pruning_type = pruning_type
        self.warmup = warm_up
        self.period = period
        self.wt_threshold = -1.0
        self.in_threshold = -1.0
        self.epoch = 0
        self.batch_idx = 0

    def __repr__(self):
        return (
            f"SWATConv2D(alpha={self.alpha}, in_channels={self.in_channels},"
            f" out_channels={self.out_channels}, kernel_size={self.kernel_size},"
            f" bias={self.b}, stride={self.stride}, padding={self.padding},"
            f" dilation={self.dilation}, groups={self.groups},"
            f" sparsity={self.sparsity}, pruning_type={self.pruning_type},"
            f" warm_up={self.warmup}, period={self.period})"
        )

    def get_weight(self):
        weight = self.weight.detach()
        if self.alpha == 1.0:
            return weight
        if self.alpha < 0:
            return spectral_norm(weight)
        return torch.sign(weight) * torch.pow(torch.abs(weight), self.alpha)

    def _call_swat_conv2d(self, input, weight) -> torch.Tensor:

        if self.training:
            # for the activation the sparsity used is proportional to the weight sparsity
            sparsity = get_tensor_sparsity(weight)
        else:
            # Avoid to sparsify during the evaluation
            sparsity = 0.0

        output, in_threshold = swat_conv2d.apply(
            input,
            weight,
            self.bias,
            sparsity,
            self.in_threshold,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

        # update self.in_threshold
        if sparsity != 0.0:
            # otherwise, it is not updated
            self.in_threshold = in_threshold

        return output

    def forward(self, input):
        # Apply the re-parametrisation to `self.weight` using `self.alpha`
        if self.alpha == 1.0:
            powerprop_weight = self.weight
        elif self.alpha < 0:
            powerprop_weight = spectral_norm(self.weight)
        else:
            powerprop_weight = torch.sign(self.weight) * torch.pow(
                torch.abs(self.weight), self.alpha
            )
            # powerprop_weight = self.weight * torch.pow(
            #     self.weight.abs(), self.alpha - 1
            # )

        # Perform the forward pass
        output = self._call_swat_conv2d(
            input,
            powerprop_weight,
        )

        # Return the output
        return output
