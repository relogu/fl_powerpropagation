"""
TODO:
"""

from typing import Union
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.types import _int, _size

from drop import (
    drop_nhwc_send_th,
    drop_structured,
    drop_structured_filter,
    drop_threshold,
    matrix_drop,
)


class swat_linear(Function):
    @staticmethod
    def forward(ctx, input, weight, bias, sparsity):
        weight = matrix_drop(weight, 1 - sparsity)
        if input.dim() == 2 and bias is not None:
            # The fused op is marginally faster
            output = torch.addmm(bias, input, weight.t())
        else:
            output = input.matmul(weight.t())
            if bias is not None:
                output += bias
        input = matrix_drop(input, 1 - sparsity)
        ctx.save_for_backward(input, weight)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None
        grad_input = torch.mm(grad_output, weight)
        grad_weight = torch.mm(input.t(), grad_output)
        grad_bias = torch.sum(grad_output, 0)
        return grad_input, grad_weight.t(), grad_bias, None


def convolution_backward(
    ctx,
    grad_output,
):
    input, weight, bias = ctx.saved_tensors
    conf = ctx.conf
    input_grad = (
        weight_grad
    ) = (
        bias_grad
    ) = (
        sparsity_grad
    ) = (
        grad_in_th
    ) = grad_out_th = stride_grad = padding_grad = dilation_grad = groups_grad = None
    # Compute gradient w.r.t. input
    if ctx.needs_input_grad[0]:
        input_grad = torch.nn.grad.conv2d_input(
            input.shape,
            weight,
            grad_output,
            conf["stride"],
            conf["padding"],
            conf["dilation"],
            conf["groups"],
        )
    # Compute gradient w.r.t. weight
    if ctx.needs_input_grad[1]:
        weight_grad = torch.nn.grad.conv2d_weight(
            input,
            weight.shape,
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
        grad_out_th,
        stride_grad,
        padding_grad,
        dilation_grad,
        groups_grad,
    )


class swat_conv2d_unstructured(Function):
    @staticmethod
    def forward(
        ctx,
        input,
        weight,
        bias,
        sparsity,
        wt_threshold,
        in_threshold,
        stride,
        padding,
        dilation,
        groups,
    ):
        if wt_threshold is None:
            weight, wt_threshold = drop_nhwc_send_th(weight, 1 - sparsity)
        else:
            weight = drop_threshold(weight, wt_threshold)
        output = F.conv2d(
            input=input,
            weight=weight,
            bias=bias,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )
        if in_threshold is None:
            input, in_threshold = drop_nhwc_send_th(input, 1 - sparsity)
        else:
            input = drop_threshold(input, in_threshold)
        ctx.save_for_backward(input, weight, bias)
        ctx.conf = {
            "stride": stride,
            "padding": padding,
            "dilation": dilation,
            "groups": groups,
        }
        return output, wt_threshold, in_threshold

    # Use @once_differentiable by default unless we intend to double backward
    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output, grad_wt_th, grad_in_th):
        return convolution_backward(ctx, grad_output)


class swat_conv2d_structured_channel(Function):
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
        weight = drop_structured(weight, 1 - sparsity)
        output = F.conv2d(
            input=input,
            weight=weight,
            bias=bias,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )
        if in_threshold is None:
            input, in_threshold = drop_nhwc_send_th(input, 1 - sparsity)
        else:
            input = drop_threshold(input, in_threshold)
        ctx.save_for_backward(input, weight, bias)
        ctx.conf = {
            "stride": stride,
            "padding": padding,
            "dilation": dilation,
            "groups": groups,
        }
        return output, in_threshold

    # Use @once_differentiable by default unless we intend to double backward
    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output, grad_in_th):
        return convolution_backward(ctx, grad_output)


class swat_conv2d_structured_filter(Function):
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
        weight = drop_structured_filter(weight, 1 - sparsity)
        output = F.conv2d(
            input=input,
            weight=weight,
            bias=bias,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )
        if in_threshold is None:
            input, in_threshold = drop_nhwc_send_th(input, 1 - sparsity)
        else:
            input = drop_threshold(input, in_threshold)
        ctx.save_for_backward(input, weight, bias)
        ctx.conf = {
            "stride": stride,
            "padding": padding,
            "dilation": dilation,
            "groups": groups,
        }
        return output, in_threshold

    # Use @once_differentiable by default unless we intend to double backward
    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output, grad_in_th):
        return convolution_backward(ctx, grad_output)


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
        return f"SWATLinear(alpha={self.alpha}, in_features={self.in_features}, out_features={self.out_features}, bias={self.b}, sparsity={self.sparsity})"

    def get_weights(self):
        weights = self.weight.detach()
        return torch.sign(weights) * torch.pow(torch.abs(weights), self.alpha)

    def _call_swat_linear(self, input, weight) -> torch.Tensor:
        return swat_linear.apply(input, weight, self.bias, self.sparsity)

    def forward(self, input):
        # Apply the re-parametrisation to `self.weight` using `self.alpha`
        weight = self.weight * torch.pow(torch.abs(self.weight), self.alpha - 1.0)
        # Perform SWAT forward pass
        output = self._call_swat_linear(input, weight)
        # Return the output
        return output


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
        self.sparsity = sparsity
        self.pruning_type = pruning_type
        self.warmup = warm_up
        self.period = period
        self.wt_threshold = None
        self.in_threshold = None
        self.epoch = 0
        self.batch_idx = 0

    def __repr__(self):
        return f"SWATConv2D(alpha={self.alpha}, in_channels={self.in_channels}, out_channels={self.out_channels}, kernel_size={self.kernel_size}, bias={self.b}, stride={self.stride}, padding={self.padding}, dilation={self.dilation}, groups={self.groups}, sparsity={self.sparsity}, pruning_type={self.pruning_type}, warm_up={self.warmup}, period={self.period})"

    def get_weight(self):
        weight = self.weight.detach()
        return torch.sign(weight) * torch.pow(torch.abs(weight), self.alpha)

    def _call_swat_conv2d(self, input, weight) -> torch.Tensor:
        # TODO: Add the decision pipeline proposed in the original paper
        if self.pruning_type == "unstructured":
            output, wt_threshold, in_threshold = swat_conv2d_unstructured.apply(
                input,
                weight,
                self.bias,
                self.sparsity,
                deepcopy(self.in_threshold),
                deepcopy(self.wt_threshold),
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )
        elif self.pruning_type == "structured_channel":
            output, in_threshold = swat_conv2d_structured_channel.apply(
                input,
                weight,
                self.bias,
                self.sparsity,
                deepcopy(self.in_threshold),
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )
        elif self.pruning_type == "structured_filter":
            output, in_threshold = swat_conv2d_structured_filter.apply(
                input,
                weight,
                self.bias,
                self.sparsity,
                deepcopy(self.in_threshold),
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )
        else:
            assert (0, "Illegal Pruning Type")
        if self.epoch >= self.warmup:
            if self.batch_idx % self.period == 0:
                self.wt_threshold = wt_threshold
                self.in_threshold = in_threshold
        return output

    def forward(self, input):
        # Apply the re-parametrisation to `self.weight` using `self.alpha`
        weight = self.weight * torch.pow(torch.abs(self.weight), self.alpha - 1.0)
        # Perform SWAT forward pass
        output = self._call_swat_conv2d(
            input,
            weight,
        )
        # Return the output
        return output
