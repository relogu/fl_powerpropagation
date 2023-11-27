"""
TODO:
"""

from typing import Union

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.grad import conv2d_input, conv2d_weight
from torch.nn.modules.utils import _pair
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.types import _int, _size

from project.task.utils.drop import (
    drop_nhwc_send_th,
    drop_structured,
    drop_structured_filter,
    drop_threshold,
    matrix_drop,
)


class swat_linear(Function):
    @staticmethod
    def forward(ctx, input, weight, bias, sparsity):
        sparse_weight = matrix_drop(weight, 1 - sparsity)
        if input.dim() == 2 and bias is not None:
            # The fused op is marginally faster
            output = torch.addmm(bias, input, sparse_weight.t())
        else:
            output = input.matmul(sparse_weight.t())
            if bias is not None:
                output += bias
        sparse_input = matrix_drop(input, 1 - sparsity)
        ctx.save_for_backward(sparse_input, sparse_weight, bias)
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


class swat_conv2d_unstructured(Function):
    @staticmethod
    def forward(
        ctx,
        input,
        weight,
        bias,
        sparsity,
        in_threshold,
        wt_threshold,
        stride,
        padding,
        dilation,
        groups,
    ):
        if wt_threshold < 0.0:
            sparse_weight, wt_threshold = drop_nhwc_send_th(weight, 1 - sparsity)
        else:
            sparse_weight = drop_threshold(weight, wt_threshold)
        # sparse_weight = torch.mul(weight, index)
        output = F.conv2d(
            input=input,
            weight=sparse_weight,
            # weight=weight,
            bias=bias,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )
        if in_threshold < 0.0:
            sparse_input, in_threshold = drop_nhwc_send_th(input, 1 - sparsity)
        else:
            sparse_input = drop_threshold(input, in_threshold)
        # ctx.save_for_backward(input, weight, bias)
        ctx.conf = {
            "stride": stride,
            "padding": padding,
            "dilation": dilation,
            "groups": groups,
        }
        # sparse_input = torch.mul(input, index)
        ctx.save_for_backward(sparse_input, sparse_weight, bias)
        return output, wt_threshold, in_threshold

    # Use @once_differentiable by default unless we intend to double backward
    @staticmethod
    # @once_differentiable
    def backward(ctx, grad_output, grad_wt_th, grad_in_th):
        # grad_output = grad_output.contiguous()
        return convolution_backward(ctx, grad_output)


class swat_conv2d_structured_channel(Function):
    @staticmethod
    def forward(
        ctx,
        input,
        weight,
        bias=None,
        sparsity=None,
        in_threshold=None,
        stride=None,
        padding=None,
        dilation=None,
        groups=None,
    ):
        weight = drop_structured(weight, 1 - sparsity.item())
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
            input, in_threshold = drop_nhwc_send_th(input, 1 - sparsity.item())
        else:
            input = drop_threshold(input, in_threshold.item())
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
        bias=None,
        sparsity=None,
        in_threshold=None,
        stride=None,
        padding=None,
        dilation=None,
        groups=None,
    ):
        weight = drop_structured_filter(weight, 1 - sparsity.item())
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
            input, in_threshold = drop_nhwc_send_th(input, 1 - sparsity.item())
        else:
            input = drop_threshold(input, in_threshold.item())
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
        return (
            f"SWATLinear(alpha={self.alpha}, in_features={self.in_features},"
            f" out_features={self.out_features}, bias={self.b},"
            f" sparsity={self.sparsity})"
        )

    def get_weights(self):
        weights = self.weight.detach()
        return torch.sign(weights) * torch.pow(torch.abs(weights), self.alpha)

    def _call_swat_linear(self, input, weight) -> torch.Tensor:
        return swat_linear.apply(input, weight, self.bias, self.sparsity)

    def forward(self, input):
        # Apply the re-parametrisation to `self.weight` using `self.alpha`
        if self.alpha == 1.0:
            powerprop_weight = self.weight
        else:
            powerprop_weight = self.weight * torch.pow(
                torch.abs(self.weight), self.alpha - 1.0
            )
        # Perform SWAT forward pass
        output = self._call_swat_linear(input, powerprop_weight)
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
        return torch.sign(weight) * torch.pow(torch.abs(weight), self.alpha)

    def _call_swat_conv2d(self, input, weight) -> torch.Tensor:
        # TODO: Add the decision pipeline proposed in the original paper
        if self.pruning_type == "unstructured":
            output, wt_threshold, in_threshold = swat_conv2d_unstructured.apply(
                input,
                weight,
                self.bias,
                self.sparsity,
                # deepcopy(self.in_threshold),
                # deepcopy(self.wt_threshold),
                self.in_threshold,
                self.wt_threshold,
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
                # deepcopy(self.in_threshold),
                self.in_threshold,
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
                # deepcopy(self.in_threshold),
                self.in_threshold,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )
        else:
            assert 0, "Illegal Pruning Type"
        # if self.epoch >= self.warmup:
        #     if self.batch_idx % self.period == 0:
        #         self.wt_threshold = wt_threshold
        #         self.in_threshold = in_threshold
        return output

    def forward(self, input):
        # Apply the re-parametrisation to `self.weight` using `self.alpha`
        if self.alpha == 1.0:
            powerprop_weight = self.weight
        else:
            powerprop_weight = self.weight * torch.pow(
                torch.abs(self.weight), self.alpha - 1.0
            )
        # Perform SWAT forward pass
        output = self._call_swat_conv2d(
            input,
            powerprop_weight,
        )
        # Return the output
        return output


if __name__ == "__main__":
    from logging import INFO
    from torch.autograd import gradcheck
    from flwr.common.logger import log

    # from drop import drop_nhwc_send_th

    log(INFO, "Test the gradients of SWATFunctions")
    # gradcheck takes a tuple of tensors as input, check if your gradient
    # evaluated with these tensors are close enough to numerical
    # approximations and returns True if they all verify this condition.
    log(INFO, "Set up fake inputs for `swat_linear` with sparsity level 0.0")
    input = (
        torch.randn(20, 20, dtype=torch.double, requires_grad=True),
        torch.randn(10, 20, dtype=torch.double, requires_grad=True),
        torch.randn(10, dtype=torch.double, requires_grad=True),
        0.0,
    )
    log(INFO, "Run `grad_check` function for `swat_linear`")
    test = gradcheck(swat_linear.apply, input, eps=1e-6, atol=1e-4)
    log(INFO, f"Are the numerical gradients close enough? {test}")
    log(INFO, "Set up fake inputs for `swat_linear` with sparsity level 0.5")
    input = (
        torch.randn(20, 20, dtype=torch.double, requires_grad=True),
        torch.randn(10, 20, dtype=torch.double, requires_grad=True),
        torch.randn(10, dtype=torch.double, requires_grad=True),
        torch.tensor(0.5, requires_grad=False),
    )
    log(INFO, "Run `grad_check` function for `swat_linear`")
    test = gradcheck(
        swat_linear.apply, input, eps=1e-6, atol=1e-4, raise_exception=False
    )
    log(INFO, f"Are the numerical gradients close enough? {test}")
    log(INFO, "Set up fake inputs for `swat_conv2d_*` with sparsity level 0.0")
    # (conv1): SWATConv2D(alpha=1, in_channels=3, out_channels=64, kernel_size=3, bias=False, stride=(1, 1), padding=(1, 1), dilation=1, groups=1, sparsity=0.3, pruning_type=unstructured, warm_up=0, period=1)
    input_alternative = (
        # torch.randn(1,3,32,32,dtype=torch.double,requires_grad=True),
        # torch.randn(64,3,3,3,dtype=torch.double,requires_grad=True),
        # torch.randn(64,dtype=torch.double,requires_grad=True), # Gradients are correct here
        drop_nhwc_send_th(
            torch.ones(1, 1, 3, 3, dtype=torch.double, requires_grad=True), 1.0
        )[0],
        drop_nhwc_send_th(
            torch.ones(1, 1, 1, 2, dtype=torch.double, requires_grad=True), 1.0
        )[0],
        torch.zeros(
            1, dtype=torch.double, requires_grad=True
        ),  # Gradients are correct here
        torch.tensor(0.000000, dtype=torch.double, requires_grad=False),
        torch.tensor(-1.0, dtype=torch.double, requires_grad=False),
        torch.tensor(-1.0, dtype=torch.double, requires_grad=False),
        1,
        0,
        1,
        1,
    )
    log(INFO, "Run `grad_check` function for `swat_conv2d_unstructured`")
    test = gradcheck(
        swat_conv2d_unstructured.apply, input_alternative, eps=1e-6, atol=1e-4
    )
    log(INFO, f"Are the numerical gradients close enough? {test}")

    # log(INFO, "Run `grad_check` function for `swat_conv2d_structured_filter`")
    # test = gradcheck(swat_conv2d_structured_filter.apply, input, eps=1e-6, atol=1e-4, raise_exception=False)
    # log(INFO, f"Are the numerical gradients close enough? {test}")

    # log(INFO, "Run `grad_check` function for `swat_conv2d_structured_filter`")
    # test = gradcheck(swat_conv2d_structured_filter.apply, input, eps=1e-6, atol=1e-4, raise_exception=False)
    # log(INFO, f"Are the numerical gradients close enough? {test}")

    # log(INFO, "Set up fake inputs for `swat_conv2d_*` with sparsity level 0.5")
    # input = (
    #     torch.randn(1,3,32,32,dtype=torch.double,requires_grad=True),
    #     torch.randn(64,3,3,3,dtype=torch.double,requires_grad=True),
    #     torch.randn(64,dtype=torch.double,requires_grad=True),
    #     torch.Tensor([0.5]),
    #     torch.Tensor([-1.0]),
    #     torch.Tensor([-1.0]),
    # )
    # log(INFO, "Run `grad_check` function for `swat_conv2d_unstructured`")
    # test = gradcheck(swat_conv2d_unstructured.apply, input, eps=1e-6, atol=1e-4, raise_exception=False)
    # log(INFO, f"Are the numerical gradients close enough? {test}")

    # log(INFO, "Run `grad_check` function for `swat_conv2d_structured_filter`")
    # test = gradcheck(swat_conv2d_structured_filter.apply, input, eps=1e-6, atol=1e-4, raise_exception=False)
    # log(INFO, f"Are the numerical gradients close enough? {test}")

    # log(INFO, "Run `grad_check` function for `swat_conv2d_structured_filter`")
    # test = gradcheck(swat_conv2d_structured_filter.apply, input, eps=1e-6, atol=1e-4, raise_exception=False)
    # log(INFO, f"Are the numerical gradients close enough? {test}")

    log(INFO, "Test the gradients of SWATFunctions finished!")
