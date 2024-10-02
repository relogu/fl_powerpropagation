"""CNN model architecture, training, and testing functions for MNIST."""

import torch
from torch import nn
import numpy as np

from copy import deepcopy
from collections.abc import Callable, Iterable
from project.task.cifar_resnet18.models import NetCifarResnet18, calculate_fan_in

from project.types.common import NetGen
from project.utils.utils import lazy_config_wrapper


from project.task.utils.sparsyfed_no_act_modules import (
    SparsyFed_no_act_Conv2D,
    SparsyFed_no_act_linear,
)

get_resnet: NetGen = lazy_config_wrapper(NetCifarResnet18)


def init_weights(module: nn.Module) -> None:
    """Initialise standard and custom layers in the input module."""
    if isinstance(
        module,
        SparsyFed_no_act_linear | SparsyFed_no_act_Conv2D | nn.Linear | nn.Conv2d,
    ):
        # Your code here
        fan_in = calculate_fan_in(module.weight.data)

        # constant from scipy.stats.truncnorm.std(a=-2, b=2, loc=0., scale=1.)
        distribution_stddev = 0.87962566103423978

        std = np.sqrt(1.0 / fan_in) / distribution_stddev
        a, b = -2.0 * std, 2.0 * std

        u = nn.init.trunc_normal_(module.weight.data, std=std, a=a, b=b)
        if (
            isinstance(
                module,
                SparsyFed_no_act_linear | SparsyFed_no_act_Conv2D,
            )
            and module.alpha > 1
        ):
            u = torch.sign(u) * torch.pow(torch.abs(u), 1.0 / module.alpha)

        module.weight.data = u
        if module.bias is not None:
            module.bias.data.zero_()


def replace_layer_with_powerprop(
    module: nn.Module,
    name: str = "Model",  # ? Never used. Give some problem
    alpha: float = 1.0,
    sparsity: float = 0.0,
) -> None:
    """Replace every nn.Conv2d and nn.Linear layers with the PowerProp versions."""
    for attr_str in dir(module):
        target_attr = getattr(module, attr_str)
        if type(target_attr) == nn.Conv2d:
            new_conv = SparsyFed_no_act_Conv2D(
                alpha=alpha,
                sparsity=sparsity,
                in_channels=target_attr.in_channels,
                out_channels=target_attr.out_channels,
                kernel_size=target_attr.kernel_size[0],
                bias=target_attr.bias is not None,
                padding=target_attr.padding,
                stride=target_attr.stride,
            )
            setattr(module, attr_str, new_conv)
        if type(target_attr) == nn.Linear:
            new_conv = SparsyFed_no_act_linear(
                alpha=alpha,
                sparsity=sparsity,
                in_features=target_attr.in_features,
                out_features=target_attr.out_features,
                bias=target_attr.bias is not None,
            )
            setattr(module, attr_str, new_conv)

    # ? for name, immediate_child_module in module.named_children(): # Previus version
    for model, immediate_child_module in module.named_children():
        replace_layer_with_powerprop(immediate_child_module, model, alpha, sparsity)


def get_network_generator_resnet_powerprop(
    alpha: float = 1.0,
    sparsity: float = 0.0,
    num_classes: int = 10,
) -> Callable[[dict], NetCifarResnet18]:
    """Powerprop Resnet generator."""
    untrained_net: NetCifarResnet18 = NetCifarResnet18(num_classes=num_classes)

    replace_layer_with_powerprop(
        module=untrained_net,
        name="NetCifarResnet18",
        alpha=alpha,
        sparsity=sparsity,
    )

    def init_model(
        module: nn.Module,
    ) -> None:
        """Initialize the weights of the layers."""
        init_weights(module)
        for _, immediate_child_module in module.named_children():
            init_model(immediate_child_module)

    init_model(untrained_net)

    def generated_net(_config: dict) -> NetCifarResnet18:
        """Return a deep copy of the untrained network."""
        return deepcopy(untrained_net)

    return generated_net


def get_parameters_to_prune(
    net: nn.Module,
    _first_layer: bool = False,
) -> Iterable[tuple[nn.Module, str, str]]:
    """Pruning.

    Return an iterable of tuples containing the SparsyFed_no_act_Conv2D layers in the
    input model.
    """
    parameters_to_prune = []
    first_layer = _first_layer

    def add_immediate_child(
        module: nn.Module,
        name: str,
    ) -> None:
        nonlocal first_layer
        if (
            type(module) == SparsyFed_no_act_Conv2D
            or type(module) == SparsyFed_no_act_linear
            or type(module) == nn.Conv2d
            or type(module) == nn.Linear
        ):
            if first_layer:
                first_layer = False
            else:
                parameters_to_prune.append((module, "weight", name))

        for _name, immediate_child_module in module.named_children():
            add_immediate_child(immediate_child_module, _name)

    add_immediate_child(net, "Net")

    return parameters_to_prune
