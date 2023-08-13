from copy import deepcopy
from typing import Iterable, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from powerprop_modules import PowerPropConv2D, PowerPropLinear
from swat_modules import SWATConv2D, SWATLinear
from torch import nn
from torchvision.models import resnet18


class Net(nn.Module):
    """Simple CNN adapted from 'PyTorch: A 60 Minute Blitz'."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Net_Cifar_Resnet18(nn.Module):
    def __init__(
        self, num_classes: int, device: str = "cuda", groupnorm: bool = False
    ) -> None:
        """A ResNet18 adapted to CIFAR10."""
        super(Net_Cifar_Resnet18, self).__init__()
        self.num_classes = num_classes
        self.device = device
        # As the LEAF people do
        # self.net = resnet18(num_classes=10, norm_layer=lambda x: nn.GroupNorm(2, x))
        self.net = resnet18(num_classes=self.num_classes)
        # replace w/ smaller input layer
        self.net.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        nn.init.kaiming_normal_(
            self.net.conv1.weight, mode="fan_out", nonlinearity="relu"
        )
        # no need for pooling if training for CIFAR-10
        self.net.maxpool = nn.Identity()

    def forward(self, x):
        return self.net(x)


def init_weights(module: nn.Module):
    """Initialise PowerPropLinear and PowerPropConv2D layers in the input module."""
    if isinstance(
        module,
        (
            PowerPropLinear,
            PowerPropConv2D,
            SWATLinear,
            SWATConv2D,
            nn.Linear,
            nn.Conv2d,
        ),
    ):
        fan_in = calculate_fan_in(module.weight.data)

        # constant from scipy.stats.truncnorm.std(a=-2, b=2, loc=0., scale=1.)
        distribution_stddev = 0.87962566103423978

        std = np.sqrt(1.0 / fan_in) / distribution_stddev
        a, b = -2.0 * std, 2.0 * std

        u = nn.init.trunc_normal_(module.weight.data, std=std, a=a, b=b)
        if isinstance(
            module,
            (
                PowerPropLinear,
                PowerPropConv2D,
                SWATLinear,
                SWATConv2D,
            ),
        ):
            u = torch.sign(u) * torch.pow(torch.abs(u), 1.0 / module.alpha)

        module.weight.data = u
        if module.bias is not None:
            module.bias.data.zero_()


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


def replace_layer_with_powerprop(
    module: nn.Module,
    name: str = "Model",
    alpha: float = 2.0,
):
    """Replace every nn.Conv2d and nn.Linear layers with the PowerProp versions."""
    for attr_str in dir(module):
        target_attr = getattr(module, attr_str)
        if type(target_attr) == nn.Conv2d:
            new_conv = PowerPropConv2D(
                alpha=alpha,
                in_channels=target_attr.in_channels,
                out_channels=target_attr.out_channels,
                kernel_size=target_attr.kernel_size[0],
                bias=target_attr.bias is not None,
                padding=target_attr.padding,
                stride=target_attr.stride,
            )
            setattr(module, attr_str, new_conv)
        if type(target_attr) == nn.Linear:
            new_conv = PowerPropLinear(
                alpha=alpha,
                in_features=target_attr.in_features,
                out_features=target_attr.out_features,
                bias=target_attr.bias is not None,
            )
            setattr(module, attr_str, new_conv)

    for name, immediate_child_module in module.named_children():
        replace_layer_with_powerprop(immediate_child_module, name, alpha)


def replace_layer_with_swat(
    module: nn.Module,
    name: str = "Model",
    alpha: float = 2.0,
    sparsity: float = 0.3,
):
    """Replace every nn.Conv2d and nn.Linear layers with the SWAT versions."""
    for attr_str in dir(module):
        target_attr = getattr(module, attr_str)
        if type(target_attr) == nn.Conv2d:
            new_conv = SWATConv2D(
                alpha=alpha,
                in_channels=target_attr.in_channels,
                out_channels=target_attr.out_channels,
                kernel_size=target_attr.kernel_size[0],
                bias=target_attr.bias is not None,
                padding=target_attr.padding,
                stride=target_attr.stride,
                sparsity=sparsity,
                pruning_type="unstructured",
                warm_up=0,
                period=1,
            )
            setattr(module, attr_str, new_conv)
        if type(target_attr) == nn.Linear:
            new_conv = SWATLinear(
                alpha=alpha,
                in_features=target_attr.in_features,
                out_features=target_attr.out_features,
                bias=target_attr.bias is not None,
                sparsity=sparsity,
            )
            setattr(module, attr_str, new_conv)

    for name, immediate_child_module in module.named_children():
        replace_layer_with_swat(immediate_child_module, name, alpha)


def get_parameters_to_prune(
    net: nn.Module,
) -> Iterable[Tuple[nn.Module, str]]:
    """Return an iterable of tuples containing the PowerPropConv2D layers in the input model."""
    parameters_to_prune = []

    def add_immediate_child(
        module: nn.Module,
        name: str,
    ) -> None:
        if (
            type(module) == PowerPropConv2D
            or type(module) == nn.Conv2d
            or type(module) == nn.Linear
            or type(module) == PowerPropLinear
        ):
            parameters_to_prune.append((module, "w", name))
        for name, immediate_child_module in module.named_children():
            add_immediate_child(immediate_child_module, name)

    add_immediate_child(net, "Net")

    return parameters_to_prune


# All experiments will have the exact same initialization.
# All differences in performance will come from training
def get_network_generator_cnn():
    untrained_net: Net = Net()

    def generated_net():
        return deepcopy(untrained_net)

    return generated_net


def get_network_generator_resnet() -> Net_Cifar_Resnet18:
    untrained_net: Net_Cifar_Resnet18 = Net_Cifar_Resnet18(num_classes=10)

    def generated_net() -> Net_Cifar_Resnet18:
        return deepcopy(untrained_net)

    return generated_net


def get_network_generator_resnet_powerprop(alpha: float = 2.0) -> Net_Cifar_Resnet18:
    untrained_net: Net_Cifar_Resnet18 = Net_Cifar_Resnet18(num_classes=10)
    replace_layer_with_powerprop(
        module=untrained_net,
        name="Net_Cifar_Resnet18",
        alpha=alpha,
    )

    def init_model(
        module: nn.Module,
    ) -> None:
        init_weights(module)
        for _, immediate_child_module in module.named_children():
            init_model(immediate_child_module)

    init_model(untrained_net)

    def generated_net() -> Net_Cifar_Resnet18:
        return deepcopy(untrained_net)

    return generated_net


def get_network_generator_resnet_swat(
    alpha: float = 2.0, sparsity: float = 0.3
) -> Net_Cifar_Resnet18:
    untrained_net: Net_Cifar_Resnet18 = Net_Cifar_Resnet18(num_classes=10)
    replace_layer_with_swat(
        module=untrained_net,
        name="Net_Cifar_Resnet18",
        alpha=alpha,
        sparsity=sparsity,
    )

    def init_model(
        module: nn.Module,
    ) -> None:
        init_weights(module)
        for _, immediate_child_module in module.named_children():
            init_model(immediate_child_module)

    init_model(untrained_net)

    def generated_net() -> Net_Cifar_Resnet18:
        return deepcopy(untrained_net)

    return generated_net
