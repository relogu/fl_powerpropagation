"""CNN model architecture, training, and testing functions for MNIST."""

import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

from copy import deepcopy
from collections.abc import Callable

from project.types.common import NetGen
from project.utils.utils import lazy_config_wrapper


from project.task.utils.swat_modules import SWATConv2D, SWATLinear
from torchvision.models import resnet18


class Net(nn.Module):
    """Simple CNN adapted from 'PyTorch: A 60 Minute Blitz."""

    def __init__(self) -> None:
        """Initialize the network.

        Returns
        -------
        None
        """
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the CNN.

        Parameters
        ----------
        x : torch.Tensor
            Input Tensor that will pass through the network

        Returns
        -------
        torch.Tensor
            The resulting Tensor after it has passed through the network
        """
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class NetCifarResnet18(nn.Module):
    """A ResNet18 adapted to CIFAR10."""

    def __init__(
        self, num_classes: int, device: str = "cuda", groupnorm: bool = False
    ) -> None:
        """Initialize network."""
        super().__init__()
        self.num_classes = num_classes
        self.device = device
        # As the LEAF people do
        self.net = resnet18(num_classes=10, norm_layer=lambda x: nn.GroupNorm(2, x))
        # self.net = resnet18(num_classes=self.num_classes)
        # replace w/ smaller input layer
        self.net.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        nn.init.kaiming_normal_(
            self.net.conv1.weight, mode="fan_out", nonlinearity="relu"
        )
        # no need for pooling if training for CIFAR-10
        self.net.maxpool = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.net(x)


get_resnet: NetGen = lazy_config_wrapper(NetCifarResnet18)


def init_weights(module: nn.Module) -> None:
    """Initialise PowerPropLinear and PowerPropConv2D layers in the input module."""
    if isinstance(
        module,
        SWATLinear | SWATConv2D | nn.Linear | nn.Conv2d,
    ):
        # Your code here
        fan_in = calculate_fan_in(module.weight.data)

        # constant from scipy.stats.truncnorm.std(a=-2, b=2, loc=0., scale=1.)
        distribution_stddev = 0.87962566103423978

        std = np.sqrt(1.0 / fan_in) / distribution_stddev
        a, b = -2.0 * std, 2.0 * std

        u = nn.init.trunc_normal_(module.weight.data, std=std, a=a, b=b)
        if isinstance(
            module,
            SWATLinear | SWATConv2D,
        ):
            u = torch.sign(u) * torch.pow(torch.abs(u), 1.0 / module.alpha)

        module.weight.data = u
        if module.bias is not None:
            module.bias.data.zero_()


def calculate_fan_in(tensor: torch.Tensor) -> float:
    """Calculate fan in.

    Modified from: https://github.com/pytorch/pytorch/blob/master/torch/nn/init.py
    """
    min_fan_in = 2
    dimensions = tensor.dim()
    if dimensions < min_fan_in:
        raise ValueError(
            "Fan in can not be computed for tensor with fewer than 2 dimensions"
        )

    num_input_fmaps = tensor.size(1)
    receptive_field_size = 1
    if dimensions > min_fan_in:
        for s in tensor.shape[2:]:
            receptive_field_size *= s
    fan_in = num_input_fmaps * receptive_field_size

    return float(fan_in)


def replace_layer_with_swat(
    module: nn.Module,
    name: str = "Model",
    alpha: float = 2.0,
    sparsity: float = 0.3,
    pruning_type: str = "unstructured",
    skip_first: bool = True,
) -> None:
    """Replace every nn.Conv2d and nn.Linear layers with the SWAT versions."""
    for attr_str in dir(module):
        # Skip the first layer, as in SWAT-U
        # ? if skip_first and attr_str in ["conv1", "0"]: continue

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
                pruning_type=pruning_type,
                # pruning_type="structured_channel",
                # pruning_type="structured_filter",
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

    for model, immediate_child_module in module.named_children():
        replace_layer_with_swat(
            immediate_child_module, model, alpha, sparsity, skip_first=False
        )


def get_network_generator_resnet_swat(
    alpha: float = 4, sparsity: float = 0.7, pruning_type: str = "unstructured"
) -> Callable[[dict], NetCifarResnet18]:
    """Swat network generator."""
    untrained_net: NetCifarResnet18 = NetCifarResnet18(num_classes=10)

    replace_layer_with_swat(
        module=untrained_net,
        name="NetCifarResnet18",
        alpha=alpha,
        sparsity=sparsity,
        pruning_type=pruning_type,
    )

    def init_model(
        module: nn.Module,
    ) -> None:
        """Model initializer."""
        init_weights(module)
        for _, immediate_child_module in module.named_children():
            init_model(immediate_child_module)

    init_model(untrained_net)

    def generated_net(_config: dict | None) -> NetCifarResnet18:
        """Net generator."""
        return deepcopy(untrained_net)

    return generated_net
