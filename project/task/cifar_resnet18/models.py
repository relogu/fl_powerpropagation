"""Define our models, and training and eval functions."""

from copy import deepcopy
from collections.abc import Callable
import numpy as np

import torch
import torch.nn.functional as F

from torch import nn
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.net(x)


# get_resnet18: NetGen = lazy_config_wrapper(NetCifarResnet18)


def init_weights(module: nn.Module) -> None:
    """Initialise standard and custom layers in the input module."""
    if isinstance(
        module,
        nn.Linear | nn.Conv2d,
    ):
        # print(f"init_weights: {type(module)}")
        # Your code here
        fan_in = calculate_fan_in(module.weight.data)

        # constant from scipy.stats.truncnorm.std(a=-2, b=2, loc=0., scale=1.)
        distribution_stddev = 0.87962566103423978

        std = np.sqrt(1.0 / fan_in) / distribution_stddev
        a, b = -2.0 * std, 2.0 * std

        u = nn.init.trunc_normal_(module.weight.data, std=std, a=a, b=b)

        module.weight.data = u
        if module.bias is not None:
            module.bias.data.zero_()


def new_init_weights(module: nn.Module) -> None:
    """Initialize weights for linear and convolutional layers."""
    if isinstance(module, nn.Linear | nn.Conv2d):
        fan_in = module.weight.data.size(1)
        fan_out = module.weight.data.size(0)
        if isinstance(module, nn.Conv2d):
            receptive_field_size = np.prod(module.kernel_size) * module.in_channels
            fan_out *= receptive_field_size

        std = np.sqrt(2.0 / (fan_in + fan_out))  # Xavier initialization
        nn.init.normal_(module.weight.data, 0, std)
        if module.bias is not None:
            nn.init.constant_(module.bias.data, 0)


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


def get_resnet18(num_classes: int = 10) -> Callable[[dict], NetCifarResnet18]:
    """Cifar Resnet18 network generatror."""
    untrained_net: NetCifarResnet18 = NetCifarResnet18(num_classes=num_classes)
    # untrained_net.load_state_dict(
    #     generate_random_state_dict(untrained_net, seed=42, sparsity=0.9)
    # )

    def generated_net(_config: dict) -> NetCifarResnet18:
        return deepcopy(untrained_net)

    def init_model(
        module: nn.Module,
    ) -> None:
        init_weights(module)
        for _, immediate_child_module in module.named_children():
            init_model(immediate_child_module)

    init_model(untrained_net)

    return generated_net
