"""Define our models, and training and eval functions."""

from copy import deepcopy
from collections.abc import Callable
import numpy as np

import torch
import torch.nn.functional as F

from torch import nn
from torchvision.models import resnet18
from project.task.utils.powerprop_modules import PowerPropConv2D, PowerPropLinear


class M5(nn.Module):
    """M5 model from pytorch tutorial on Speech Command."""

    def __init__(
        self,
        n_input: int = 1,
        n_output: int = 35,
        stride: int = 16,
        n_channel: int = 32,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(n_input, n_channel, kernel_size=80, stride=stride)
        self.bn1 = nn.BatchNorm1d(n_channel)
        self.pool1 = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(n_channel, n_channel, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(n_channel)
        self.pool2 = nn.MaxPool1d(4)
        self.conv3 = nn.Conv1d(n_channel, 2 * n_channel, kernel_size=3)
        self.bn3 = nn.BatchNorm1d(2 * n_channel)
        self.pool3 = nn.MaxPool1d(4)
        self.conv4 = nn.Conv1d(2 * n_channel, 2 * n_channel, kernel_size=3)
        self.bn4 = nn.BatchNorm1d(2 * n_channel)
        self.pool4 = nn.MaxPool1d(4)
        self.fc1 = nn.Linear(2 * n_channel, n_output)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool2(x)
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.pool3(x)
        x = self.conv4(x)
        x = F.relu(self.bn4(x))
        x = self.pool4(x)
        x = F.avg_pool1d(x, x.shape[-1])
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        return F.log_softmax(x, dim=2)


class NetCifarResnet18(nn.Module):
    """A ResNet18 adapted to CIFAR10."""

    def __init__(
        self,
        num_classes: int = 35,
        device: str = "cuda",
        groupnorm: bool = False,
        n_input: int = 1,
        n_output: int = 35,
        stride: int = 16,
        n_channel: int = 32,
    ) -> None:
        """Initialize network."""
        super().__init__()
        self.num_classes = num_classes
        self.device = device
        self.net = resnet18(num_classes=self.num_classes)

        # print the name and the type of all layer of the resnet18 net
        # for name, layer in self.net.named_children():
        #     print(name, layer)
        # !?!?!!??

        # Modify the first convolutional layer to accept 1 channel input
        self.net.conv1 = nn.Conv2d(
            n_input, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )

        # Initialize the weights of the modified convolutional layer
        nn.init.kaiming_normal_(
            self.net.conv1.weight, mode="fan_out", nonlinearity="relu"
        )

        # self.net.maxpool = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.net(x)


# from: https://github.com/castorini/honk/blob/master/utils/model.py
class SerializableModule(nn.Module):
    """A serializable module."""

    def __init__(self) -> None:
        super().__init__()

    def save(self, filename: str) -> None:
        """Save the model to a file."""
        torch.save(self.state_dict(), filename)

    def load(self, filename: str) -> None:
        """Load the model from a file."""
        self.load_state_dict(
            torch.load(filename, map_location=lambda storage, _: storage)
        )


class SpeechResModel(SerializableModule):
    """A ResNet18 adapted to Speech Commands."""

    def __init__(self) -> None:
        super().__init__()
        n_labels = 35
        n_maps = 64  # config["n_feature_maps"]
        self.conv0 = nn.Conv2d(1, n_maps, (3, 3), padding=(1, 1), bias=False)

        # if "res_pool" in config: self.pool = nn.AvgPool2d(config["res_pool"])

        self.n_layers = n_layers = 4  # config["n_layers"]
        dilation = False  # config["use_dilation"]
        if dilation:
            self.convs = [
                nn.Conv2d(
                    n_maps,
                    n_maps,
                    (3, 3),
                    padding=int(2 ** (i // 3)),
                    dilation=int(2 ** (i // 3)),
                    bias=False,
                )
                for i in range(n_layers)
            ]
        else:
            self.convs = [
                nn.Conv2d(n_maps, n_maps, (3, 3), padding=1, dilation=1, bias=False)
                for _ in range(n_layers)
            ]
        for i, conv in enumerate(self.convs):
            self.add_module(f"bn{i + 1}", nn.BatchNorm2d(n_maps, affine=False))
            self.add_module(f"conv{i + 1}", conv)
        self.output = nn.Linear(n_maps, n_labels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = x.unsqueeze(1)
        for i in range(self.n_layers + 1):
            y = F.relu(getattr(self, f"conv{i}")(x))
            if i == 0:
                if hasattr(self, "pool"):
                    y = self.pool(y)
                old_x = y
            if i > 0 and i % 2 == 0:
                x = y + old_x
                old_x = x
            else:
                x = y
            if i > 0:
                x = getattr(self, f"bn{i}")(x)
        x = x.view(x.size(0), x.size(1), -1)  # shape: (batch, feats, o3)
        x = torch.mean(x, 2)
        return self.output(x)


# From pollen: https://github.com/relogu/pollen_worker/.../models/resnet_util.py

# get_resnet18: NetGen = lazy_config_wrapper(NetCifarResnet18)


def init_weights(module: nn.Module) -> None:
    """Initialise PowerPropLinear and PowerPropConv2D layers in the input module."""
    if isinstance(
        module,
        PowerPropLinear | PowerPropConv2D | nn.Linear | nn.Conv2d,
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
                PowerPropLinear | PowerPropConv2D,
            )
            and module.alpha > 1
        ):
            u = torch.sign(u) * torch.pow(torch.abs(u), 1.0 / module.alpha)

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


def replace_layer_with_powerprop(
    module: nn.Module,
    name: str = "Model",  # ? Never used. Give some problem
    alpha: float = 1.0,
    sparsity: float = 0.0,
) -> None:
    """Replace every nn.Conv2d and nn.Linear layers with the PowerProp versions."""
    # for name, layer in module.named_children():
    #     print(name, layer)

    for attr_str in dir(module):
        target_attr = getattr(module, attr_str)
        if type(target_attr) == nn.Conv1d:
            new_conv = PowerPropConv2D(
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
            new_conv = PowerPropLinear(
                alpha=alpha,
                sparsity=sparsity,
                in_features=target_attr.in_features,
                out_features=target_attr.out_features,
                bias=target_attr.bias is not None,
            )
            setattr(module, attr_str, new_conv)

    # for name, layer in module.named_children():
    #     print(name, layer)

    # ? for name, immediate_child_module in module.named_children(): # Previus version
    for model, immediate_child_module in module.named_children():
        replace_layer_with_powerprop(immediate_child_module, model, alpha, sparsity)


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


def get_resnet18(
    alpha: float = 1.0,
    sparsity: float = 0.0,
    num_classes: int = 10,
    pruning_type: str = "unstructured",
) -> Callable[[dict], NetCifarResnet18]:
    """Cifar Resnet18 network generatror."""
    # untrained_net: NetCifarResnet18 = NetCifarResnet18(num_classes=num_classes)
    # untrained_net: NetCifarResnet18 = M5(n_input=1,n_output=35,stride=16,n_channel=32)
    untrained_net: NetCifarResnet18 = SpeechResModel()

    # replace_layer_with_powerprop(
    #     module=untrained_net,
    #     name="NetCifarResnet18",
    #     alpha=alpha,
    #     sparsity=sparsity,
    #     # pruning_type=pruning_type,
    # )

    def init_model(
        module: nn.Module,
    ) -> None:
        init_weights(module)
        for _, immediate_child_module in module.named_children():
            init_model(immediate_child_module)

    init_model(untrained_net)

    def generated_net(_config: dict) -> NetCifarResnet18:
        return deepcopy(untrained_net)

    return generated_net
