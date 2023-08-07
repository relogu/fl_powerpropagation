from copy import deepcopy
from typing import Iterable, Tuple
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import resnet18

from powerprop_modules import PowerPropConv, init_weights


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
    def __init__(self, num_classes: int, device: str = 'cuda', groupnorm: bool = False) -> None:
        """ A ResNet18 adapted to CIFAR10. """
        super(Net_Cifar_Resnet18, self).__init__()
        self.num_classes = num_classes
        self.device = device
        # As the LEAF people do
        # self.net = resnet18(num_classes=10, norm_layer=lambda x: nn.GroupNorm(2, x))
        self.net = resnet18(num_classes=self.num_classes)
        # replace w/ smaller input layer
        self.net.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        nn.init.kaiming_normal_(self.net.conv1.weight, mode='fan_out', nonlinearity='relu')
        # no need for pooling if training for CIFAR-10
        self.net.maxpool = torch.nn.Identity()
    def forward(self, x):
        return self.net(x)
    
def replace_conv_with_powerprop(
    module: nn.Module,
    name: str = 'Model',
    alpha: float = 2.0,
):
    """Replace every torch.nn.Conv2d layer with PowerPropConv"""
    for attr_str in dir(module):
        target_attr = getattr(module, attr_str)
        if type(target_attr) == torch.nn.Conv2d:
            new_conv = PowerPropConv(
                alpha=alpha,
                in_channels=target_attr.in_channels,
                out_channels=target_attr.out_channels,
                kernel_size=target_attr.kernel_size[0],
                padding=target_attr.padding,
                stride=target_attr.stride,
            )
            setattr(module, attr_str, new_conv)
    
    for name, immediate_child_module in module.named_children():
        replace_conv_with_powerprop(immediate_child_module, name, alpha)
        
def get_parameters_to_prune(
    net: nn.Module,
) -> Iterable[Tuple[nn.Module, str]]:
    """Return an iterable of tuples containing the PowerPropConv layers in the input model."""
    parameters_to_prune = []
    
    def add_immediate_child(
        module: nn.Module,
    ) -> None:
        if type(module) == PowerPropConv:
            parameters_to_prune.append(
                (module, 'w')
            )
        for _, immediate_child_module in module.named_children():
            add_immediate_child(immediate_child_module)
    
    add_immediate_child(net)

    return parameters_to_prune


# All experiments will have the exact same initialization.
# All differences in performance will come from training
def get_network_generator_cnn():
    untrained_net: Net = Net()

    def generated_net():
        return deepcopy(untrained_net)

    return generated_net

def get_network_generator_resnet() -> Net_Cifar_Resnet18:
    untrained_net: Net_Cifar_Resnet18 = Net_Cifar_Resnet18(
        num_classes=10
    )

    def generated_net() -> Net_Cifar_Resnet18:
        return deepcopy(untrained_net)

    return generated_net

def get_network_generator_resnet_powerprop(alpha: float = 2.0) -> Net_Cifar_Resnet18:
    untrained_net: Net_Cifar_Resnet18 = Net_Cifar_Resnet18(
        num_classes=10
    )
    replace_conv_with_powerprop(
        module=untrained_net,
        name='Net_Cifar_Resnet18',
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




