"""SWAT model architecture, training, and testing functions for CIFAR."""

from collections.abc import Iterable
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

from copy import deepcopy
from collections.abc import Callable
from project.fed.utils.utils import generate_random_state_dict
from project.task.cifar_resnet18.models import (
    NetCifarResnet18,
    calculate_fan_in,
    get_resnet18,
)

from project.types.common import NetGen
from project.utils.utils import lazy_config_wrapper


from project.task.utils.swat_modules import SWATConv2D, SWATLinear

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
        if (
            isinstance(
                module,
                SWATLinear | SWATConv2D,
            )
            and module.alpha > 1
        ):
            u = torch.sign(u) * torch.pow(torch.abs(u), 1.0 / module.alpha)

        module.weight.data = u
        if module.bias is not None:
            module.bias.data.zero_()


def replace_layer_with_swat(
    module: nn.Module,
    name: str = "Model",
    alpha: float = 1.0,
    sparsity: float = 0.0,
    pruning_type: str = "unstructured",
    first_layer: bool = True,
) -> None:
    """Replace every nn.Conv2d and nn.Linear layers with the SWAT versions."""
    for attr_str in dir(module):
        target_attr = getattr(module, attr_str)
        if type(target_attr) == nn.Conv2d:
            if first_layer:
                first_layer = False
                continue
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
                warm_up=0,
                period=1,
            )
            setattr(module, attr_str, new_conv)
            # print(f"Replaced {type(target_attr)} with SWATConv2D in {name}")
        if type(target_attr) == nn.Linear:
            if first_layer:
                first_layer = False
                continue
            new_conv = SWATLinear(
                alpha=alpha,
                in_features=target_attr.in_features,
                out_features=target_attr.out_features,
                bias=target_attr.bias is not None,
                sparsity=sparsity,
            )
            setattr(module, attr_str, new_conv)
            # print(f"Replaced {type(target_attr)} with SWATLinear in {name}")

    for model, immediate_child_module in module.named_children():
        replace_layer_with_swat(
            immediate_child_module, model, alpha, sparsity, first_layer=first_layer
        )


def get_network_generator_resnet_swat(
    alpha: float = 1.0,
    sparsity: float = 0.0,
    num_classes: int = 10,
    pruning_type: str = "unstructured",
) -> Callable[[dict], NetCifarResnet18]:
    """Swat network generator."""
    untrained_net: NetCifarResnet18 = NetCifarResnet18(num_classes=num_classes)

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
        """Initialize the weights of the layers."""
        init_weights(module)
        for _, immediate_child_module in module.named_children():
            init_model(immediate_child_module)

    init_model(untrained_net)

    def generated_net(_config: dict | None) -> NetCifarResnet18:
        """Return a deep copy of the untrained network."""
        return deepcopy(untrained_net)

    return generated_net


def get_parameters_to_prune(
    net: nn.Module,
    _first_layer: bool = False,
) -> Iterable[tuple[nn.Module, str, str]]:
    """Pruning.

    Return an iterable of tuples containing the PowerPropConv2D layers in the input
    model.
    """
    parameters_to_prune = []
    first_layer = _first_layer

    def add_immediate_child(
        module: nn.Module,
        name: str,
    ) -> None:
        nonlocal first_layer
        if (
            type(module) == SWATConv2D
            or type(module) == SWATLinear
            or type(module) == nn.Conv2d
            or type(module) == nn.Linear
        ):
            if first_layer and type(module) == SWATLinear:  # !?!?!?
                first_layer = False
            else:
                parameters_to_prune.append((module, "weight", name))

        for _name, immediate_child_module in module.named_children():
            add_immediate_child(immediate_child_module, _name)

    add_immediate_child(net, "Net")

    return parameters_to_prune


def test_out() -> None:
    """Test the output of the SWAT ResNet model."""
    # Create a random input tensor, with seed=42
    torch.manual_seed(42)
    input_tensor = torch.randn(1, 3, 32, 32)
    fake_config = dict([[], []])

    # Create the original ResNet model
    original_model = NetCifarResnet18(num_classes=10)
    original_model.load_state_dict(
        generate_random_state_dict(original_model, seed=42, sparsity=0)
    )

    # Create the SWAT ResNet model
    swat_model = get_network_generator_resnet_swat()(fake_config)
    # swat_model = get_network_generator_resnet_swat()(None)
    swat_model.load_state_dict(
        generate_random_state_dict(swat_model, seed=42, sparsity=0)
    )

    # Pass the input tensor through both models
    original_output = original_model(input_tensor)
    swat_output = swat_model(input_tensor)

    # print("[test_out] ", original_output)
    # print("[test_out] ", swat_output)

    # check if all the value of the output are the same
    assert torch.allclose(
        original_output, swat_output, atol=1e-7
    ), "The output of the two models are different"

    # print("[test_out] passed successfully!")


def test_gradient() -> None:
    """Test the gradient of the SWAT ResNet model."""
    # Create a random input tensor
    input_tensor = torch.randn(1, 3, 32, 32)
    fake_config = dict([[], []])

    # Create the SWAT ResNet model
    swat_model = get_network_generator_resnet_swat()(fake_config)

    # Pass the input tensor through the model
    output = swat_model(input_tensor)

    # Create a random target tensor
    target = torch.randint(0, 10, (1,))

    # Calculate the loss
    loss = F.cross_entropy(output, target)

    # Zero the gradients
    swat_model.zero_grad()

    # Backward pass
    loss.backward()

    # print("[test_gradient] passed successfully!")


def compare_gradients() -> None:
    """Test if the gradient.

    Check if the SWAT ResNet model is the same as the original ResNet.
    """
    # Create a random input tensor
    input_tensor = torch.randn(1, 3, 32, 32)

    fake_config = dict([[], []])
    # Create the original ResNet model
    # print("[compare_gradients] Creating the original ResNet model")
    original_model = get_resnet18()(fake_config)
    # original_model = get_resnet18()(None)
    # original_model.load_state_dict(generate_random_state_dict(original_model,
    # seed=42, sparsity=0))

    # Create the SWAT ResNet model
    # print("[compare_gradients] Creating the SWAT ResNet model")
    swat_model = get_network_generator_resnet_swat()(fake_config)
    # swat_model = get_network_generator_resnet_swat()(None)
    # swat_model.load_state_dict(generate_random_state_dict(
    # swat_model, seed=42, sparsity=0))

    # Pass the input tensor through the models
    original_output = original_model(input_tensor)
    swat_output = swat_model(input_tensor)

    # Create a random target tensor
    target = torch.randint(0, 10, (1,))

    # Calculate the loss
    original_loss = F.cross_entropy(original_output, target)
    swat_loss = F.cross_entropy(swat_output, target)

    # Zero the gradients
    original_model.zero_grad()
    swat_model.zero_grad()

    # Backward pass
    original_loss.backward()
    swat_loss.backward()

    # check if all the gradient are the same
    for original_param, swat_param in zip(
        original_model.parameters(), swat_model.parameters(), strict=True
    ):
        assert torch.allclose(
            original_param.grad, swat_param.grad, atol=1e-7
        ), "The gradient of the two models are different"

    # print("[compare_gradients] passed successfully!")


def test_main() -> None:
    """Test the SWAT ResNet model."""
    # test_out()
    # test_gradient()
    # compare_gradients()


if __name__ == "__main__":
    test_main()
