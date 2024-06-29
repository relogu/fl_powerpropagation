"""MNIST training and testing functions, local and federated."""

from collections.abc import Callable, Sized
from pathlib import Path
import time
from typing import cast

import logging
from logging import ERROR
from flwr.common import log
from project.task.cifar_powerprop.models import (
    get_parameters_to_prune,
)


import torch
from pydantic import BaseModel
from torch import nn
from torch.nn.utils import prune
from torch.utils.data import DataLoader
import torch.nn.functional as F

from project.task.default.train_test import get_fed_eval_fn as get_default_fed_eval_fn
from project.task.default.train_test import (
    get_on_evaluate_config_fn as get_default_on_evaluate_config_fn,
)
from project.task.default.train_test import (
    get_on_fit_config_fn as get_default_on_fit_config_fn,
)

from project.task.utils.powerprop_modules import PowerPropConv2D, PowerPropLinear


class TrainConfig(BaseModel):
    """Training configuration, allows '.' member acces and static checking.

    Guarantees that all necessary components are present, fails early if config is
    mismatched to client.
    """

    device: torch.device
    epochs: int
    learning_rate: float
    # final_learning_rate: float  # ? to remove
    curr_round: int
    cid: int

    class Config:
        """Setting to allow any types, including library ones like torch.device."""

        arbitrary_types_allowed = True


def spectral_norm(
    weight: torch.Tensor, num_iterations: int = 1, epsilon: float = 1e-12
) -> torch.Tensor:
    """Spectral Normalization with sign handling and stability check."""
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
    exponent = 1 + weight_normalized.view_as(weight)
    exponent = torch.clamp(exponent, max=10)  # Clamp to prevent overflow

    weight_updated = sign_weight * torch.pow(weight_abs, exponent)

    return weight_updated, exponent


def compute_average_exponent(model: nn.Module) -> float:
    """Compute the average exponent across all layers in the model."""
    total_exponent_sum = 0.0
    total_non_zero_weights = 0

    for layer in model.modules():
        if isinstance(
            layer,
            PowerPropLinear | PowerPropConv2D,
        ):
            weight = layer.weight.data
            weight = weight[weight != 0]  # Filter out zero weights
            if weight.numel() > 0:
                _, exponent = spectral_norm(weight)
                average_exponent = torch.mean(exponent)
                total_exponent_sum += average_exponent.item()  # * weight.numel()
                total_non_zero_weights += 1  # weight.numel()

    if total_non_zero_weights > 0:
        return total_exponent_sum / total_non_zero_weights
    else:
        return 0.0


def train(  # pylint: disable=too-many-arguments
    net: nn.Module,
    trainloader: DataLoader,
    _config: dict,
    _working_dir: Path,
) -> tuple[int, dict]:
    """Train the network on the training set.

    Parameters
    ----------
    net : nn.Module
        The neural network to train.
    trainloader : DataLoader
        The DataLoader containing the data to train the network on.
    _config : Dict
        The configuration for the training.
        Contains the device, number of epochs and learning rate.
        Static type checking is done by the TrainConfig class.

    Returns
    -------
    Tuple[int, Dict]
        The number of samples used for training,
        the loss, and the accuracy of the input model on the given data.
    """
    if len(cast(Sized, trainloader.dataset)) == 0:
        raise ValueError(
            "Trainloader can't be 0, exiting...",
        )

    config: TrainConfig = TrainConfig(**_config)
    del _config

    net.to(config.device)
    net.train()

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(
        net.parameters(),
        lr=config.learning_rate,
        weight_decay=0.001,
    )

    final_epoch_per_sample_loss = 0.0
    num_correct = 0
    for _ in range(config.epochs):
        final_epoch_per_sample_loss = 0.0
        num_correct = 0
        for data, target in trainloader:
            data, target = (
                data.to(
                    config.device,
                ),
                target.to(config.device),
            )
            optimizer.zero_grad()
            output = net(data)
            loss = criterion(output, target)
            final_epoch_per_sample_loss += loss.item()
            num_correct += (output.max(1)[1] == target).clone().detach().sum().item()
            loss.backward()
            optimizer.step()

    return len(cast(Sized, trainloader.dataset)), {
        "train_loss": final_epoch_per_sample_loss / len(
            cast(Sized, trainloader.dataset)
        ),
        "train_accuracy": float(num_correct) / len(cast(Sized, trainloader.dataset)),
    }


# def mixed_precision_train(
#     net: nn.Module,
#     trainloader: DataLoader,
#     _config: dict,
#     _working_dir: Path,
# ) -> tuple[int, dict]:
#     """Train the network on the training set.

#     Parameters
#     ----------
#     net : nn.Module
#         The neural network to train.
#     trainloader : DataLoader
#         The DataLoader containing the data to train the network on.
#     _config : dict
#         The configuration for the training.
#         Contains the device, number of epochs and learning rate.

#     Returns
#     -------
#     Tuple[int, dict]
#         The number of samples used for training,
#         the loss, and the accuracy of the input model on the given data.
#     """
#     if len(trainloader.dataset) == 0:
#         raise ValueError("Trainloader cannot be empty.")

#     config: TrainConfig = TrainConfig(**_config)
#     del _config

#     net.to(config.device)
#     net.train()

#     criterion = nn.CrossEntropyLoss()

#     optimizer = torch.optim.SGD(
#         net.parameters(),
#         lr=config.learning_rate,
#         weight_decay=0.001,
#     )

#     scaler = GradScaler()

#     final_epoch_per_sample_loss = 0.0
#     num_correct = 0

#     for _ in range(config.epochs):
#         final_epoch_per_sample_loss = 0.0
#         num_correct = 0

#         for data, target in trainloader:
#             data, target = data.to(config.device), target.to(config.device)

#             optimizer.zero_grad()

#             with autocast():
#                 output = net(data)
#                 loss = criterion(output, target)

#             final_epoch_per_sample_loss += loss.item()

#             scaler.scale(loss).backward()
#             scaler.step(optimizer)
#             scaler.update()

#             num_correct += (output.argmax(1) == target).sum().item()

#         # Calculate epoch-wise metrics
#         train_loss = final_epoch_per_sample_loss / len(trainloader.dataset)
#         train_accuracy = num_correct / len(trainloader.dataset)

#         # print(
#         #     f"Epoch {epoch + 1}: Train Loss {train_loss:.4f}, Train Accuracy"
#         #     f" {train_accuracy:.4f}"
#         # )

#     return len(trainloader.dataset), {
#         "train_loss": train_loss,
#         "train_accuracy": train_accuracy,
#     }


def get_train_and_prune(
    alpha: float = 1.0, amount: float = 0.0, pruning_method: str = "l1"
) -> Callable[[nn.Module, DataLoader, dict, Path], tuple[int, dict]]:
    """Return the training loop with one step pruning at the end.

    Think about moving 'amount' to the config file
    """
    if pruning_method == "base":  # ? not working
        pruning_method = prune.BasePruningMethod
    elif pruning_method == "l1":
        pruning_method = prune.L1Unstructured
    else:
        log(ERROR, f"Pruning method {pruning_method} not recognised, using base")

    def train_and_prune(
        net: nn.Module,
        trainloader: DataLoader,
        _config: dict,
        _working_dir: Path,
    ) -> tuple[int, dict]:
        """Training and pruning process."""
        log(logging.DEBUG, "Start training")

        average_exp = compute_average_exponent(net)

        # train the network, with the current parameter
        metrics = train(
            # metrics = mixed_precision_train(
            net=net,
            trainloader=trainloader,
            _config=_config,
            _working_dir=_working_dir,
        )
        base_alpha = 1.0
        num_pruning_round = 1000
        # num_scales = 5
        # sparsity_range = 1 - amount
        # sparsity_inc = sparsity_range * int(_config["cid"]) / num_scales
        sparsity_inc = 0

        # print(f"[client_{_config['cid']}]sparsity_inc: {sparsity_inc}")

        if (
            _config["curr_round"] < num_pruning_round or alpha == base_alpha
        ) and amount > 0:
            """
            The net must be pruned:
            - at the first round if we are using powerprop
            - every round if we are not using powerprop (alpha=1.0)
            """
            parameters_to_prune = get_parameters_to_prune(net)

            prune.global_unstructured(
                parameters=[
                    (module, tensor_name)
                    for module, tensor_name, _ in parameters_to_prune
                ],
                pruning_method=pruning_method,
                amount=amount + sparsity_inc,
            )
            for module, name, _ in parameters_to_prune:
                prune.remove(module, name)

            # del parameters_to_prune
            torch.cuda.empty_cache()

        # get the amount of parameters to prune from the first module that has sparsity
        metrics[1]["exponet"] = average_exp
        return metrics

    return train_and_prune


class TestConfig(BaseModel):
    """Testing configuration, allows '.' member acces and static checking.

    Guarantees that all necessary components are present, fails early if config is
    mismatched to client.
    """

    device: torch.device

    class Config:
        """Setting to allow any types, including library ones like torch.device."""

        arbitrary_types_allowed = True


def test(
    net: nn.Module,
    testloader: DataLoader,
    _config: dict,
    _working_dir: Path,
) -> tuple[float, int, dict]:
    """Evaluate the network on the test set.

    Parameters
    ----------
    net : nn.Module
        The neural network to test.
    testloader : DataLoader
        The DataLoader containing the data to test the network on.
    config : Dict
        The configuration for the testing.
        Contains the device.
        Static type checking is done by the TestConfig class.

    Returns
    -------
    Tuple[float, int, float]
        The loss, number of test samples,
        and the accuracy of the input model on the given data.
    """
    if len(cast(Sized, testloader.dataset)) == 0:
        raise ValueError(
            "Testloader can't be 0, exiting...",
        )

    config: TestConfig = TestConfig(**_config)
    del _config

    net.to(config.device)
    net.eval()

    criterion = nn.CrossEntropyLoss()
    correct, per_sample_loss = 0, 0.0

    start_time = time.time()
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = (
                images.to(
                    config.device,
                ),
                labels.to(config.device),
            )
            outputs = net(images)
            per_sample_loss += criterion(
                outputs,
                labels,
            ).item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
    elapsed_time = time.time() - start_time
    # print(f"Elapsed time for testing: {elapsed_time}")

    torch.cuda.empty_cache()

    return (
        per_sample_loss / len(cast(Sized, testloader.dataset)),
        len(cast(Sized, testloader.dataset)),
        {
            "test_accuracy": float(correct) / len(cast(Sized, testloader.dataset)),
            "test_time": elapsed_time,
        },
    )


# Use defaults as they are completely determined
# by the other functions defined in mnist_classification
get_fed_eval_fn = get_default_fed_eval_fn
get_on_fit_config_fn = get_default_on_fit_config_fn
get_on_evaluate_config_fn = get_default_on_evaluate_config_fn
