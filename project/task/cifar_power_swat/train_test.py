"""MNIST training and testing functions, local and federated."""

from collections.abc import Sized
from copy import deepcopy
from pathlib import Path
from typing import cast
from collections.abc import Callable

import logging
from logging import ERROR
from flwr.common import log
from project.fed.utils.utils import (
    generic_get_parameters,
    generic_set_parameters,
    net_compare,
    print_nonzeros,
)
from project.task.cifar_power_swat.models import (
    get_parameters_to_prune,
)

import torch
from pydantic import BaseModel
from torch import nn
from torch.nn.utils import prune
from torch.utils.data import DataLoader
from project.client.client import ClientConfig

from project.task.default.train_test import get_fed_eval_fn as get_default_fed_eval_fn
from project.task.default.train_test import (
    get_on_evaluate_config_fn as get_default_on_evaluate_config_fn,
)
from project.types.common import (
    OnFitConfigFN,
)


class TrainConfig(BaseModel):
    """Training configuration, allows '.' member acces and static checking.

    Guarantees that all necessary components are present, fails early if config is
    mismatched to client.
    """

    device: torch.device
    epochs: int
    learning_rate: float
    final_learning_rate: float  # ? to remove
    curr_round: int = 0

    class Config:
        """Setting to allow any types, including library ones like torch.device."""

        arbitrary_types_allowed = True


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
        # weight_decay=0.001,
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

        print_nonzeros(net, "[train] Before training:")
        before_train_net = deepcopy(net)
        # train the network, with the current parameter
        metrics = train(
            net=net,
            trainloader=trainloader,
            _config=_config,
            _working_dir=_working_dir,
        )
        after_train_net = deepcopy(net)
        after_training_sparsity = print_nonzeros(net, "[train] After training:")
        after_training_metrics = net_compare(before_train_net, after_train_net)
        # print("!!![NET_COMPARE] After training", after_training_metrics)

        base_alpha = 1.0
        num_pruning_round = 1000
        if (
            _config["curr_round"] <= num_pruning_round or alpha == base_alpha
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
                amount=amount,
            )
            for module, name, _ in parameters_to_prune:
                prune.remove(module, name)

            del parameters_to_prune
            # torch.cuda.empty_cache()
        after_pruning_sparsity = print_nonzeros(net, "[train] After pruning:")
        after_prune_net = deepcopy(before_train_net)
        generic_set_parameters(after_prune_net, generic_get_parameters(net))
        after_pruning_metrics = net_compare(before_train_net, after_prune_net)
        # print("!!![NET_COMPARE] After pruning:", after_pruning_metrics)
        # get the amount of parameters to prune from the first module that has sparsity

        if _config["curr_round"] > 1:
            metrics[1]["after_training_activation"] = after_training_metrics[
                "activated"
            ]
            metrics[1]["after_training_deactivation"] = after_training_metrics[
                "deactivated"
            ]
            metrics[1]["after_training_sparsity"] = after_training_sparsity
            metrics[1]["after_pruning_activation"] = after_pruning_metrics["activated"]
            metrics[1]["after_pruning_deactivation"] = after_pruning_metrics[
                "deactivated"
            ]
            metrics[1]["after_pruning_sparsity"] = after_pruning_sparsity
        # print(f"The metrics are {metrics[1]}")

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

    # print("Hi Alex, I am in test function")

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

    return (
        per_sample_loss / len(cast(Sized, testloader.dataset)),
        len(cast(Sized, testloader.dataset)),
        {
            "test_accuracy": float(correct) / len(cast(Sized, testloader.dataset)),
        },
    )


def get_on_fit_config_fn(fit_config: dict) -> OnFitConfigFN:
    """Generate on_fit_config_fn based on a dict from the hydra config,.

    Parameters
    ----------
    fit_config : Dict
        The configuration for the fit function.
        Loaded dynamically from the config file.

    Returns
    -------
    Optional[OnFitConfigFN]
        The on_fit_config_fn for the server if the fit_config is not empty, else None.
    """
    # Fail early if the fit_config does not match expectations
    ClientConfig(**fit_config)

    def fit_config_fn(server_round: int) -> dict:
        """CIFAR on_fit_config_fn.

        Parameters
        ----------
        server_round : int
            The current server round.
            Passed to the client

        Returns
        -------
        Dict
            The configuration for the fit function.
            Loaded dynamically from the config file.
        """
        # resolve and convert to python dict
        fit_config["extra"]["curr_round"] = server_round  # add round info
        return fit_config

    return fit_config_fn


get_fed_eval_fn = get_default_fed_eval_fn
# get_on_fit_config_fn = get_default_on_fit_config_fn
get_on_evaluate_config_fn = get_default_on_evaluate_config_fn
