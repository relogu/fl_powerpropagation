"""Taken from the old code, so could give some problems.

Problems:
-   Self in most of the function. Not sure for the type to use.
    Since is not used I'm putting :int.
"""

from logging import ERROR
from collections.abc import Callable


import logging
from flwr.common import log

from collections.abc import Sized
from pathlib import Path
from typing import cast

import torch
from pydantic import BaseModel
from project.task.cifar.models import get_parameters_to_prune
from torch import nn
from torch.nn import Module
from torch.nn.utils import prune
from torch.utils.data import DataLoader


from flwr.common import NDArrays

from project.client.client import ClientConfig
from project.fed.utils.utils import generic_set_parameters
from project.types.common import (
    FedDataloaderGen,
    FedEvalFN,
    NetGen,
    TestFunc,
)
from project.utils.utils import obtain_device


class TrainConfig(BaseModel):
    """Training configuration, allows '.' member acces and static checking.

    Guarantees that all necessary components are present, fails early if config is
    mismatched to client.
    """

    device: torch.device
    epochs: int
    learning_rate: float

    class Config:
        """Setting to allow any types, including library ones like torch.device."""

        arbitrary_types_allowed = True


def train(
    # self: None,  #? Necessary since the implementation of the `main.py`
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
        weight_decay=0.001,  # ?
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


def train_one_epoch(
    # self: None,  #? Necessary since the implementation of the `main.py`
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
        weight_decay=0.001,  # ?
    )

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
        final_loss = loss  # ? don't think is necessary
        (output.max(1)[1] == target).clone().detach().sum().item()
        loss.backward()
        optimizer.step()

    return 1, {
        "train_loss": final_loss,
        "train_accuracy": (output.max(1)[1] == target).clone().detach().sum().item(),
    }


def get_train_with_hooks() -> (
    Callable[[Module, DataLoader, dict, Path], tuple[int, dict]]
):
    """Training with hooks."""

    def train_with_hooks(
        net: Module,
        trainloader: DataLoader,
        _config: dict,
        _working_dir: Path,
    ) -> tuple[int, dict]:
        """Training with hooks.

        It actually does a simple training. Since, all the code was commented.
        """
        # ?
        # Big chunk of commented code removed

        metrics = train(
            net=net,
            trainloader=trainloader,
            _config=_config,
            _working_dir=_working_dir,
        )

        # ?
        # Big chunk of commented code removed

        return metrics

    return train_with_hooks


def get_train_and_prune(
    amount: float = 0.3, pruning_method: str = "base"
) -> Callable[[Module, DataLoader, dict, Path], tuple[int, dict]]:
    """Return the training loop with one step pruning at the end.

    Think about moving 'amount' to the config file
    """
    if pruning_method == "base":
        pruning_method = prune.BasePruningMethod
    elif pruning_method == "l1":
        pruning_method = prune.L1Unstructured
    else:
        log(ERROR, f"Pruning method {pruning_method} not recognised, using base")

    def train_and_prune(
        net: Module,
        trainloader: DataLoader,
        _config: dict,
        _working_dir: Path,
    ) -> tuple[int, dict]:
        """Training and pruning process."""
        parameters_to_prune = get_parameters_to_prune(net)

        log(logging.DEBUG, "Start training")
        metrics = train(
            net=net,
            trainloader=trainloader,
            _config=_config,
            _working_dir=_working_dir,
        )
        """Parameters = [ (module, tensor_name) for module, tensor_name, _ in
        parameters_to_prune ] #?"""

        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=pruning_method,
            amount=amount,
        )

        for module, name, _ in parameters_to_prune:
            prune.remove(module, name)

        return metrics

    return train_and_prune


def get_iterative_train_and_prune(
    amount: float = 0.3,
    pruning_method: str = "base",
) -> Callable[[Module, DataLoader, dict, Path], tuple[int, dict]]:
    """Return the training loop with one step pruning after every epoch."""
    # ? Must be moved to hydra?
    if pruning_method == "base":
        pruning_method = prune.BasePruningMethod
    elif pruning_method == "l1":
        pruning_method = prune.L1Unstructured
    else:
        log(ERROR, f"Pruning method {pruning_method} not recognised, using base")

    def iterative_train_and_prune(
        net: nn.Module,
        trainloader: DataLoader,
        _config: dict,
        _working_dir: Path,
    ) -> tuple[int, dict]:
        """Training and pruning Iterative."""
        # ? How can I call the training with one epoch?

        config: TrainConfig = TrainConfig(**_config)
        # del _config

        parameters_to_prune = get_parameters_to_prune(net)
        per_epoch_amount = amount / config.epochs
        for _ in range(config.epochs):
            metrics = train_one_epoch(
                net=net,
                trainloader=trainloader,
                _config=_config,
                _working_dir=_working_dir,
            )

            prune.global_unstructured(
                parameters_to_prune,
                pruning_method=pruning_method,
                amount=per_epoch_amount,
            )
            for module, name, _ in parameters_to_prune:
                prune.remove(module, name)

        return metrics

    return iterative_train_and_prune


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
    net: Module,
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

    per_sample_loss /= len(cast(Sized, testloader.dataset))
    return (
        per_sample_loss / len(cast(Sized, testloader.dataset)),
        len(cast(Sized, testloader.dataset)),
        {
            "test_accuracy": float(correct) / len(cast(Sized, testloader.dataset)),
        },
    )


def get_fed_eval_fn(
    net_generator: NetGen,
    fed_dataloater_generator: FedDataloaderGen,
    test_func: TestFunc,
    _config: dict,
    working_dir: Path,
) -> FedEvalFN | None:
    """Get the federated evaluation function.

    Parameters
    ----------
    net_generator : NetGenerator
        The function to generate the network.
    testloader : DataLoader
        The DataLoader containing the data to test the network on.

    Returns
    -------
    Optional[FedEvalFN]
        The evaluation function for the server
        if the testloader is not empty, else None.
    """
    config: ClientConfig = ClientConfig(**_config)
    del _config

    testloader = fed_dataloater_generator(
        True,
        config.dataloader_config,
    )

    def fed_eval_fn(
        _server_round: int,
        parameters: NDArrays,
        fake_config: dict,
    ) -> tuple[float, dict] | None:
        """Evaluate the model on the given data.

        Parameters
        ----------
        server_round : int
            The current server round.
        parameters : NDArrays
            The parameters of the model to evaluate.
        _config : Dict
            The configuration for the evaluation.

        Returns
        -------
        Optional[Tuple[float, Dict]]
            The loss and the accuracy of the input model on the given data.
        """
        net = net_generator(config.net_config)
        generic_set_parameters(net, parameters)
        config.run_config["device"] = obtain_device()

        if len(cast(Sized, testloader.dataset)) == 0:
            return None

        loss, _num_samples, metrics = test_func(
            net,
            testloader,
            config.run_config,
            working_dir,
        )
        return loss, metrics

    return fed_eval_fn
