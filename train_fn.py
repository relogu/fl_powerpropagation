from logging import ERROR, INFO, DEBUG
from typing import Callable, Dict

import torch
from flwr.common.logger import log
from flwr.common.typing import Scalar
from model import get_parameters_to_prune
from torch import nn
from torch.nn import Module
from torch.nn.utils import prune
from torch.optim import Optimizer
from torch.utils.data import DataLoader


def train(
    net: Module,
    trainloader: DataLoader,
    optimizer: Optimizer,
    device: str,
    epochs: int,
) -> Dict[str, Scalar]:
    """Train the network."""
    net.to(device)
    net.train()

    running_loss, correct, total = 0.0, 0, 0
    criterion = nn.CrossEntropyLoss()
    for epoch in range(1, epochs + 1):
        for data in trainloader:
            # log(DEBUG, f"Get data")
            images, labels = data[0].to(device), data[1].to(device)

            # log(DEBUG, f"Zeroing gradients")
            # zero the parameter gradients
            optimizer.zero_grad()

            # log(DEBUG, f"Forward pass")
            # forward + backward + optimize
            outputs = net(images)
            # log(DEBUG, f"Computing loss")
            loss = criterion(outputs, labels)

            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            # log(DEBUG, f"Backward pass")
            loss.backward()
            # log(DEBUG, f"Optimising")
            optimizer.step()

            # Get statistics
            running_loss += loss.item()
            total += len(labels)

    return {"loss": running_loss / total, "accuracy": correct / total}


def get_train_and_prune(
    amount: float = 0.3,
    pruning_method: str = "base",
) -> Callable[[Module, DataLoader, Optimizer, str, int], Dict[str, Scalar]]:
    """Return the training loop with one step pruning at the end."""
    if pruning_method == "base":
        pruning_method = prune.BasePruningMethod
    elif pruning_method == "l1":
        pruning_method = prune.L1Unstructured
    else:
        log(ERROR, f"Pruning method {pruning_method} not recognised, using base")

    def train_and_prune(
        self,  # Necessary since the implementation of the `main.py`
        net: Module,
        trainloader: DataLoader,
        optimizer: Optimizer,
        device: str,
        epochs: int,
        **kwargs,
    ) -> Dict[str, Scalar]:
        parameters_to_prune = get_parameters_to_prune(net)
        metrics = train(
            net=net,
            trainloader=trainloader,
            optimizer=optimizer,
            device=device,
            epochs=epochs,
        )
        prune.global_unstructured(
            parameters=[(module, tensor_name) for module, tensor_name, _ in parameters_to_prune],
            pruning_method=pruning_method,
            amount=amount,
        )
        for module, name in parameters_to_prune:
            prune.remove(module, name)
        return metrics

    return train_and_prune


def get_iterative_train_and_prune(
    amount: float = 0.3,
    pruning_method: str = "base",
) -> Callable[[Module, DataLoader, Optimizer, str, int], Dict[str, Scalar]]:
    """Return the training loop with one step pruning after every epoch."""
    if pruning_method == "base":
        pruning_method = prune.BasePruningMethod
    elif pruning_method == "l1":
        pruning_method = prune.L1Unstructured
    else:
        log(ERROR, f"Pruning method {pruning_method} not recognised, using base")

    def iterative_train_and_prune(
        self,  # Necessary since the implementation of the `main.py`
        net: Module,
        trainloader: DataLoader,
        optimizer: Optimizer,
        device: str,
        epochs: int,
        **kwargs,
    ) -> Dict[str, Scalar]:
        parameters_to_prune = get_parameters_to_prune(net)
        per_epoch_amount = amount / epochs
        for _ in range(epochs):
            metrics = train(
                net=net,
                trainloader=trainloader,
                optimizer=optimizer,
                device=device,
                epochs=1,
            )
            prune.global_unstructured(
                parameters=[(module, tensor_name) for module, tensor_name, _ in parameters_to_prune],
                pruning_method=pruning_method,
                amount=per_epoch_amount,
            )
            for module, name in parameters_to_prune:
                prune.remove(module, name)
        return metrics

    return iterative_train_and_prune
