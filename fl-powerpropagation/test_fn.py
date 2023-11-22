from pathlib import Path
from typing import Callable, Dict, Tuple

import torch
from cifar10_utils import load_centralised_test_set
from flwr.common import Scalar
from torch import nn
from torch.nn import Module
from torch.utils.data import DataLoader


def test(
    net: Module,
    testloader: DataLoader,
    device: str,  # pylint: disable=no-member
) -> Tuple[float, Dict[str, Scalar]]:
    """Validate the network on the entire test set."""
    # Define loss and metrics
    criterion = nn.CrossEntropyLoss()
    correct, loss, total = 0, 0.0, 0

    # Evaluate the network
    net.to(device)
    net.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            total += len(outputs)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)  # pylint: disable=no-member
            correct += (predicted == labels).sum().item()

    return loss / total, {"accuracy": correct / total}


def centralised_evaluate(
    partition_root: Path,
    net: Module,
    device: str = "cpu",  # pylint: disable=no-member
    batch_size: int = 32,
    test_loop: Callable[
        [Module, DataLoader, str], Tuple[float, Dict[str, Scalar]]
    ] = test,
) -> Tuple[float, Dict[str, Scalar]]:
    # Load the test set
    testset, testloader, n_samples = load_centralised_test_set(
        partition_root, batch_size
    )
    # Evaluate the model
    return test_loop(net=net, testloader=testloader, device=device)
