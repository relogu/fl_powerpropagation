"""Flower client example using PyTorch for CIFAR-10 image classification."""

import argparse
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Tuple, Callable

import flwr as fl
from flwr.common.typing import Scalar
import numpy as np
import torch
from torch.optim import Optimizer, SGD
from torch.nn import Module
from torch.utils.data import DataLoader

from utils import get_device
from model import get_network_generator_cnn
from train import train
from test import test
from cifar10_utils import load_partitioned_data


# Flower Client
class CifarClient(fl.client.NumPyClient):
    """Flower client implementing CIFAR-10 image classification using
    PyTorch."""

    def __init__(
        self,
        client_id: int,
        init_model_fn: Callable[[], Module],
        device: torch.device = None,
    ) -> None:
        self.client_id = client_id
        self.init_model_fn = init_model_fn
        self.device = device if device is not None else get_device()

    def get_parameters(self, config: Dict[str, str], net: Module = None) -> List[np.ndarray]:
        if net is None:
            net = self.init_model_fn()
        net.eval()
        # Return model parameters as a list of NumPy ndarrays
        return [val.cpu().numpy() for _, val in net.state_dict().items()]

    def set_parameters(self, net: Module, parameters: List[np.ndarray]) -> None:
        if net is None:
            net = self.init_model_fn()
        # Set model parameters from a list of NumPy ndarrays
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)
        return net
    
    def _train_loop(self, net: Module, trainloader: DataLoader, optimizer: Optimizer, epochs: int, device: torch.device) -> Dict[str, Scalar]:
        return train(
            net=net, trainloader=trainloader, optimizer=optimizer, epochs=epochs, device=device
        )

    def fit(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[List[np.ndarray], int, Dict[str, Scalar]]:
        # Load the train set
        trainloader, _, num_examples = load_partitioned_data(
            partitions_root=Path(config['partitions_root']+str(self.client_id)),
            batch_size=config['batch_size'],
        )
        # Create the model
        net: Module = self.init_model_fn()
        # Set model parameters
        net = self.set_parameters(net, parameters)
        # Set the local optimizer
        optimizer = SGD(net.parameters(), lr=float(config["learning_rate"]))
        # Train the model
        metrics = self._train_loop(net=net, trainloader=trainloader, optimizer=optimizer, epochs=config['epochs'], device=self.device)
        # Get parameters
        parameters = self.get_parameters(net=net, config={})
        # Check the magnitude and the number of non-zero weigths in conv layers
        conv_params = []
        for name, val in net.state_dict().items():
            if 'conv' in name:
                conv_params.append(val.cpu().numpy())
        metrics.update({'max_weight': np.max([np.max(np.abs(a)) for a in conv_params])})
        metrics.update({'non_zero_params': np.max([np.count_nonzero(a) for a in conv_params])})
        # Return training resutls
        return (
            parameters,
            num_examples["trainset"],
            metrics,
        )

    def evaluate(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[float, int, Dict]:
        # Load the test set
        _, testloader, num_examples = load_partitioned_data(
            partitions_root=Path(config['partitions_root']+str(self.client_id)),
            batch_size=config['batch_size'],
        )
        # Create the model
        net = self.init_model_fn()
        # Set model parameters
        net = self.set_parameters(net, parameters)
        # Evaluate the model
        loss, accuracy = test(net, testloader, device=self.device)
        # Return evaluation results
        return float(loss), num_examples["testset"], {"accuracy": accuracy}


def main(args, ClientType) -> None:
    """Load data, start CifarClient."""
    # Instantiate client
    client = ClientType(args.cid, get_network_generator_cnn(), get_device())
    if not args.test:
        # Start client
        fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client)
    else:
        print("Running client in test mode")
        from cifar10_utils import CIFAR10_DATA_ROOT
        parameters = client.get_parameters({})
        fit_config = {
            "partitions_root": CIFAR10_DATA_ROOT+'/lda/100/0.50/',
            "batch_size": 32,
            "learning_rate": 0.1,
        }
        loss, n_samples, res = client.evaluate(parameters, fit_config)
        print(loss, n_samples, res)
        model, n_samples, res = client.fit(parameters, fit_config)
        print(n_samples, res)
        res = client.evaluate(model, fit_config)
        print(res)


def execute():
    if __name__ == "__main__":
        parser = argparse.ArgumentParser(
            prog="Flower Client",
            description="This client trains a CNN on a partition of CIFAR10",
        )
        parser.add_argument("--cid", type=int, default=0, help="Client ID.")
        parser.add_argument(
            "--test",
            type=bool,
            default=False,
            help="If True, run client in test mode (no training).",
        )
        args = parser.parse_args()
        main(args, CifarClient)

execute()
