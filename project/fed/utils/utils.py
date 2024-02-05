"""FL-related utility functions for the project."""

import logging
import struct
from collections import OrderedDict, defaultdict
from collections.abc import Callable
from pathlib import Path

import torch.nn.functional as F

import numpy as np

import torch
from flwr.common import (
    NDArrays,
    Parameters,
    log,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from torch import nn

from project.types.common import ClientGen, NetGen, OnEvaluateConfigFN, OnFitConfigFN


def generic_set_parameters(
    net: nn.Module,
    parameters: NDArrays,
    to_copy: bool = True,
) -> None:
    """Set the parameters of a network.

    Parameters
    ----------
    net : nn.Module
        The network whose parameters should be set.
    parameters : NDArrays
        The parameters to set.
    to_copy : bool (default=False)
        Whether to copy the parameters or use them directly.

    Returns
    -------
    None
    """
    sorted_dict = sorted(net.state_dict().items(), key=lambda x: x[0])  # Sort by keys

    params_dict = zip(
        (keys for keys, _ in sorted_dict),
        parameters,
        strict=False,
    )
    state_dict = OrderedDict(
        # {k: torch.Tensor(v if not to_copy else v.copy()) for k, v in params_dict},
        # The commented version raise an error: !?
        # IndexError: index 0 is out of bounds for dimension 0 with size 0 ?
        {k: torch.tensor(v if not to_copy else v.copy()) for k, v in params_dict},
    )

    net.load_state_dict(state_dict)


def generic_get_parameters(net: nn.Module) -> NDArrays:
    """Implement generic `get_parameters` for Flower Client.

    Parameters
    ----------
    net : nn.Module
        The network whose parameters should be returned.

    Returns
    -------
        NDArrays
        The parameters of the network.
    """
    state_dict_items = sorted(
        net.state_dict().items(), key=lambda x: x[0]
    )  # Sort by keys
    parameters = [val.cpu().numpy() for _, val in state_dict_items]

    return parameters


def load_parameters_from_file(path: Path) -> Parameters:
    """Load parameters from a binary file.

    Parameters
    ----------
    path : Path
        The path to the parameters file.

    Returns
    -------
    'Parameters
        The parameters.
    """
    byte_data = []
    if path.suffix == ".bin":
        with open(path, "rb") as f:
            while True:
                # Read the length (4 bytes)
                length_bytes = f.read(4)
                if not length_bytes:
                    break  # End of file
                length = struct.unpack("I", length_bytes)[0]

                # Read the data of the specified length
                data = f.read(length)
                byte_data.append(data)

        return Parameters(
            tensors=byte_data,
            tensor_type="numpy.ndarray",
        )

    raise ValueError(f"Unknown parameter format: {path}")


def get_initial_parameters(
    net_generator: NetGen,
    config: dict,
    load_from: Path | None,
    server_round: int | None,
) -> Parameters:
    """Get the initial parameters for the network.

    Parameters
    ----------
    net_generator : NetGen
        The function to generate the network.
    config : Dict
        The configuration.
    load_from : Optional[Path]
        The path to the parameters file.

    Returns
    -------
    'Parameters
        The parameters.
    """
    if load_from is None:
        log(
            logging.INFO,
            "Generating initial parameters with config: %s",
            config,
        )
        return ndarrays_to_parameters(
            generic_get_parameters(net_generator(config)),
        )
    try:
        if server_round is not None:
            # Load specific round parameters
            load_from = load_from / f"parameters_{server_round}.bin"
        else:
            # Load only the most recent parameters
            load_from = max(
                Path(load_from).glob("parameters_*.bin"),
                key=lambda f: (
                    int(f.stem.split("_")[1]),
                    int(f.stem.split("_")[2]),
                ),
            )

        log(
            logging.INFO,
            "Loading initial parameters from: %s",
            load_from,
        )

        return load_parameters_from_file(load_from)
    except (
        ValueError,
        FileNotFoundError,
        PermissionError,
        OSError,
        EOFError,
        IsADirectoryError,
    ):
        log(
            logging.INFO,
            f"Loading parameters failed from: {load_from}",
        )
        log(
            logging.INFO,
            "Generating initial parameters with config: %s",
            config,
        )

        return ndarrays_to_parameters(
            generic_get_parameters(net_generator(config)),
        )


def get_save_parameters_to_file(
    working_dir: Path,
) -> Callable[[Parameters], None]:
    """Get a function to save parameters to a file.

    Parameters
    ----------
    working_dir : Path
        The working directory.

    Returns
    -------
    Callable[[Parameters], None]
        A function to save parameters to a file.
    """

    def save_parameters_to_file(
        parameters: Parameters,
    ) -> None:
        """Save the parameters to a file.

        Parameters
        ----------
        parameters : Parameters
            The parameters to save.

        Returns
        -------
        None
        """
        parameters_path = working_dir / "parameters"
        parameters_path.mkdir(parents=True, exist_ok=True)
        with open(
            parameters_path / "parameters.bin",
            "wb",
        ) as f:
            # Since Parameters is a list of bytes
            # save the length of each row and the data
            # for deserialization
            for data in parameters.tensors:
                # Prepend the length of the data as a 4-byte integer
                f.write(struct.pack("I", len(data)))
                f.write(data)

    return save_parameters_to_file


def get_weighted_avg_metrics_agg_fn(
    to_agg: set[str],
) -> Callable[[list[tuple[int, dict]]], dict]:
    """Return a function to compute a weighted average over pre-defined metrics.

    Parameters
    ----------
    to_agg : Set[str]
        The metrics to aggregate.

    Returns
    -------
    Callable[[List[Tuple[int, Dict]]], Dict]
        A function to compute a weighted average over pre-defined metrics.
    """

    def weighted_avg(
        metrics: list[tuple[int, dict]],
    ) -> dict:
        """Compute a weighted average over pre-defined metrics.

        Parameters
        ----------
        metrics : List[Tuple[int, Dict]]
            The metrics to aggregate.

        Returns
        -------
        Dict
            The weighted average over pre-defined metrics.
        """
        total_num_examples = sum(
            [num_examples for num_examples, _ in metrics],
        )
        weighted_metrics: dict = defaultdict(float)
        for num_examples, metric in metrics:
            for key, value in metric.items():
                if key in to_agg:
                    weighted_metrics[key] += num_examples * value

        return {
            key: value / total_num_examples for key, value in weighted_metrics.items()
        }

    return weighted_avg


def test_client(
    test_all_clients: bool,
    test_one_client: bool,
    client_generator: ClientGen,
    initial_parameters: Parameters,
    total_clients: int,
    on_fit_config_fn: OnFitConfigFN | None,
    on_evaluate_config_fn: OnEvaluateConfigFN | None,
) -> None:
    """Debug the client code.

    Avoids the complexity of Ray.
    """
    parameters = parameters_to_ndarrays(initial_parameters)
    if test_all_clients or test_one_client:
        if test_one_client:
            client = client_generator(0)
            _, *res_fit = client.fit(
                parameters,
                on_fit_config_fn(0) if on_fit_config_fn else {},
            )
            res_eval = client.evaluate(
                parameters,
                on_evaluate_config_fn(0) if on_evaluate_config_fn else {},
            )
            log(
                logging.INFO,
                "Fit debug fit: %s  and eval: %s",
                res_fit,
                res_eval,
            )
        else:
            for i in range(total_clients):
                client = client_generator(i)
                _, *res_fit = client.fit(
                    parameters,
                    on_fit_config_fn(i) if on_fit_config_fn else {},
                )
                res_eval = client.evaluate(
                    parameters,
                    on_evaluate_config_fn(i) if on_evaluate_config_fn else {},
                )
                log(
                    logging.INFO,
                    "Fit debug fit: %s  and eval: %s",
                    res_fit,
                    res_eval,
                )


def net_compare(net1: nn.Module, net2: nn.Module, msg: str = "") -> float:
    """Count the rate of different parameter between two network."""
    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu",
    )

    net1.to(device)
    net2.to(device)

    state_dict1 = [1 if param != 0 else 0 for param in net1.state_dict()]
    state_dict2 = [1 if param != 0 else 0 for param in net2.state_dict()]

    # for k in state_dict1.keys():
    for k in state_dict1:
        state_dict1[k] = torch.sub(state_dict1[k], state_dict2[k], alpha=1)

    nz_count = 0
    total_params = 0
    # for k in state_dict1.keys():
    for k in state_dict1:
        nz, param = nonzeros_tensor(state_dict1[k])
        nz_count += nz
        total_params += param

    log(
        logging.INFO,
        f"[{msg}] Modified: {nz_count:7}, Equal: {total_params - nz_count:7}, total:"
        f" {total_params:7},"
        f" ({(nz_count / total_params) * 100:6.2f}% Modified)",
    )
    return round((nz_count / total_params) * 100, 1)


def nonzeros_tensor(p: torch.tensor) -> tuple[int, int]:
    """Count the rate of non-zero parameter in a tensor."""
    tensor = p.data.cpu().numpy()
    nz_count = np.count_nonzero(tensor)
    total_params = np.prod(tensor.shape)
    return nz_count, total_params


def print_nonzeros_tensor(p: torch.tensor, msg: str = "") -> float:
    """Print the count the rate of non-zero parameter in a tensor."""
    nz_count, total_params = nonzeros_tensor(p)
    # log(
    #     logging.INFO,
    #     f"{msg}       nonzeros ="
    #     f" {nz_count:7}/{total_params:7} ({100 * nz_count / total_params:6.2f}%) |"
    #     f" total_pruned = {total_params - nz_count:7} | shape = {p.shape}",
    # )
    return round((nz_count / total_params) * 100, 1)


def print_nonzeros(model: nn.Module, msg: str = "") -> float:
    """Print the rate of non-zero parameter in a model."""
    nonzero = total = 0
    for _, p in model.named_parameters():
        tensor = p.data.cpu().numpy()
        nz_count = np.count_nonzero(tensor)
        total_params = np.prod(tensor.shape)
        nonzero += nz_count
        total += total_params
    log(
        logging.INFO,
        f"{msg}   alive: {nonzero}, pruned : {total - nonzero}, total: {total},"
        f" ({100 * (total - nonzero) / total:6.2f}% pruned)",
    )
    return round(((total - nonzero) / total) * 100, 1)


def print_nonzeros_grad(model: nn.Module, msg: str = "") -> float:
    """Count the rate of non-zero parameter in a model."""
    nonzero = total = 0
    for _, p in model.named_parameters():
        tensor = p.grad.cpu().numpy()
        nz_count = np.count_nonzero(tensor)
        total_params = np.prod(tensor.shape)
        nonzero += nz_count
        total += total_params
    log(
        logging.INFO,
        f"{msg}   alive: {nonzero}, pruned : {total - nonzero}, total: {total},"
        f" Compression rate : {total / nonzero:10.2f}x "
        f" ({100 * (total - nonzero) / total:6.2f}% pruned)",
    )
    return round((nonzero / total) * 100, 1)


def generate_random_state_dict(
    model: nn.Module, seed: int = 42, sparsity: float = 0.0
) -> OrderedDict:
    """Generate a random, eventually sparse, state dict for a model."""
    # Set seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    random_state_dict = OrderedDict()

    # Create random tensors matching the shapes of the model parameters
    for key, data in model.state_dict().items():
        random_tensor = torch.randn(
            data.shape
        )  # Create a random tensor with the same shape
        random_tensor = F.dropout(random_tensor, sparsity)
        random_state_dict[key] = random_tensor

    return random_state_dict
