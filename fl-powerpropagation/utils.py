import numbers
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
from flwr.common import NDArrays


def compute_model_delta(
    trained_parameters: NDArrays, og_parameters: NDArrays
) -> NDArrays:
    """Compute the layer-wise difference between two lists of parameters.

    Args:
        trained_parameters (NDArrays): updated parameters
        og_parameters (NDArrays): original parameters

    Returns:
        NDArrays: difference between the two lists of parameters
    """
    return [np.subtract(x, y) for (x, y) in zip(trained_parameters, og_parameters)]


def compute_norm(update: NDArrays) -> float:
    """Compute the l1 norm of a parameter update with mismatched np array shapes, to be used in clipping"""
    flat_update = update[0]
    for i in range(1, len(update)):
        flat_update = np.append(flat_update, update[i])  # type: ignore
    summed_update = np.abs(flat_update)
    norm_sum = np.sum(summed_update)
    norm = np.sqrt(norm_sum)
    return norm


def get_device() -> str:
    """Determine which device to use for PyTorch.

    Returns:
        str: device for PyTorch
    """
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = "mps"
    return device


def aggregate_weighted_average(metrics: List[Tuple[int, Dict]]) -> Dict:
    """Generic function to combine results from multiple clients
    following training or evaluation.

    Args:
        metrics (List[Tuple[int, dict]]): collected clients metrics

    Returns:
        dict: result dictionary containing the aggregate of the metrics passed.
    """
    average_dict: dict = defaultdict(list)
    total_examples: int = 0
    for num_examples, metrics_dict in metrics:
        for key, val in metrics_dict.items():
            if isinstance(val, numbers.Number):
                average_dict[key].append((num_examples, val))  # type:ignore
        total_examples += num_examples
    return {
        key: {
            "avg": float(
                sum([num_examples * metr for num_examples, metr in val])
                / float(total_examples)
            ),
            "all": val,
        }
        for key, val in average_dict.items()
    }
