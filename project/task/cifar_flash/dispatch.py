"""Dispatch the MNIST functionality to project.main.

The dispatch functions are used to
dynamically select the correct functions from the task
based on the hydra config file.
The following categories of functionality are grouped together:
    - train/test and fed test functions
    - net generator and dataloader generator functions
    - fit/eval config functions

The top-level project.dipatch module operates as a pipeline
and selects the first function which does not return None.

Do not throw any errors based on not finding a given attribute
in the configs under any circumstances.

If you cannot match the config file,
return None and the dispatch of the next task
in the chain specified by project.dispatch will be used.
"""

from pathlib import Path

from omegaconf import DictConfig
from project.task.cifar_flash.models import get_network_generator_resnet_powerprop

from project.task.default.dispatch import dispatch_config as dispatch_default_config
from project.task.cifar_flash.dataset import (
    get_dataloader_generators as new_generators,
)
from project.task.cifar_flash.train_test import (
    get_fed_eval_fn,
    get_train_and_prune,
    test,
)
from project.types.common import DataStructure, TrainStructure


def dispatch_train(
    cfg: DictConfig,
) -> TrainStructure | None:
    """Dispatch the train/test and fed test functions based on the config file.

    Do not throw any errors based on not finding a given attribute
    in the configs under any circumstances.

    If you cannot match the config file,
    return None and the dispatch of the next task
    in the chain specified by project.dispatch will be used.

    Parameters
    ----------
    cfg : DictConfig
        The configuration for the train function.
        Loaded dynamically from the config file.

    Returns
    -------
    Optional[TrainStructure]
        The train function, test function and the get_fed_eval_fn function.
        Return None if you cannot match the cfg.
    """
    # Select the value for the key with None default
    train_structure: str | None = cfg.get("task", {}).get(
        "train_structure",
        None,
    )

    # Only consider not None and uppercase matches
    if train_structure is not None and train_structure.upper() == "CIFAR_FLASH_PRUNE":
        sparsity = cfg.get("task", {}).get("sparsity", 0.0)
        alpha = cfg.get("task", {}).get("alpha", 1.0)
        return (
            get_train_and_prune(alpha=alpha, amount=sparsity, pruning_method="l1"),
            test,
            get_fed_eval_fn,
        )
    # Cannot match, send to next dispatch in chain
    return None


def dispatch_data(cfg: DictConfig) -> DataStructure | None:
    """Dispatch the train/test and fed test functions based on the config file.

    Do not throw any errors based on not finding a given attribute
    in the configs under any circumstances.

    If you cannot match the config file,
    return None and the dispatch of the next task
    in the chain specified by project.dispatch will be used.

    Parameters
    ----------
    cfg : DictConfig
        The configuration for the data functions.
        Loaded dynamically from the config file.

    Returns
    -------
    Optional[DataStructure]
        The net generator, client dataloader generator and fed dataloader generator.
        Return None if you cannot match the cfg.
    """
    # Select the value for the key with {} default at nested dicts
    # and None default at the final key
    client_model_and_data: str | None = cfg.get(
        "task",
        {},
    ).get("model_and_data", None)

    # Select the partition dir
    # if it does not exist data cannot be loaded
    # for MNIST and the dispatch should return None
    partition_dir: str | None = cfg.get("dataset", {}).get(
        "partition_dir",
        None,
    )

    # Only consider situations where both are not None
    # otherwise data loading would failr later
    if client_model_and_data is not None and partition_dir is not None:
        # Obtain the dataloader generators
        # for the provided partition dir
        alpha = cfg.get("task", {}).get("alpha", 1.0)
        sparsity = cfg.get("task", {}).get("sparsity", 0.0)
        num_classes: int = cfg.get("dataset", {}).get(
            "num_classes",
            10,
        )

        # Case insensitive matches
        if client_model_and_data.upper() == "RESNET_FLASH":
            (
                client_dataloader_gen,
                fed_dataloater_gen,
            ) = new_generators(
                Path(partition_dir),
            )
            return (
                get_network_generator_resnet_powerprop(
                    alpha=alpha, sparsity=sparsity, num_classes=num_classes
                ),
                client_dataloader_gen,
                fed_dataloater_gen,
            )

    # Cannot match, send to next dispatch in chain
    return None


dispatch_config = dispatch_default_config
