"""Dispatches the functionality of the task.

This gives us the ability to dynamically choose functionality based on the hydra dict
config without losing static type checking.
"""

from collections.abc import Callable

from omegaconf import DictConfig

from sparsyfed.task.default.dispatch import dispatch_config as dispatch_default_config


from sparsyfed.task.cifar_resnet18.dispatch import (
    dispatch_config as dispatch_resnet18_config,
)
from sparsyfed.task.cifar_resnet18.dispatch import (
    dispatch_data as dispatch_resnet18_data,
)
from sparsyfed.task.cifar_resnet18.dispatch import (
    dispatch_train as dispatch_resnet18_train,
)


# Speech command dispatch
from sparsyfed.task.speech_resnet18.dispatch import (
    dispatch_config as dispatch_speech_resnet18_config,
)
from sparsyfed.task.speech_resnet18.dispatch import (
    dispatch_data as dispatch_speech_resnet18_data,
)
from sparsyfed.task.speech_resnet18.dispatch import (
    dispatch_train as dispatch_speech_resnet18_train,
)

# ViT dispatch


from sparsyfed.types.common import ConfigStructure, DataStructure, TrainStructure


def dispatch_train(cfg: DictConfig) -> TrainStructure:
    """Dispatch the train/test and fed test functions based on the config file.

    Functionality should be added to the dispatch.py file in the task folder.
    Statically specify the new dispatch function in the list,
    function order determines precedence if two different tasks may match the config.

    Parameters
    ----------
    cfg : DictConfig
        The configuration for the train function.
        Loaded dynamically from the config file.

    Returns
    -------
    TrainStructure
        The train function, test function and the get_fed_eval_fn function.
    """
    # Create the list of task dispatches to try
    task_train_functions: list[Callable[[DictConfig], TrainStructure | None]] = [
        dispatch_resnet18_train,
        dispatch_speech_resnet18_train,
    ]

    # Match the first function which does not return None
    for task in task_train_functions:
        result = task(cfg)
        if result is not None:
            return result

    raise ValueError(
        f"Unable to match the train/test and fed_test functions: {cfg}",
    )


def dispatch_data(cfg: DictConfig) -> DataStructure:
    """Dispatch the net generator and dataloader client/fed generator functions.

    Functionality should be added to the dispatch.py file in the task folder.
    Statically specify the new dispatch function in the list,
    function order determines precedence if two different tasks may match the config.

    Parameters
    ----------
    cfg : DictConfig
        The configuration for the data function.
        Loaded dynamically from the config file.

    Returns
    -------
    DataStructure
        The net generator and dataloader generator functions.
    """
    # Create the list of task dispatches to try
    task_data_dependent_functions: list[
        Callable[[DictConfig], DataStructure | None]
    ] = [
        dispatch_resnet18_data,
        dispatch_speech_resnet18_data,
    ]

    # Match the first function which does not return None
    for task in task_data_dependent_functions:
        result = task(cfg)
        if result is not None:
            return result

    raise ValueError(
        f"Unable to match the net generator and dataloader generator functions: {cfg}",
    )


def dispatch_config(cfg: DictConfig) -> ConfigStructure:
    """Dispatch the fit/eval config functions based on on the hydra config.

    Functionality should be added to the dispatch.py
    file in the task folder.
    Statically specify the new dispatch function in the list,
    function order determines precedence
    if two different tasks may match the config.

    Parameters
    ----------
    cfg : DictConfig
        The configuration for the config function.
        Loaded dynamically from the config file.

    Returns
    -------
    ConfigStructure
        The config functions.
    """
    # Create the list of task dispatches to try
    task_config_functions: list[Callable[[DictConfig], ConfigStructure | None]] = [
        dispatch_default_config,
        dispatch_resnet18_config,
        dispatch_speech_resnet18_config,
    ]

    # Match the first function which does not return None
    for task in task_config_functions:
        result = task(cfg)
        if result is not None:
            return result

    raise ValueError(
        f"Unable to match the config generation functions: {cfg}",
    )
