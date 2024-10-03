"""Dispatches the functionality of the task.

This gives us the ability to dynamically choose functionality based on the hydra dict
config without losing static type checking.
"""

from collections.abc import Callable

from omegaconf import DictConfig

from project.task.default.dispatch import dispatch_config as dispatch_default_config
from project.task.default.dispatch import dispatch_data as dispatch_default_data
from project.task.default.dispatch import dispatch_train as dispatch_default_train

from project.task.mnist_classification.dispatch import (
    dispatch_config as dispatch_mnist_config,
)
from project.task.mnist_classification.dispatch import (
    dispatch_data as dispatch_mnist_data,
)
from project.task.mnist_classification.dispatch import (
    dispatch_train as dispatch_mnist_train,
)

from project.task.cifar.dispatch import (
    dispatch_config as dispatch_cifar_config,
)
from project.task.cifar.dispatch import (
    dispatch_data as dispatch_cifar_data,
)
from project.task.cifar.dispatch import (
    dispatch_train as dispatch_cifar_train,
)

from project.task.cifar_powerprop.dispatch import (
    dispatch_config as dispatch_cifar_powerprop_config,
)
from project.task.cifar_powerprop.dispatch import (
    dispatch_data as dispatch_cifar_powerprop_data,
)
from project.task.cifar_powerprop.dispatch import (
    dispatch_train as dispatch_cifar_powerprop_train,
)

from project.task.cifar_swat.dispatch import (
    dispatch_config as dispatch_cifar_swat_config,
)
from project.task.cifar_swat.dispatch import (
    dispatch_data as dispatch_cifar_swat_data,
)
from project.task.cifar_swat.dispatch import (
    dispatch_train as dispatch_cifar_swat_train,
)

from project.task.cifar_power_swat.dispatch import (
    dispatch_config as dispatch_cifar_power_swat_config,
)
from project.task.cifar_power_swat.dispatch import (
    dispatch_data as dispatch_cifar_power_swat_data,
)
from project.task.cifar_power_swat.dispatch import (
    dispatch_train as dispatch_cifar_power_swat_train,
)


from project.task.cifar_resnet18.dispatch import (
    dispatch_config as dispatch_resnet18_config,
)
from project.task.cifar_resnet18.dispatch import dispatch_data as dispatch_resnet18_data
from project.task.cifar_resnet18.dispatch import (
    dispatch_train as dispatch_resnet18_train,
)

from project.task.cifar_zerofl.dispatch import (
    dispatch_config as dispatch_cifar_zeroFL_config,
)
from project.task.cifar_zerofl.dispatch import (
    dispatch_data as dispatch_cifar_zeroFL_data,
)
from project.task.cifar_zerofl.dispatch import (
    dispatch_train as dispatch_cifar_zeroFL_train,
)

# Cifar FLASH
from project.task.cifar_flash.dispatch import (
    dispatch_config as dispatch_cifar_flash_config,
)
from project.task.cifar_flash.dispatch import (
    dispatch_data as dispatch_cifar_flash_data,
)
from project.task.cifar_flash.dispatch import (
    dispatch_train as dispatch_cifar_flash_train,
)


# Speech command dispatch
from project.task.speech_resnet18.dispatch import (
    dispatch_config as dispatch_speech_resnet18_config,
)
from project.task.speech_resnet18.dispatch import (
    dispatch_data as dispatch_speech_resnet18_data,
)
from project.task.speech_resnet18.dispatch import (
    dispatch_train as dispatch_speech_resnet18_train,
)


from project.types.common import ConfigStructure, DataStructure, TrainStructure


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
        dispatch_default_train,
        dispatch_mnist_train,
        dispatch_cifar_train,
        dispatch_cifar_swat_train,
        dispatch_cifar_power_swat_train,
        dispatch_cifar_powerprop_train,
        dispatch_resnet18_train,
        dispatch_cifar_zeroFL_train,
        dispatch_speech_resnet18_train,
        dispatch_cifar_flash_train,
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
        dispatch_mnist_data,
        dispatch_default_data,
        dispatch_cifar_data,
        dispatch_cifar_powerprop_data,
        dispatch_cifar_swat_data,
        dispatch_cifar_power_swat_data,
        dispatch_resnet18_data,
        dispatch_cifar_zeroFL_data,
        dispatch_speech_resnet18_data,
        dispatch_cifar_flash_data,
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
        dispatch_mnist_config,
        dispatch_cifar_config,
        dispatch_default_config,
        dispatch_cifar_powerprop_config,
        dispatch_cifar_swat_config,
        dispatch_cifar_power_swat_config,
        dispatch_resnet18_config,
        dispatch_cifar_zeroFL_config,
        dispatch_speech_resnet18_config,
        dispatch_cifar_flash_config,
    ]

    # Match the first function which does not return None
    for task in task_config_functions:
        result = task(cfg)
        if result is not None:
            return result

    raise ValueError(
        f"Unable to match the config generation functions: {cfg}",
    )
