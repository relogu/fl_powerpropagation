"""Create and connect the building blocks for your experiments; start the simulation.

It includes processing the dataset, instantiate strategy, specifying how the global
model will be evaluated, etc. In the end, this script saves the results.
"""

import logging
import os
from pathlib import Path
import time

from flwr.common import log


import hydra
from project.fed.utils.plot_utils import plot_abs_parameter_distribution
from project.task.cifar.dataset import get_dataloader_generators
from project.task.cifar.models import get_network_generator_resnet_powerprop
from project.task.cifar.train_test import train


from omegaconf import DictConfig

# Only import from the project root
# Never do a relative import nor one that assumes a given folder structure
from project.fed.utils.utils import (
    generate_random_state_dict,
    generic_get_parameters,
    generic_set_parameters,
)
from project.task.cifar_swat.models import get_network_generator_resnet_swat
from project.types.common import NetGen
from project.utils.utils import (
    obtain_device,
)


# Make debugging easier when using Hydra + Ray
os.environ["HYDRA_FULL_ERROR"] = "1"
os.environ["OC_CAUSE"] = "1"
os.environ["RAY_MEMORY_MONITOR_REFRESH_MS"] = "0"


@hydra.main(
    config_path="conf",
    config_name="base",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    """Powerprop test."""
    _cfg = dict([[], []])  # ? remove this to run

    device = obtain_device()
    config = {
        "device": device,
        "epochs": 1,
        "learning_rate": 0.03,
        "batch_size": 32,
    }
    sparsity = 0.0
    log(logging.INFO, f"[main-powerprop] config:{config}")

    get_resnet_swat: NetGen = get_network_generator_resnet_swat()
    net = get_resnet_swat(_cfg)
    net.load_state_dict(generate_random_state_dict(net, seed=42, sparsity=sparsity))
    net.to(device)

    # Full training
    get_dataloader, _ = get_dataloader_generators(Path("data/cifar/partition"))
    data = get_dataloader(0, False, config)

    start_time = time.time()
    log(logging.INFO, train(net, data, config, Path()))
    # print(train_and_prune(net, data, config, Path('')))
    log(logging.INFO, f"[main] It took {int(time.time() - start_time)}s to run")

    # Net genereting
    get_resnet_power: NetGen = get_network_generator_resnet_powerprop()
    p_net = get_resnet_power(_cfg)
    generic_set_parameters(p_net, generic_get_parameters(net))
    p_net.to(device)
    # Create a copy to check the difference after the training

    start_time = time.time()
    log(logging.INFO, train(p_net, data, config, Path()))
    # print(train_and_prune(net, data, config, Path('')))
    log(logging.INFO, f"[main] It took {int(time.time() - start_time)}s to run")

    plot_abs_parameter_distribution(p_net, net)
    log(logging.INFO, f"[main] It took {int(time.time() - start_time)}s to run")


if __name__ == "__main__":
    main()
