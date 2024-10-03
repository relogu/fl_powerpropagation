"""Create and connect the building blocks for your experiments; start the simulation.

It includes processing the dataset, instantiate strategy, specifying how the global
model will be evaluated, etc. In the end, this script saves the results.
"""

from copy import deepcopy
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
)
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
    device = obtain_device()
    config = {
        "device": device,
        "epochs": 1,
        "learning_rate": 0.03,
        "batch_size": 32,
    }
    sparsity = 0.0
    log(logging.INFO, f"[main-powerprop] config:{config}")

    # Net genereting
    get_resnet_power: NetGen = get_network_generator_resnet_powerprop()
    fake_config = dict([[], []])
    net = get_resnet_power(fake_config)
    net.load_state_dict(generate_random_state_dict(net, seed=42, sparsity=sparsity))
    net.to(device)
    # Create a copy to check the difference after the training
    _net = deepcopy(net)

    # Test training
    # Reproducible input data
    # input = (torch.randn(1, 3, 224, 224)).to(device)
    # Random target
    # target = (torch.randint(0,10,(1,)).long()).to(device)
    # train_swat_test(net=model, _config=config, input=input, target=target)

    # plot_abs_parameter_distribution(net)

    # Full training
    get_dataloader, _ = get_dataloader_generators(Path("data/cifar/partition"))
    data = get_dataloader(0, False, config)
    start_time = time.time()

    # train_and_prune: NetGen = get_train_and_prune(amount=0.3, pruning_method="l1")

    log(logging.INFO, train(net, data, config, Path()))
    # print(train_and_prune(net, data, config, Path('')))
    log(logging.INFO, f"[main] It took {int(time.time() - start_time)}s to run")

    # Comparing original network and the trained one
    # net_compare(_net, net, "main")

    plot_abs_parameter_distribution(_net, net)
    log(logging.INFO, f"[main] It took {int(time.time() - start_time)}s to run")


if __name__ == "__main__":
    main()
