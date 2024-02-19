"""Functions for CIFAR download and processing."""

import logging
from collections.abc import Sequence

from pathlib import Path
from typing import cast

import hydra
import numpy as np
import torch
from flwr.common.logger import log
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import ConcatDataset, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10

from project.task.utils.common import XYList, create_lda_partitions

HYDRA_FULL_ERROR = 1


def _download_data(
    dataset_dir: Path,
) -> tuple[CIFAR10, CIFAR10]:  # ?
    """Download (if necessary) and returns the CIFAR10 dataset.

    Returns
    -------
    Tuple[CIFAR10, CIFAR10]
        The dataset for training and the dataset for testing CIFAR10.
    """
    transform = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset_dir.mkdir(parents=True, exist_ok=True)

    trainset = CIFAR10(
        str(dataset_dir),
        train=True,
        download=True,
        transform=transform,
    )
    testset = CIFAR10(
        str(dataset_dir),
        train=False,
        download=True,
        transform=transform,
    )
    return trainset, testset


# pylint: disable=too-many-locals
def _partition_data(
    trainset: CIFAR10,
    testset: CIFAR10,
    num_clients: int,
    seed: int,
    balance: bool,
    lda_alpha: float = 1,
) -> tuple[list[Subset] | list[ConcatDataset] | tuple[XYList, np.ndarray], CIFAR10]:
    """Split training set into iid or non iid partitions to simulate the federated.

    setting.

    Parameters
    ----------
    num_clients : int
        The number of clients that hold a part of the data
    lda_alpha : float
        The concentration parameter for the dirichlet distribution, by default 0.5

    balance : bool
        Whether the dataset should contain an equal number of samples in each class,
        by default False
    seed : int
        Used to set a fix seed to replicate experiments, by default 42

    Returns
    -------
    Tuple[List[CIFAR10], CIFAR10]
        A list of dataset for each client and a single dataset to be used for testing
        the model.
    """
    if balance:
        trainset = _balance_classes(trainset, seed)

    x = torch.from_numpy(np.array([sample[0] for sample in trainset]))
    y = torch.from_numpy(np.array([sample[1] for sample in trainset]))
    # Pack into a tuple
    xy = (x, y)
    datasets = create_lda_partitions(
        dataset=xy,
        num_partitions=num_clients,
        concentration=lda_alpha,
    )

    return datasets, testset


def _balance_classes(
    trainset: CIFAR10,
    seed: int,
) -> CIFAR10:
    """Balance the classes of the trainset.

    Trims the dataset so each class contains as many elements as the
    class that contained the least elements.

    Parameters
    ----------
    trainset : CIFAR10
        The training dataset that needs to be balanced.
    seed : int, optional
        Used to set a fix seed to replicate experiments, by default 42.

    Returns
    -------
    CIFAR10
        The balanced training dataset.
    """
    class_counts = np.bincount(trainset.targets)
    smallest = np.min(class_counts)
    idxs = trainset.targets.argsort()
    tmp = [
        Subset(
            trainset,
            cast(Sequence[int], idxs[: int(smallest)]),
        ),
    ]
    tmp_targets = [trainset.targets[idxs[: int(smallest)]]]
    for count in np.cumsum(class_counts):
        tmp.append(
            Subset(
                trainset,
                cast(
                    Sequence[int],
                    idxs[int(count) : int(count + smallest)],
                ),
            ),
        )
        tmp_targets.append(
            trainset.targets[idxs[int(count) : int(count + smallest)]],
        )
    unshuffled = ConcatDataset(tmp)
    unshuffled_targets = torch.cat(tmp_targets)
    shuffled_idxs = torch.randperm(
        len(unshuffled),
        generator=torch.Generator().manual_seed(seed),
    )
    shuffled = cast(
        CIFAR10,
        Subset(
            unshuffled,
            cast(Sequence[int], shuffled_idxs),
        ),
    )
    shuffled.targets = unshuffled_targets[shuffled_idxs]

    return shuffled


@hydra.main(
    config_path="../../conf",
    config_name="base",
    version_base=None,
)
def download_and_preprocess(cfg: DictConfig) -> None:
    """Download and preprocess the dataset.

    Please include here all the logic
    Please use the Hydra config style as much as possible specially
    for parts that can be customised (e.g. how data is partitioned)

    Parameters
    ----------
    cfg : DictConfig
        An omegaconf object that stores the hydra config.
    """
    # 1. print parsed config
    log(logging.INFO, OmegaConf.to_yaml(cfg))

    # Check if the partitioning already exists
    if Path.exists(Path(cfg.dataset.partition_dir)):
        log(
            logging.INFO, f"Partitioning already exists at: {cfg.dataset.partition_dir}"
        )
        return

    # Download the dataset
    trainset, testset = _download_data(
        Path(cfg.dataset.dataset_dir),
    )

    # Partition the dataset
    # ideally, the fed_test_set can be composed in three ways:
    # 1. fed_test_set = centralised test set like MNIST
    # 2. fed_test_set = concatenation of all test sets of all clients
    # 3. fed_test_set = test sets of reserved unseen clients
    client_datasets, fed_test_set = _partition_data(
        trainset,
        testset,
        cfg.dataset.num_clients,
        cfg.dataset.seed,
        cfg.dataset.iid,
        cfg.dataset.power_law,
        cfg.dataset.lda,
        cfg.dataset.lda_alpha,
        cfg.dataset.balance,
    )

    # 2. Save the datasets
    # unnecessary for this small dataset, but useful for large datasets
    partition_dir = Path(cfg.dataset.partition_dir)
    partition_dir.mkdir(parents=True, exist_ok=True)

    # Save the centralised test set a centrailsed training set would also be possible
    # but is not used here
    torch.save(fed_test_set, partition_dir / "test.pt")

    client_trainsets = client_datasets[0]

    # Create the test sets for each client, following the same distribution
    dirichlet_dict = client_datasets[1]
    x = torch.from_numpy(np.array([sample[0] for sample in fed_test_set]))
    y = torch.from_numpy(np.array([sample[1] for sample in fed_test_set]))
    xy = (x, y)
    client_testsets, _ = create_lda_partitions(
        dataset=xy,
        dirichlet_dist=dirichlet_dict,
        num_partitions=cfg.dataset.num_clients,
        concentration=cfg.dataset.lda_alpha,
    )

    for idx, client_dataset in enumerate(client_trainsets):
        log(logging.INFO, len(client_dataset[0]))
        client_dir = partition_dir / f"client_{idx}"
        client_dir.mkdir(parents=True, exist_ok=True)

        # Saving the client trainset
        subset_dict = {"data": client_dataset[0], "targets": client_dataset[1]}
        torch.save(subset_dict, client_dir / "train.pt")
        # Saving the client testset

        subset_dict = {
            "data": client_testsets[idx][0],
            "targets": client_testsets[idx][1],
        }
        torch.save(subset_dict, client_dir / "test.pt")
