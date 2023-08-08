import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from lda_utils import create_lda_partitions
from numpy.random import BitGenerator, Generator, SeedSequence
from torch import Tensor
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Normalize, ToTensor

CIFAR10_DATA_ROOT = "/datasets/CIFAR10"


def load_data(
    root_dir: str = CIFAR10_DATA_ROOT,
    batch_size: int = 32,
) -> Tuple[DataLoader, DataLoader, Dict]:
    """Download and load the centralised CIFAR-10 (both training and test set)."""
    transform = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = CIFAR10(root_dir, train=True, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testset = CIFAR10(root_dir, train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    num_examples = {"trainset": len(trainset), "testset": len(testset)}
    return trainloader, testloader, num_examples


def create_lda_cifar10_partitions(
    root_dir: str = CIFAR10_DATA_ROOT,
    num_partitions: int = 100,
    concentration: float = 0.5,
    seed: Optional[Union[int, SeedSequence, BitGenerator, Generator]] = None,
) -> List[str]:
    """Create imbalanced non-iid partitions using Latent Dirichlet Allocation
    (LDA) without resampling for the CIFAR10 dataset. The dataset is loaded
    from the `root_dir` directory. Note that the dataset will be
    downloaded if it does not exist in `root_dir` directory. The
    partitions are saved as pickle files in the following directory structure:
    `root_dir`
        |--- `lda`
            |--- `num_partitions`
                |--- `concentration`
                    |--- `cid` from 0 to `num_partitions`
                        |--- `train.pickle`
                        |--- `test.pickle`

    Args:
        num_partitions (int): number of partitions (clients) to be created
            Defaults to 100.
        concentration (float): Dirichlet Concentration
            (:math:)`\\alpha` parameter. Set to float('inf') to get uniform partitions.
            An :math:`\\alpha \\to \\Inf` generates uniform distributions over classes.
            An :math:`\\alpha \\to 0.0` generates one class per client. Defaults to 0.5.
        seed (None, int, SeedSequence, BitGenerator, Generator):
            A seed to initialize the BitGenerator for generating the Dirichlet
            distribution. This is defined in Numpy's official documentation as follows:
            If None, then fresh, unpredictable entropy will be pulled from the OS.
            One may also pass in a SeedSequence instance.
            Additionally, when passed a BitGenerator, it will be wrapped by Generator.
            If passed a Generator, it will be returned unaltered.
            See official Numpy Documentation for further details.

    Returns:
        List[str]: list of strings containing the paths to the partitions.
    """

    # Set the partition root
    partition_root = (
        Path(root_dir) / "lda" / f"{num_partitions}" / f"{concentration:.2f}"
    )

    # Check whether the partition exists
    if partition_root.is_dir():
        print("Partitions already exist. Delete if necessary.")
        return [str(p) for p in partition_root.glob("*")]
    else:
        # Get the train and test set for the CIFAR10 dataset
        trainset = CIFAR10(
            root_dir,
            train=True,
            download=True,
        )
        testset = CIFAR10(
            root_dir,
            train=False,
            download=True,
        )

        # Cast train features and labels into numpy arrays to be used by LDA utilities
        x = np.array(trainset.data)
        y = np.array(trainset.targets)

        # Create LDA partitions for train set and the LDA distributions
        train_clients_partitions, train_dists = create_lda_partitions(
            dataset=(x, y),
            dirichlet_dist=None,
            num_partitions=num_partitions,
            concentration=concentration,
            accept_imbalanced=True,
            seed=seed,
        )

        # Cast test features and labels into numpy arrays to be used by LDA utilities
        x = np.array(testset.data)
        y = np.array(testset.targets)

        # Create LDA partitions for test set and from the previous distributions
        test_clients_partitions, _ = create_lda_partitions(
            dataset=(x, y),
            dirichlet_dist=train_dists,
            num_partitions=num_partitions,
            concentration=concentration,
            accept_imbalanced=True,
            seed=seed,
        )

        # Store the poartitions into the disk
        local_datasets_paths = []
        for cid in range(num_partitions):
            # Create client partition root
            partition_cid = partition_root / f"{cid}"
            partition_cid.mkdir(parents=True)
            # Store the local train set as a pickle file
            train_file = partition_cid / "train.pickle"
            train_partition = train_clients_partitions[cid]
            with open(train_file, "wb") as f:
                pickle.dump(train_partition, f)
            # Store the local test set as a pickle file
            test_file = partition_cid / "test.pickle"
            test_partition = test_clients_partitions[cid]
            with open(test_file, "wb") as f:
                pickle.dump(test_partition, f)
            # Store the local dataset path
            local_datasets_paths.append(str(partition_cid))

        # Return the local datasets paths
        return local_datasets_paths


def load_partitioned_data(
    partitions_root: Path,
    batch_size: int = 32,
) -> Tuple[DataLoader, DataLoader, Dict]:
    """Load the train and test partitions from the specified root directory.
    The function will load a `train.pickle` and a `test.pickle` file from the
    directory specified by `partitions_root`. The files are expected to exist.
    Data loaders are shuffled for both train and test. Data loaders use the
    default parameters where not specified.

    Args:
        partitions_root (Path): the path to the root directory containing the partitions.
        batch_size (int, optional): the batch size pf the data loaders. Defaults to 32.

    Returns:
        Tuple[DataLoader, DataLoader, Dict]: tuple containing the train loader,
            the test loader and a dictionary containing the number of examples.
    """
    # Set the transform
    transform = Compose([Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    # Load and instantiate the local train set
    with open(partitions_root / "train.pickle", "rb") as f:
        np_trainset = pickle.load(f)
    trainset = CustomTensorDataset(
        tensors=[
            Tensor(np.transpose(np_trainset[0], (0, 3, 1, 2))),
            Tensor(np_trainset[1]).to(torch.int64),
        ],
        transform=transform,
    )
    # Instantiate the train loader
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

    # Load and instantiate the local test set
    with open(partitions_root / "test.pickle", "rb") as f:
        np_testset = pickle.load(f)
    testset = CustomTensorDataset(
        tensors=[
            Tensor(np.transpose(np_testset[0], (0, 3, 1, 2))),
            Tensor(np_testset[1]).to(torch.int64),
        ],
        transform=transform,
    )
    # Instantiate the test loader
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    # Get the number of examples by set
    num_examples = {"trainset": len(trainset), "testset": len(testset)}
    # Return the loaders and the number of examples
    return trainloader, testloader, num_examples


def load_centralised_test_set(
    partitions_root: Path,
    batch_size: int = 32,
) -> Tuple[DataLoader, DataLoader, Dict]:
    """Load the centralised test set from the specified root directory.
    The centralised test set is intended to be composed of the concatenation
    of the local test sets of clients. The function will load all the
    `test.pickle` files from the directory specified by `partitions_root`.
    The files are expected to exist. The `partitions_root` folder should
    contain one folder for each client, and each of these folders should
    contain a `test.pickle` file.


    Args:
        partitions_root (Path): the path to the root directory containing the partitions.
        batch_size (int, optional): the batch size pf the data loaders. Defaults to 32.

    Returns:
        Tuple[DataLoader, DataLoader, Dict]: tuple containing the train loader,
            the test loader and a dictionary containing the number of examples.
    """
    # Set the transform
    transform = Compose([Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    list_of_clients_folders = list(partitions_root.glob("*"))
    clients_test_sets = []
    for client_folder in list_of_clients_folders:
        # Load and instantiate the local test set
        with open(client_folder / "test.pickle", "rb") as f:
            np_testset = pickle.load(f)
        clients_test_sets.append(
            CustomTensorDataset(
                tensors=[
                    Tensor(np.transpose(np_testset[0], (0, 3, 1, 2))),
                    Tensor(np_testset[1]).to(torch.int64),
                ],
                transform=transform,
            )
        )
    # Concatenate the clients test sets
    testset = ConcatDataset(clients_test_sets)
    # Instantiate the test loader
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    # Return the loaders and the number of examples
    return testset, testloader, len(testset)


class CustomTensorDataset(Dataset):
    """TensorDataset with support of transforms."""

    def __init__(self, tensors, transform=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        x = self.tensors[0][index] / 255

        if self.transform:
            x = self.transform(x)

        y = self.tensors[1][index]

        return x, y

    def __len__(self):
        return self.tensors[0].size(0)


if __name__ == "__main__":
    import os

    print("Running cifar10_utils.py script as a main file.")
    print(
        f"The dataset will be downloaded at {CIFAR10_DATA_ROOT} and partitioned in the same folder."
    )
    print(
        "The download will be skipped if the dataset is already present in the folder."
    )
    print(
        "The script will use LDA to partition the dataset, with the default parameters: number of clients = 100, concentration = 0.5, and seed = 51550."
    )

    response = input("Do you want to change the default parameters? [y/n] ")

    n, alpha, seed = 100, 0.5, 51550

    if response == "y":
        n = int(input("Set the number of clients"))
        alpha = float(input("Set the concentration parameter"))
        seed = int(input("Set the seed"))

    response = input("Do you want to continue downloading and partitioning? [y/n] ")

    if response == "y":
        print("Downloading the dataset...")
        load_data()
        print("Starting the partitioning process...")
        partitioned_data_paths = create_lda_cifar10_partitions()
        print(f"Partitioned data saved at {partitioned_data_paths}")
    else:
        print("Exiting...")
        os._exit(0)
