"""Functions for CIFAR download and processing."""

import logging
from pathlib import Path

import hydra
import numpy as np
import torch
from flwr.common.logger import log
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import ConcatDataset, Subset, DataLoader
from torchaudio.datasets import SPEECHCOMMANDS


from project.utils.utils import obtain_device
from project.task.utils.common import XY, XYList, create_lda_partitions

HYDRA_FULL_ERROR = 1


class SubsetSC(SPEECHCOMMANDS):
    """Subset of speech commands dataset.

    Source:
    """

    def __init__(
        self,
        subset: str,
        dataset_dir: Path = Path("./data/speech/data/"),
    ) -> None:
        dataset_dir.mkdir(parents=True, exist_ok=True)
        dataset_dir = Path.resolve(dataset_dir)
        super().__init__(str(dataset_dir), download=True)

        dataset_dir = dataset_dir / "SpeechCommands/speech_commands_v0.02/"

        def load_list(filename: Path) -> list[Path]:
            """Return a list of file names."""
            filepath = dataset_dir / filename
            with open(filepath, encoding="utf-8") as fileobj:  # Add encoding argument
                return [Path(dataset_dir) / line.strip() for line in fileobj]

        if subset == "validation":
            self._walker = load_list(Path("validation_list.txt"))
        elif subset == "testing":
            self._walker = load_list(Path("testing_list.txt"))
        elif subset == "training":
            excludes = set(
                load_list(Path("validation_list.txt"))
                + load_list(Path("testing_list.txt"))
            )
            self._walker = [w for w in self._walker if w not in excludes]


def pad_sequence(batch: torch.Tensor) -> torch.Tensor:
    """Pad a batch of variable length tensors with zeros."""
    # Make all tensor in a batch the same length by padding with zeros
    batch = [item.t() for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.0)
    return batch.permute(0, 2, 1)


def label_to_index(labels: list[str], word: str) -> torch.Tensor:
    """Return the position of the word in labels."""
    # Return the position of the word in labels
    return torch.tensor(labels.index(word))


def index_to_label(labels: list[str], index: int) -> str:
    """Return the word corresponding to the index in labels."""
    # Return the word corresponding to the index in labels
    # This is the inverse of label_to_index
    return labels[index]


labels = [
    "backward",
    "bed",
    "bird",
    "cat",
    "dog",
    "down",
    "eight",
    "five",
    "follow",
    "forward",
    "four",
    "go",
    "happy",
    "house",
    "learn",
    "left",
    "marvin",
    "nine",
    "no",
    "off",
    "on",
    "one",
    "right",
    "seven",
    "sheila",
    "six",
    "stop",
    "three",
    "tree",
    "two",
    "up",
    "visual",
    "wow",
    "yes",
    "zero",
]


def _download_data(
    dataset_dir: Path,
) -> tuple[SPEECHCOMMANDS, SPEECHCOMMANDS]:
    """Download the dataset and return the training and testing set."""
    trainset = SubsetSC(dataset_dir=dataset_dir, subset="training")
    testset = SubsetSC(dataset_dir=dataset_dir, subset="testing")

    return trainset, testset


# pylint: disable=too-many-locals
def _partition_data(
    trainset: SPEECHCOMMANDS,
    testset: SPEECHCOMMANDS,
    num_clients: int,
    lda: bool,
    lda_alpha: float,
) -> tuple[
    list[Subset] | list[ConcatDataset] | tuple[XYList, np.ndarray], SPEECHCOMMANDS
]:
    """Partition the dataset into training and testing set for each client."""
    # PARTITIONING THE TRAINING SET
    tensors, targets = [], []

    # Gather in lists, and encode labels as indices
    for waveform, _, label, *_ in trainset:
        tensors += [waveform]
        targets += [label_to_index(labels=labels, word=label)]

    # Group the list of tensors into a batched tensor
    tensors = pad_sequence(tensors)
    targets = torch.stack(targets)

    # Pack into a tuple
    xy: XY = (tensors, targets)

    # Create LDA partitions
    partitioned_trainset, dirichlet_dist = create_lda_partitions(
        dataset=xy,
        num_partitions=num_clients,
        concentration=lda_alpha,
        accept_imbalanced=True,
    )

    # PARTITIONING THE TEST SET
    tensors, targets = [], []
    # Gather in lists, and encode labels as indices
    # for waveform, _, label, *_ in trainset:

    # print("Untile here everithing is fine")
    # for metadata in testset._walker:
    #     print("Metadata:", metadata)
    for waveform, _, label, *_ in testset:
        tensors += [waveform]
        targets += [label_to_index(labels=labels, word=label)]

    # Group the list of tensors into a batched tensor
    tensors = pad_sequence(tensors)
    targets = torch.stack(targets)

    xy = (tensors, targets)

    partitioned_testset, _ = create_lda_partitions(
        dataset=xy,
        dirichlet_dist=dirichlet_dist,
        num_partitions=num_clients,
        concentration=lda_alpha,
        accept_imbalanced=True,
    )

    return partitioned_trainset, partitioned_testset


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
    fed_train_set, fed_test_set = _partition_data(
        trainset,
        testset,
        cfg.dataset.num_clients,
        cfg.dataset.lda,
        cfg.dataset.lda_alpha,
    )

    # 2. Save the datasets
    # unnecessary for this small dataset, but useful for large datasets
    partition_dir = Path(cfg.dataset.partition_dir)
    partition_dir.mkdir(parents=True, exist_ok=True)

    # Save the centralised test set a centrailsed training set would also be possible
    # but is not used here
    torch.save(testset, partition_dir / "test.pt")

    for idx, client_dataset in enumerate(fed_train_set):
        log(logging.INFO, len(client_dataset[0]))
        client_dir = partition_dir / f"client_{idx}"
        client_dir.mkdir(parents=True, exist_ok=True)
        # Saving the client trainset
        subset_dict = {"data": client_dataset[0], "targets": client_dataset[1]}
        torch.save(subset_dict, client_dir / "train.pt")
        # Saving the client testset
        subset_dict = {
            "data": fed_test_set[idx][0],
            "targets": fed_test_set[idx][1],
        }
        torch.save(subset_dict, client_dir / "test.pt")

        # test_client_dataloader(
        #     partition_dir,
        #     idx,
        #     64,
        #     test=False,
        # )


# TEST
def test_client_dataloader(
    partition_dir: Path,
    cid: str | int,
    batch_size: int,
    test: bool,
) -> DataLoader:
    """Return a DataLoader for a client's dataset.

    Parameters
    ----------
    cid : str|int
        The client's ID
    test : bool
        Whether to load the test set or not
    config : Dict
        The configuration for the dataset

    Returns
    -------
    DataLoader
        The DataLoader for the client's dataset
    """
    client_dir = partition_dir / f"client_{cid}"
    if not test:
        dataset = torch.load(client_dir / "train.pt")
    else:
        dataset = torch.load(client_dir / "test.pt")

    dataset = DataLoader(
        list(zip(dataset["data"], dataset["targets"], strict=True)),
        batch_size=batch_size,
        shuffle=not test,
    )

    device = obtain_device()
    # for data, target in list(zip(dataset['data'], dataset['targets'])):
    # print(dataset['data'].shape)
    for data, target in dataset:
        data, target = (
            data.to(
                device,
            ),
            target.to(device),
        )
        break

    return dataset


if __name__ == "__main__":
    download_and_preprocess()
