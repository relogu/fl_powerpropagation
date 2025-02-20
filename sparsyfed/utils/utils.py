"""Define any utility function.

Generic utilities.
"""

import logging
import random
import re
import shutil
from collections.abc import Callable, Iterator
from itertools import chain
from pathlib import Path
from types import TracebackType
from typing import Any, cast

import numpy as np
import ray
import torch
from flwr.common.logger import log

import wandb


def obtain_device() -> torch.device:
    """Get the device (CPU or GPU) for torch.

    Returns
    -------
    torch.device
        The device.
    """
    return torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu",
    )


def lazy_wrapper(x: Callable) -> Callable[[], Any]:
    """Wrap a value in a function that returns the value.

    For easy instantion through hydra.

    Parameters
    ----------
    x : Callable
        The value to wrap.

    Returns
    -------
    Callable[[], Any]
        The wrapped value.
    """
    return lambda: x


def lazy_config_wrapper(
    x: Callable,
) -> Callable[[dict], Any]:
    """Wrap a value in a function that returns the value given a config.

    For easy instantion through hydra.

    Parameters
    ----------
    x : Callable
        The value to wrap.

    Returns
    -------
    Callable[[Dict], Any]
        The wrapped value.
    """
    return lambda _config: x()


def seed_everything(seed: int) -> None:
    """Seed everything for reproducibility.

    Parameters
    ----------
    seed : int
        The seed.

    Returns
    -------
        None
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class NoOpContextManager:
    """A context manager that does nothing."""

    def __enter__(self) -> None:
        """Do nothing."""
        return

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_value: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        """Do nothing."""


def wandb_init(
    wandb_enabled: bool,
    *args: Any,
    **kwargs: Any,
) -> NoOpContextManager | Any:
    """Initialize wandb if enabled.

    Parameters
    ----------
    wandb_enabled : bool
        Whether wandb is enabled.
    *args : Any
        The arguments to pass to wandb.init.
    **kwargs : Any
        The keyword arguments to pass to wandb.init.

    Returns
    -------
    Optional[Union[NoOpContextManager, Any]]
        The wandb context manager if enabled, otherwise a no-op context manager
    """
    if wandb_enabled:
        return wandb.init(*args, **kwargs)

    return NoOpContextManager()


class RayContextManager:
    """A context manager for cleaning up after ray."""

    def __enter__(self) -> "RayContextManager":
        """Initialize the context manager."""
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_value: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        """Cleanup the files.

        Parameters
        ----------
        _exc_type : Any
            The exception type.
        _exc_value : Any
            The exception value.
        _traceback : Any
            The traceback.

        Returns
        -------
        None
        """
        if ray.is_initialized():
            temp_dir = Path(
                ray.worker._global_node.get_session_dir_path(),
            )
            ray.shutdown()
            directory_size = shutil.disk_usage(
                temp_dir,
            ).used
            shutil.rmtree(temp_dir)
            log(
                logging.INFO,
                f"Cleaned up ray temp session: {temp_dir} with size: {directory_size}",
            )


def cleanup(working_dir: Path, to_clean: list[str]) -> None:
    """Cleanup the files in the working dir.

    Parameters
    ----------
    working_dir : Path
        The working directory.
    to_clean : List[str]
        The tokens to clean.

    Returns
    -------
        None
    """
    children: list[Path] = []
    for file in working_dir.iterdir():
        if file.is_file():
            for clean_token in to_clean:
                if clean_token in file.name and file.exists():
                    file.unlink()
                break
        else:
            children.append(file)

    for child in children:
        cleanup(child, to_clean)


def get_checkpoint_index(
    output_dir: Path,
    file_limit: int | None,
) -> int:
    """Get the index of the next checkpoint.

    Parameters
    ----------
    output_dir : Path
        The output directory.
    file_limit : Optional[int]
        The maximal number of files to search.
        If None, then there is no limit.

    Returns
    -------
    int
        The index of the next checkpoint.
    """
    same_name_files = cast(
        Iterator[Path],
        chain(
            output_dir.glob("*_*"),
            output_dir.glob("*/*_*"),
        ),
    )

    same_name_files = (
        same_name_files
        if file_limit is None
        else (next(same_name_files) for _ in range(file_limit))
    )

    indicies = (
        int(v.group(1))
        for f in same_name_files
        if (v := re.search(r"_([0-9]+)", f.stem))
    )

    return max(indicies, default=-1) + 1


def get_clients_window_range(
    current_round: int,
    window_size: int = 50,  # CIFAR10
    # window_size: int = 25, # CIFAR100
    window_training_rounds: int = 200,
    num_clients: int = 100,
    overlap: int = 0,
) -> tuple[int, int]:
    """Get the clients window range for the round.

    Parameters
    ----------
    round : int
        The round.
    window_size : int
        The window size.
    num_clients : int
        The number of clients.
    overlap : int
        The overlap between windows.

    Returns
    -------
    Tuple[int, int]
        The window range.
    """
    start = ((current_round // window_training_rounds) * window_size) % num_clients
    end = start + window_size + overlap
    # print(f"RANGE: start: {start}, end: {end}")
    return int(start), int(end)


def save_files(
    working_dir: Path,
    output_dir: Path,
    to_save: list[str],
    checkpoint_index: int,
    ending: int | None = None,
    top_level: bool = True,
) -> None:
    """Save the files in the working dir.

    Parameters
    ----------
    working_dir : Path
        The working directory.
    output_dir : Path
        The output directory.

    Returns
    -------
        None
    """
    if not top_level:
        output_dir = output_dir / working_dir.name

    children: list[Path] = []
    for file in working_dir.iterdir():
        if file.is_file():
            for save_token in to_save:
                if save_token in file.name and file.exists():
                    true_ending = (
                        f"{checkpoint_index}" + ("_" + str(ending))
                        if ending is not None
                        else f"{checkpoint_index}"
                    )
                    destination_file = (
                        output_dir
                        / file.with_stem(
                            f"{file.stem}_{true_ending}",
                        ).name
                    )

                    destination_file.parent.mkdir(
                        parents=True,
                        exist_ok=True,
                    )
                    shutil.copy(file, destination_file)
                    break
        else:
            children.append(file)

    for child in children:
        save_files(
            child,
            output_dir,
            to_save=to_save,
            ending=ending,
            top_level=False,
            checkpoint_index=checkpoint_index,
        )


class FileSystemManager:
    """A context manager for saving and cleaning up files."""

    def __init__(
        self,
        working_dir: Path,
        output_dir: Path,
        to_clean_once: list[str],
        to_save_once: list[str],
        original_hydra_dir: Path,
        reuse_output_dir: bool,
        file_limit: int | None = None,
    ) -> None:
        """Initialize the context manager.

        Parameters
        ----------
        working_dir : Path
            The working directory.
        output_dir : Path
            The output directory.
        to_clean_once : List[str]
            The tokens to clean once.
        to_save_once : List[str]
            The tokens to save once.
        original_hydra_dir : Path
            The original hydra directory.
            For copying the hydra directory to the working directory.
        reuse_output_dir : bool
            Whether to reuse the output directory.
        file_limit : Optional[int]
            The maximal number of files to search.
            If None, then there is no limit.

        Returns
        -------
            None
        """
        self.to_clean_once = to_clean_once
        self.working_dir = working_dir
        self.output_dir = output_dir
        self.to_save_once = to_save_once
        self.original_hydra_dir = original_hydra_dir
        self.reuse_output_dir = reuse_output_dir
        self.checkpoint_index = get_checkpoint_index(
            self.output_dir,
            file_limit,
        )

    def get_save_files_every_round(
        self,
        to_save: list[str],
        save_frequency: int,
    ) -> Callable[[int], None]:
        """Get a function that saves files every save_frequency rounds.

        Parameters
        ----------
        to_save : List[str]
            The tokens to save.
        save_frequency : int
            The frequency to save.

        Returns
        -------
        Callable[[int], None]
            The function that saves the files.
        """

        def save_files_round(cur_round: int) -> None:
            if cur_round % save_frequency == 0:
                save_files(
                    self.working_dir,
                    self.output_dir,
                    to_save=to_save,
                    ending=cur_round,
                    checkpoint_index=self.checkpoint_index,
                )

        return save_files_round

    def __enter__(self) -> "FileSystemManager":
        """Initialize the context manager and cleanup."""
        log(
            logging.INFO,
            f"Pre-cleaning {self.to_clean_once}",
        )
        cleanup(self.working_dir, self.to_clean_once)

        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_value: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        """Cleanup the files."""
        log(logging.INFO, f"Saving {self.to_save_once}")

        # Copy the hydra directory to the working directory
        # so that multiple runs can be ran
        # in the same output directory and configs versioned
        hydra_dir = self.working_dir / ".hydra"

        shutil.copytree(
            str(self.original_hydra_dir / ".hydra"),
            str(object=hydra_dir),
            dirs_exist_ok=True,
        )

        # Move main.log to the working directory
        main_log = self.original_hydra_dir / "main.log"
        shutil.copy2(
            str(main_log),
            str(self.working_dir / "main.log"),
        )
        save_files(
            self.working_dir,
            self.output_dir,
            to_save=self.to_save_once,
            checkpoint_index=self.checkpoint_index,
        )
        log(
            logging.INFO,
            f"Post-cleaning {self.to_clean_once}",
        )
        cleanup(
            self.working_dir,
            to_clean=self.to_clean_once,
        )
