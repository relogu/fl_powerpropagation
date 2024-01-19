"""The default client implementation.

Make sure the model and dataset are not loaded before the fit function.
"""

from pathlib import Path
import pickle

import flwr as fl
from flwr.common import NDArrays
import numpy as np
from pydantic import BaseModel
from torch import nn

from project.fed.utils.utils import (
    generic_get_parameters,
    generic_set_parameters,
)
from project.types.common import (
    ClientDataloaderGen,
    ClientGen,
    EvalRes,
    FitRes,
    NetGen,
    TestFunc,
    TrainFunc,
)
from project.utils.utils import obtain_device


class ClientConfig(BaseModel):
    """Fit/eval config, allows '.' member access and static checking.

    Used to check weather each component has its own independent config present. Each
    component should then use its own Pydantic model to validate its config. For
    anything extra, use the extra field as a simple dict.
    """

    # Instantiate model
    net_config: dict
    # Instantiate dataloader
    dataloader_config: dict
    # For train/test
    run_config: dict
    # Additional params used like a Dict
    extra: dict

    class Config:
        """Setting to allow any types, including library ones like torch.device."""

        arbitrary_types_allowed = True


class Client(fl.client.NumPyClient):
    """Virtual client for ray."""

    def __init__(
        self,
        cid: int | str,
        working_dir: Path,
        net_generator: NetGen,
        dataloader_gen: ClientDataloaderGen,
        train: TrainFunc,
        test: TestFunc,
    ) -> None:
        """Initialize the client.

        Only ever instantiate the model or load dataset
        inside fit/eval, never in init.

        Parameters
        ----------
        cid : int | str
            The client's ID.
        working_dir : Path
            The path to the working directory.
        net_generator : NetGen
            The network generator.
        dataloader_gen : ClientDataloaderGen
            The dataloader generator.
            Uses the client id to determine partition.

        Returns
        -------
        None
        """
        super().__init__()
        self.cid = cid
        self.net_generator = net_generator
        self.working_dir = working_dir
        self.net: nn.Module | None = None
        self.dataloader_gen = dataloader_gen
        self.train = train
        self.test = test

    def fit(
        self,
        parameters: NDArrays,
        _config: dict,
    ) -> FitRes:
        """Fit the model using the provided parameters.

        Only ever instantiate the model or load dataset
        inside fit, never in init.

        Parameters
        ----------
        parameters : NDArrays
            The parameters to use for training.
        _config : Dict
            The configuration for the training.
            Uses the pydantic model for static checking.

        Returns
        -------
        FitRes
            The parameters after training, the number of samples used and the metrics.
        """
        config: ClientConfig = ClientConfig(**_config)
        del _config

        config.run_config["device"] = obtain_device()

        # Check, from the config, if the mask has to be used
        if config.extra["mask"]:
            # mask_path = self.working_dir / f"mask_{self.cid}.npy"
            mask_path = self.working_dir / f"mask_{self.cid}.pickle"
            if mask_path.exists():
                # mask = np.load(mask_path)
                with open(mask_path, "rb") as f:
                    mask = pickle.load(f)
                # print(f"[clien_{self.cid}] Used Mask, saved in: ", mask_path)
                # print the number of non zero elements in the mask
                # print(f"[clien_{self.cid}] Number of non zero elements in the mask: ",
                #    np.count_nonzero(mask))
                # Create noise, random sampling (1 - mask), for each layer's parameters
                # np.random.normal(0, 1, param.shape) * (1 - m)
                # np.random.rand(*param.shape) < 0.5 * (1 - m)
                #
                noise = [
                    np.random.rand(*param.shape) < 0.5 * (1 - m)
                    for param, m in zip(parameters, mask, strict=True)
                ]
                # Apply the mask and the noise to the parameters
                parameters = [
                    param * (m + n)
                    for param, m, n in zip(parameters, mask, noise, strict=True)
                ]
                # parameters = [
                #     param * m
                #     for param, m in zip(parameters, mask, strict=True)
                # ]
        """
        # Alternative way to create the mask
         if config.extra["mask"]:
    mask_path = self.working_dir / f"mask_{self.cid}.pickle"
    if mask_path.exists():
        with open(mask_path, "rb") as f:
            mask = pickle.load(f)
    else:
        # Create a Bayesian mask using the Beta distribution
        mask = [beta.rvs(1, 1, size=param.shape) for param in parameters]

    # Create noise, random sampling (1 - mask), for each layer's parameters
    noise = [
        np.random.rand(*param.shape) < 0.5 * (1 - m)
        for param, m in zip(parameters, mask)
    ]

    # Apply the mask and the noise to the parameters
    parameters = [
        param * (m + n)
        for param, m, n in zip(parameters, mask, noise)
    ]
        """

        """
            if config.extra["mask"]:
    mask_path = self.working_dir / f"mask_{self.cid}.pickle"
    if mask_path.exists():
        with open(mask_path, "rb") as f:
            mask = pickle.load(f)
    else:
        # Initialize the mask to give more importance to the early and late layers
        mask = [
            np.ones_like(param)
            if i < len(parameters) // 4
            or i >= 3 * len(parameters) // 4
            else np.zeros_like(param) for i, param in enumerate(parameters)]

    # Create noise, random sampling (1 - mask), for each layer's parameters
    noise = [
        np.random.rand(*param.shape) < 0.5 * (1 - m)
        for param, m in zip(parameters, mask)
    ]

    # Apply the mask and the noise to the parameters
    parameters = [
        param * (m + n)
        for param, m, n in zip(parameters, mask, noise)
    ]

        """

        self.net = self.set_parameters(
            parameters,
            config.net_config,
        )

        trainloader = self.dataloader_gen(
            self.cid,
            False,
            config.dataloader_config,
        )

        num_samples, metrics = self.train(
            self.net,
            trainloader,
            config.run_config,
            self.working_dir,
        )

        trained_parameters = generic_get_parameters(self.net)

        # Check if the mask has been used
        if config.extra["mask"]:
            # Estract the mask from the parameters
            # mask = trained_parameters != 0
            mask = [param != 0 for param in parameters]
            # Save the mask in the output dir
            mask_path = self.working_dir / f"mask_{self.cid}.pickle"
            with open(mask_path, "wb") as f:
                pickle.dump(mask, f)
            # print(f"[clien_{self.cid}] Saving mask in: ", mask_path)

        return (
            trained_parameters,
            num_samples,
            metrics,
        )

    def evaluate(
        self,
        parameters: NDArrays,
        _config: dict,
    ) -> EvalRes:
        """Evaluate the model using the provided parameters.

        Only ever instantiate the model or load dataset
        inside eval, never in init.

        Parameters
        ----------
        parameters : NDArrays
            The parameters to use for evaluation.
        _config : Dict
            The configuration for the evaluation.
            Uses the pydantic model for static checking.

        Returns
        -------
        EvalRes
            The loss, the number of samples used and the metrics.
        """
        config: ClientConfig = ClientConfig(**_config)
        del _config

        config.run_config["device"] = obtain_device()

        # Check, from the config, if the mask has to be used
        if config.extra["mask"]:
            # mask_path = self.working_dir / f"mask_{self.cid}.npy"
            mask_path = self.working_dir / f"mask_{self.cid}.pickle"
            if mask_path.exists():
                with open(mask_path, "rb") as f:
                    mask = pickle.load(f)
                # print(f"[EVAL{self.cid}] Used Mask, saved in: ", mask_path)
                # Apply the to the parameters
                parameters = [
                    param * m for param, m in zip(parameters, mask, strict=True)
                ]

        self.net = self.set_parameters(
            parameters,
            config.net_config,
        )
        testloader = self.dataloader_gen(
            self.cid,
            True,
            config.dataloader_config,
        )
        loss, num_samples, metrics = self.test(
            self.net,
            testloader,
            config.run_config,
            self.working_dir,
        )
        return loss, num_samples, metrics

    def get_parameters(self, config: dict) -> NDArrays:
        """Obtain client parameters.

        If the network is currently none,generate a network using the net_generator.

        Parameters
        ----------
        config : Dict
            The configuration for the training.

        Returns
        -------
        NDArrays
            The parameters of the network.
        """
        if self.net is None:
            except_str: str = """Network is None.
                Call set_parameters first and
                do not use this template without a get_initial_parameters function.
            """
            raise ValueError(
                except_str,
            )

        return generic_get_parameters(self.net)

    def set_parameters(
        self,
        parameters: NDArrays,
        config: dict,
    ) -> nn.Module:
        """Set client parameters.

        First generated the network. Only call this in fit/eval.

        Parameters
        ----------
        parameters : NDArrays
            The parameters to set.
        config : Dict
            The configuration for the network generator.

        Returns
        -------
        nn.Module
            The network with the new parameters.
        """
        net = self.net_generator(config)
        generic_set_parameters(
            net,
            parameters,
            to_copy=False,
        )
        return net

    def __repr__(self) -> str:
        """Implement the string representation based on cid."""
        return f"Client(cid={self.cid})"

    def get_properties(self, config: dict) -> dict:
        """Implement how to get properties."""
        return {}


def get_client_generator(
    working_dir: Path,
    net_generator: NetGen,
    dataloader_gen: ClientDataloaderGen,
    train: TrainFunc,
    test: TestFunc,
) -> ClientGen:
    """Return a function which creates a new Client.

    Client has access to the working dir,
    can generate a network and can generate a dataloader.
    The client receives train and test functions with pre-defined APIs.

    Parameters
    ----------
    working_dir : Path
        The path to the working directory.
    net_generator : NetGen
        The network generator.
        Please respect the pydantic schema.
    dataloader_gen : ClientDataloaderGen
        The dataloader generator.
        Uses the client id to determine partition.
        Please respect the pydantic schema.
    train : TrainFunc
        The train function.
        Please respect the interface and pydantic schema.
    test : TestFunc
        The test function.
        Please respect the interface and pydantic schema.

    Returns
    -------
    ClientGen
        The function which creates a new Client.
    """

    def client_generator(cid: int | str) -> Client:
        """Return a new Client.

        Parameters
        ----------
        cid : int | str
            The client's ID.

        Returns
        -------
        Client
            The new Client.
        """
        return Client(
            cid,
            working_dir,
            net_generator,
            dataloader_gen,
            train,
            test,
        )

    return client_generator
