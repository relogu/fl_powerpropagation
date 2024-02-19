"""The default client implementation.

Make sure the model and dataset are not loaded before the fit function.
"""

import logging
import math
from pathlib import Path
import pickle


import flwr as fl
from flwr.common import NDArrays
from flwr.common.logger import log
import numpy as np
from pydantic import BaseModel
from torch import nn

from project.fed.utils.utils import (
    generic_get_parameters,
    generic_set_parameters,
    print_nonzeros,
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
        config.run_config["curr_round"] = config.extra["curr_round"]

        # print(f"[client_{self.cid}] current_round: ", config.extra["curr_round"])

        # Check, from the config, if the mask has to be used
        if config.extra["mask"]:
            mask_path = self.working_dir / f"mask_{self.cid}.pickle"
            if mask_path.exists():
                with open(mask_path, "rb") as f:
                    mask = pickle.load(f)
                noise = [
                    np.random.rand(*param.shape) < config.extra["noise"] * (1 - m)
                    for param, m in zip(parameters, mask, strict=True)
                ]
                # Apply the mask and the noise to the parameters
                parameters = [
                    param * (m + n)
                    for param, m, n in zip(parameters, mask, noise, strict=True)
                ]

        self.net = self.set_parameters(
            parameters,
            config.net_config,
        )

        trainloader = self.dataloader_gen(
            self.cid,
            False,
            config.dataloader_config,
        )

        tot_rounds = 700

        # FROM ZEROFL
        def update_learing_rate(
            inittial_value: float,
            final_value: float,
            curr_round: int,
            total_rounds: int = tot_rounds,
        ) -> float:
            """Update the learning rate using the exponential decay."""
            ratio = final_value / inittial_value
            log_ratio = math.log(ratio)
            exponential_term = (curr_round / total_rounds) * log_ratio
            eta_t = inittial_value * math.exp(exponential_term)
            return eta_t

        # config.run_config["learning_rate"] = _interpolate_initial_final_value(
        config.run_config["learning_rate"] = update_learing_rate(
            inittial_value=config.run_config["learning_rate"],
            final_value=config.run_config["final_learning_rate"],
            curr_round=config.run_config["curr_round"],
            total_rounds=tot_rounds,
            # total_rounds=config.extra["total_rounds"]
        )

        config.run_config["cid"] = self.cid

        log(
            logging.INFO,
            f"[client_{self.cid}] lr: {config.run_config['learning_rate']}",
        )

        def _changing_sparsity(net: nn.Module, sparsity: float) -> None:
            """Change the sparsity of the SWAT layers."""
            for module in net.modules():
                if hasattr(module, "sparsity"):
                    module.sparsity = sparsity

        def _changing_alpha(net: nn.Module, alpha: float) -> None:
            """Change the alpha of the SWAT layers."""
            for module in net.modules():
                if hasattr(module, "alpha"):
                    module.alpha = alpha

        # if config.run_config["curr_round"] != 1:
        #     self.net.apply(lambda x: _changing_sparsity(x, 0.0))

        # print(f"[client_{self.cid}] config.run_config: ", config.run_config)

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
            mask = [param != 0 for param in trained_parameters]
            # Save the mask in the output dir
            mask_path = self.working_dir / f"mask_{self.cid}.pickle"
            with open(mask_path, "wb") as fw:
                pickle.dump(mask, fw)

        metrics["learning_rate"] = config.run_config["learning_rate"]

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

        self.net = self.set_parameters(
            parameters,
            config.net_config,
        )
        sparsity = print_nonzeros(
            self.net, f"[client_{self.cid}] before first evaluation:"
        )

        # layer_sparsity = get_layer_sparsity(self.net)
        # print(f"[client_{self.cid}] layer sparsity: ", layer_sparsity)

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

        # Check, from the config, if the mask has to be used
        if config.extra["mask"]:
            mask_path = self.working_dir / f"mask_{self.cid}.pickle"
            if mask_path.exists():
                with open(mask_path, "rb") as f:
                    mask = pickle.load(f)
                # Apply the to the parameters
                parameters = [
                    param * m for param, m in zip(parameters, mask, strict=True)
                ]

        self.net = self.set_parameters(
            parameters,
            config.net_config,
        )

        metrics["sparsity"] = sparsity
        # metrics["layer_sparsity"] = layer_sparsity

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
