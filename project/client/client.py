"""The default client implementation.

Make sure the model and dataset are not loaded before the fit function.
"""

import math
from pathlib import Path
import pickle


import flwr as fl
from flwr.common import NDArrays
from pydantic import BaseModel
from torch import nn
import torch
from torch.utils.data import DataLoader, TensorDataset

from project.fed.utils.utils import (
    generic_get_parameters,
    generic_set_parameters,
    get_nonzeros,
)

from project.types.common import (
    ClientDataloaderGen,
    ClientGen,
    EvalRes,
    FedDataloaderGen,
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


# FROM ZEROFL
def update_learing_rate(
    inittial_value: float,
    final_value: float,
    curr_round: int,
    total_rounds: int = 500,
) -> float:
    """Update the learning rate using the exponential decay."""
    ratio = final_value / inittial_value
    log_ratio = math.log(ratio)
    exponential_term = (curr_round / total_rounds) * log_ratio
    eta_t = inittial_value * math.exp(exponential_term)
    return eta_t


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
        fed_dataloader_gen: FedDataloaderGen,  # for the in-out local tests !?
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
        self.fed_dataloader_gen = fed_dataloader_gen

    def _evaluate_partition(self, dataloader: DataLoader, partition_name: str) -> dict:
        """Evaluate the model on a specific partition."""
        # self.net.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        # device = obtain_device()

        # with torch.no_grad():
        #     for data, targets in dataloader:
        #         data, targets = data.to(device), targets.to(device)
        #         outputs = self.net(data)
        #         loss = torch.nn.functional.cross_entropy(outputs, targets)
        #         total_loss += loss.item() * data.size(0)
        #         _, predicted = outputs.max(1)
        #         total += targets.size(0)
        #         correct += predicted.eq(targets).sum().item()

        #         # if partition_name == "out_local":
        #         #     log(logging.DEBUG, f"[client_{self.cid}] Out-local batch - "
        #         #         f"Predictions: {predicted.tolist()}, "
        #         #         f"Targets: {targets.tolist()}")

        accuracy = correct / total if total > 0 else 0.0
        avg_loss = total_loss / total if total > 0 else 0.0

        return {
            f"{partition_name}_loss": avg_loss,
            f"{partition_name}_accuracy": accuracy,
            f"{partition_name}_samples": total,
        }

    def _evaluate_in_out_local(
        self, trainloader: DataLoader, testloader: DataLoader
    ) -> tuple[dict, dict]:
        """Evaluate the model on in-local and out-local partitions."""
        train_classes = set()
        test_classes = set()
        for _, labels in trainloader:
            train_classes.update(labels.numpy())
        for _, labels in testloader:
            test_classes.update(labels.numpy())

        in_local_data, in_local_labels = [], []
        out_local_data, out_local_labels = [], []

        for data, labels in testloader:
            for i, label in enumerate(labels):
                if label.item() in train_classes:
                    in_local_data.append(data[i])
                    in_local_labels.append(label)
                else:
                    out_local_data.append(data[i])
                    out_local_labels.append(label)

        in_local_results = {
            "in_local_loss": 0.0,
            "in_local_accuracy": 0.0,
            "in_local_samples": len(in_local_data),
            "in_local_classes": len(train_classes.intersection(test_classes)),
        }

        out_local_results = {
            "out_local_loss": None,
            "out_local_accuracy": None,
            "out_local_samples": len(out_local_data),
            "out_local_classes": len(test_classes - train_classes),
        }

        if len(in_local_data) > 0:
            in_local_dataset = TensorDataset(
                torch.stack(in_local_data), torch.stack(in_local_labels)
            )
            in_local_loader = DataLoader(
                in_local_dataset, batch_size=testloader.batch_size
            )
            in_local_results.update(
                self._evaluate_partition(in_local_loader, "in_local")
            )

        if len(out_local_data) > 0:
            out_local_dataset = TensorDataset(
                torch.stack(out_local_data), torch.stack(out_local_labels)
            )
            out_local_loader = DataLoader(
                out_local_dataset, batch_size=testloader.batch_size
            )
            out_local_results.update(
                self._evaluate_partition(out_local_loader, "out_local")
            )

        return in_local_results, out_local_results

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
        # if config.extra["mask"]:
        #     mask_path = self.working_dir / f"mask_{self.cid}.pickle"
        #     if mask_path.exists():
        #         with open(mask_path, "rb") as f:
        #             mask = pickle.load(f)
        #         noise = [
        #             np.random.rand(*param.shape) < config.extra["noise"] * (1 - m)
        #             for param, m in zip(parameters, mask, strict=True)
        #         ]
        #         # Apply the mask and the noise to the parameters
        #         parameters = [
        #             param * (m + n)
        #             for param, m, n in zip(parameters, mask, noise, strict=True)
        #         ]

        # trained_parameters = generic_get_parameters(self.net)
        # if config.extra["mask"]:
        #     # Estract the mask from the parameters
        #     # mask = [param != 0 for param in trained_parameters]
        #     mask = [param != 0 for param in parameters]
        #     # Save the mask in the output dir
        #     mask_path = (
        #         self.working_dir / f"mask_{config.run_config['curr_round']}.pickle"
        #     )
        #     with open(mask_path, "wb") as fw:
        #         pickle.dump(mask, fw)

        self.net = self.set_parameters(
            parameters,
            config.net_config,
        )

        del parameters

        trainloader = self.dataloader_gen(
            self.cid,
            False,
            config.dataloader_config,
        )

        # tot_rounds = 1000

        # config.run_config["learning_rate"] = _interpolate_initial_final_value(
        config.run_config["learning_rate"] = update_learing_rate(
            inittial_value=config.run_config["learning_rate"],
            final_value=config.run_config["final_learning_rate"],
            curr_round=config.run_config["curr_round"],
            # total_rounds=config.run_config["total_rounds"],
            # total_rounds=config.extra["total_rounds"]
        )

        config.run_config["cid"] = self.cid

        # log(
        #     logging.INFO,
        #     f"[client_{self.cid}] lr: {config.run_config['learning_rate']}",
        # )

        # def _changing_sparsity(net: nn.Module, sparsity: float) -> None:
        #     """Change the sparsity of the SWAT layers."""
        #     for module in net.modules():
        #         if hasattr(module, "sparsity"):
        #             module.sparsity = sparsity

        # def _changing_alpha(net: nn.Module, alpha: float) -> None:
        #     """Change the alpha of the SWAT layers."""
        #     for module in net.modules():
        #         if hasattr(module, "alpha"):
        #             module.alpha = alpha

        # if config.run_config["curr_round"] != 1:
        #     self.net.apply(lambda x: _changing_sparsity(x, 0.0))

        # print(f"[client_{self.cid}] config.run_config: ", config.run_config)

        num_samples, metrics = self.train(
            self.net,
            trainloader,
            config.run_config,
            self.working_dir,
        )

        metrics["learning_rate"] = config.run_config["learning_rate"]

        updated_parameters = generic_get_parameters(self.net)

        # if config.extra["in_out_eval"]:
        #     # Post training evaluation
        #     testloader = self.fed_dataloader_gen(True, config.dataloader_config)
        #     server_results = self._evaluate_partition(testloader, "server")
        #     in_local_results, out_local_results = self._evaluate_in_out_local(
        #         trainloader, testloader
        #     )
        #     metrics.update(server_results)
        #     metrics.update(in_local_results)
        #     metrics.update(out_local_results)

        return (
            # trained_parameters,
            # generic_get_parameters(self.net),
            updated_parameters,
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
        config.run_config["curr_round"] = config.extra["curr_round"]

        self.net = self.set_parameters(
            parameters,
            config.net_config,
        )
        sparsity = get_nonzeros(self.net)

        testloader = self.dataloader_gen(
            self.cid,
            True,
            config.dataloader_config,
        )
        # start_time = time.time()
        loss, num_samples, metrics = self.test(
            self.net,
            testloader,
            config.run_config,
            self.working_dir,
        )

        # Saving the mask of the global model
        if config.extra["mask"]:
            # Estract the mask from the parameters
            mask = [param != 0 for param in parameters]
            # Save the mask in the output dir
            mask_path = (
                self.working_dir / f"mask_{config.run_config['curr_round']}.pickle"
            )
            with open(mask_path, "wb") as fw:
                # save the binary mask
                pickle.dump(mask, fw)

        self.net = self.set_parameters(
            parameters,
            config.net_config,
        )

        metrics["sparsity"] = sparsity

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
    fed_dataloader_gen: FedDataloaderGen,
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
            fed_dataloader_gen,
        )

    return client_generator
