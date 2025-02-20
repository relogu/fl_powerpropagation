# Copyright 2020 Flower Labs GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Federated Averaging (FedAvg) [McMahan et al., 2016] strategy.

Paper: arxiv.org/abs/1602.05629
"""


from logging import WARNING
from pathlib import Path
from typing import Optional, Union
from collections.abc import Callable

from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

from flwr.server.strategy.aggregate import weighted_loss_avg
from flwr.server.strategy.strategy import Strategy

from functools import reduce

import numpy as np

WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW = """
Setting `min_available_clients` lower than `min_fit_clients` or
`min_evaluate_clients` can cause the server to fail when there are too few clients
connected to the server. `min_available_clients` must be set to a value larger
than or equal to the values of `min_fit_clients` and `min_evaluate_clients`.
"""


class CidWindowCriterion:
    """Custom criterion to select clients within a window."""

    def __init__(self, upper_bound: int = 100, lower_bound: int = 0):
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound

    def select(self, client: ClientProxy, num_client: int = 100) -> bool:
        # Assuming the client's cid is an integer or can be converted to an integer
        return (
            int(client.cid) < self.upper_bound and int(client.cid) >= self.lower_bound
        )


def original_aggregate(results: list[tuple[NDArrays, int]]) -> NDArrays:
    """Compute weighted average."""
    # Calculate the total number of examples used during training
    num_examples_total = sum([num_examples for _, num_examples in results])

    # Create a list of weights, each multiplied by the related number of examples
    weighted_weights = [
        [layer * num_examples for layer in weights] for weights, num_examples in results
    ]

    # Compute average weights of each layer
    weights_prime: NDArrays = [
        reduce(np.add, layer_updates) / num_examples_total
        for layer_updates in zip(*weighted_weights)
    ]
    return weights_prime


def old_aggregate(results: list[tuple[NDArrays, int]]) -> NDArrays:
    """Compute weighted average for non-zero weights."""
    # Calculate the total number of examples used during training
    num_examples_total = sum([num_examples for _, num_examples in results])

    # Create a list of weights, each multiplied by the related number of examples
    weighted_weights = [
        [
            (layer * num_examples).astype(bool) * layer for layer in weights
        ]  # Modify this line
        for weights, num_examples in results
    ]

    # Compute average weights of each layer
    weights_prime: NDArrays = [
        np.divide(
            reduce(np.add, layer_updates),
            num_examples_total,
            where=(num_examples_total > 0),
        )  # Modify this line
        for layer_updates in zip(*weighted_weights)
    ]
    return weights_prime


def aggregate(results: list[tuple[NDArrays, int]]) -> NDArrays:
    """Compute weighted average for non-zero weights."""
    # Calculate the total number of examples used during training
    num_examples_total = sum([num_examples for _, num_examples in results])

    # Create a list of weights, each multiplied by the related number of examples
    weighted_weights = [
        [(layer * num_examples, layer != 0) for layer in weights]
        for weights, num_examples in results
    ]

    # Compute average weights of each layer
    weights_prime: NDArrays = [
        np.divide(
            np.sum([mask * layer for layer, mask in layer_updates], axis=0),
            num_examples_total,
            where=(num_examples_total > 0),
        )
        for layer_updates in zip(*weighted_weights)
    ]
    return weights_prime


# pylint: disable=line-too-long
class FedAvgHetero(Strategy):
    """Federated Averaging strategy.

    Implementation based on https://arxiv.org/abs/1602.05629

    Parameters
    ----------
    fraction_fit : float, optional
        Fraction of clients used during training. In case `min_fit_clients`
        is larger than `fraction_fit * available_clients`, `min_fit_clients`
        will still be sampled. Defaults to 1.0.
    fraction_evaluate : float, optional
        Fraction of clients used during validation. In case `min_evaluate_clients`
        is larger than `fraction_evaluate * available_clients`,
        `min_evaluate_clients` will still be sampled. Defaults to 1.0.
    min_fit_clients : int, optional
        Minimum number of clients used during training. Defaults to 2.
    min_evaluate_clients : int, optional
        Minimum number of clients used during validation. Defaults to 2.
    min_available_clients : int, optional
        Minimum number of total clients in the system. Defaults to 2.
    evaluate_fn : Optional[Callable[[int, NDArrays, Dict[str, Scalar]],
                    Optional[Tuple[float, Dict[str, Scalar]]]]]
        Optional function used for validation. Defaults to None.
    on_fit_config_fn : Callable[[int], Dict[str, Scalar]], optional
        Function used to configure training. Defaults to None.
    on_evaluate_config_fn : Callable[[int], Dict[str, Scalar]], optional
        Function used to configure validation. Defaults to None.
    accept_failures : bool, optional
        Whether or not accept rounds containing failures. Defaults to True.
    initial_parameters : Parameters, optional
        Initial global model parameters.
    fit_metrics_aggregation_fn : Optional[MetricsAggregationFn]
        Metrics aggregation function, optional.
    evaluate_metrics_aggregation_fn : Optional[MetricsAggregationFn]
        Metrics aggregation function, optional.
    """

    # pylint: disable=too-many-arguments,too-many-instance-attributes, line-too-long
    def __init__(
        self,
        *,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[
            Callable[
                [int, NDArrays, dict[str, Scalar]],
                Optional[tuple[float | dict[str, Scalar]]],
            ]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        working_dir: Path,
    ) -> None:
        super().__init__()

        if (
            min_fit_clients > min_available_clients
            or min_evaluate_clients > min_available_clients
        ):
            log(WARNING, WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW)

        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.evaluate_fn = evaluate_fn
        self.on_fit_config_fn = on_fit_config_fn
        self.on_evaluate_config_fn = on_evaluate_config_fn
        self.accept_failures = accept_failures
        self.initial_parameters = initial_parameters
        self.fit_metrics_aggregation_fn = fit_metrics_aggregation_fn
        self.evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn
        # Hetero specific
        self.working_dir = working_dir
        self.bounds = [(0, 40), (40, 70), (70, 100)]
        self.sparsities: list[float] = []

    def __repr__(self) -> str:
        """Compute a string representation of the strategy."""
        rep = f"FedAvg(accept_failures={self.accept_failures})"
        return rep

    def num_fit_clients(self, num_available_clients: int) -> tuple[int, int]:
        """Return the sample size and the required number of available clients."""
        num_clients = int(num_available_clients * self.fraction_fit)
        return max(num_clients, self.min_fit_clients), self.min_available_clients

    def num_evaluation_clients(self, num_available_clients: int) -> tuple[int, int]:
        """Use a fraction of available clients for evaluation."""
        num_clients = int(num_available_clients * self.fraction_evaluate)
        return max(num_clients, self.min_evaluate_clients), self.min_available_clients

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize global model parameters."""
        initial_parameters = self.initial_parameters
        self.initial_parameters = None  # Don't keep initial parameters in memory
        return initial_parameters

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[tuple[float, dict[str, Scalar]]]:
        """Evaluate model parameters using an evaluation function."""
        if self.evaluate_fn is None:
            # No evaluation function provided
            return None
        parameters_ndarrays = parameters_to_ndarrays(parameters)
        eval_res = self.evaluate_fn(server_round, parameters_ndarrays, {})
        if eval_res is None:
            return None
        loss, metrics = eval_res  # type: ignore[misc]
        return loss, metrics  # type: ignore[has-type]

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> list[tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)

        # to-do: in flash, the mask must be applied to the parameters
        fit_ins = FitIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )

        # to-do: - the clien must be sampled from different groups for differetn density
        cluster_clients: list = [[] for _ in range(3)]
        # remaining_clients = self.num_clients % 4
        # Sample clients for each cluster
        for idx, (lower_bound, upper_bound) in enumerate(self.bounds):
            num_clients = int((upper_bound - lower_bound) / 10)
            # Sample clients within the specified bounds
            cluster_clients[idx] = client_manager.sample(
                num_clients=num_clients,
                min_num_clients=num_clients,
                criterion=CidWindowCriterion(  # type: ignore[arg-type]
                    upper_bound=upper_bound, lower_bound=lower_bound
                ),
            )
        # aggregate all clients
        clients = reduce(lambda x, y: x + y, cluster_clients)
        # Return client/config pairs
        return [(client, fit_ins) for client in clients]

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> list[tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        # Do not configure federated evaluation if fraction eval is 0.
        if self.fraction_evaluate == 0.0:
            return []

        # Parameters and config
        config = {}
        if self.on_evaluate_config_fn is not None:
            # Custom evaluation config function provided
            config = self.on_evaluate_config_fn(server_round)

        # to-do: in flash, the mask must be applied to the parameters
        evaluate_ins = EvaluateIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )

        # to-do: - the clien must be sampled from different groups for differetn density
        cluster_clients: list = [[] for _ in range(3)]
        # remaining_clients = self.num_clients % 4
        # Sample clients for each cluster
        for idx, (lower_bound, upper_bound) in enumerate(self.bounds):
            num_clients = int((upper_bound - lower_bound) / 10)
            # Sample clients within the specified bounds
            cluster_clients[idx] = client_manager.sample(
                num_clients=num_clients,
                min_num_clients=num_clients,
                criterion=CidWindowCriterion(  # type: ignore[arg-type]
                    upper_bound=upper_bound, lower_bound=lower_bound
                ),
            )
        # aggregate all clients
        clients = reduce(lambda x, y: x + y, cluster_clients)
        # Return client/config pairs
        return [(client, evaluate_ins) for client in clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[Union[tuple[ClientProxy, FitRes], BaseException]],
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Convert results
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        parameters_aggregated = ndarrays_to_parameters(aggregate(weights_results))

        # to-do: - in flash the masks for the models must be created here, 1st round

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated

    def aggregate_evaluate(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, EvaluateRes]],
        failures: list[Union[tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> tuple[Optional[float], dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Aggregate loss
        loss_aggregated = weighted_loss_avg([
            (evaluate_res.num_examples, evaluate_res.loss)
            for _, evaluate_res in results
        ])

        # to-do: - the accuracy for the different models density must be aggregated separately

        # Aggregate custom metrics if aggregation fn was provided
        # metrics_aggregated = {}
        # if self.evaluate_metrics_aggregation_fn:
        #     eval_metrics = [(res.num_examples, res.metrics) for _, res in results]
        #     metrics_aggregated = self.evaluate_metrics_aggregation_fn(eval_metrics)

        #     cluster_bounds = [(0, 40), (40, 70), (70, 100)]
        #     clusters_accuracy = [[] for _ in range(len(cluster_bounds))]

        #     for _, metrics in eval_metrics:
        #         for idx, cluster in enumerate(cluster_bounds):
        #             if int(metrics["cid"]) in range(cluster[0], cluster[1]):
        #                 clusters_accuracy[idx].append(metrics["test_accuracy"])

        #     for idx, cluster_accuracy in enumerate(clusters_accuracy):
        #         if cluster_accuracy:
        #             metrics_aggregated[f"cluster_{idx}_accuracy"] = sum(cluster_accuracy) / len(cluster_accuracy)
        # elif server_round == 1:  # Only log this warning once
        #     log(WARNING, "No evaluate_metrics_aggregation_fn provided")

        # (CLASSIC) Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.evaluate_metrics_aggregation_fn:
            eval_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.evaluate_metrics_aggregation_fn(eval_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No evaluate_metrics_aggregation_fn provided")

        return loss_aggregated, metrics_aggregated
