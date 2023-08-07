import timeit
import numbers
from logging import INFO
from typing import Optional, List, Tuple, Dict
from collections import defaultdict
from pathlib import Path
import pickle
import wandb

from flwr.server import Server, History
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.strategy import FedAvg, Strategy

from flwr.common import Parameters

def aggregate_weighted_average(metrics: List[Tuple[int, dict]]) -> dict:
    """Generic function to combine results from multiple clients
    following training or evaluation.

    Args:
        metrics (List[Tuple[int, dict]]): collected clients metrics

    Returns:
        dict: result dictionary containing the aggregate of the metrics passed.
    """
    average_dict: dict = defaultdict(list)
    total_examples: int = 0
    for num_examples, metrics_dict in metrics:
        for key, val in metrics_dict.items():
            if isinstance(val, numbers.Number):
                average_dict[key].append((num_examples, val))  # type:ignore
        total_examples += num_examples
    return {
        key: {
            "avg": float(
                sum([num_examples * metr for num_examples, metr in val])
                / float(total_examples)
            ),
            "all": val,
        }
        for key, val in average_dict.items()
    }


def log_history_to_wandb(history: History):
    """Log the received `history` object to Weights and Biases."""
    wandb_dict = {
        # The losses in the history object are stored in the form of a list of tuples: (server_round, loss).
        'losses_distributed': history.losses_distributed[-1][1] if len(history.losses_distributed) > 0 else 0,
        'losses_centralized': history.losses_centralized[-1][1] if len(history.losses_centralized) > 0 else 0,
    }
    # NOTE: the `Strategy` has to aggregate the metrics through the `aggregate_fit` and
    # the `aggregate_evaluate`. The distributed metrics then contain the aggregated
    # metrics in the form of a list of tuples (server_round, aggregated_metric).
    # For better understanding the composition of the History object, see its
    # implementation.
    wandb_dict.update({
        f'{key}_distributed_fit': val[-1][1]
        # key -> metric name, val -> list of tuples (server_round, aggregated_metric)
        for key, val in history.metrics_distributed_fit.items()
    })
    wandb_dict.update({
        f'{key}_metrics_distributed': val[-1][1]
        # key -> metric name, val -> list of tuples (server_round, aggregated_metric)
        for key, val in history.metrics_distributed.items()
    })
    wandb_dict.update({
        f'{key}_metrics_centralized': val[-1][1]
        # key -> metric name, val -> list of tuples (server_round, aggregated_metric)
        for key, val in history.metrics_centralized.items()
    })
    # Log metrics to wandb
    wandb.log(wandb_dict)

def dump_history_to_files(history: History, root_path: str):
    """Dump the received `history` object to files in the `root_path`."""
    with open(Path(root_path, 'losses_distributed.pkl'), 'wb') as f:
        pickle.dump(history.losses_distributed, f)
    with open(Path(root_path, 'losses_centralized.pkl'), 'wb') as f:
        pickle.dump(history.losses_centralized, f)
    with open(Path(root_path, 'metrics_distributed_fit.pkl'), 'wb') as f:
        pickle.dump(history.metrics_distributed_fit, f)
    with open(Path(root_path, 'metrics_distributed.pkl'), 'wb') as f:
        pickle.dump(history.metrics_distributed, f)
    with open(Path(root_path, 'metrics_centralized.pkl'), 'wb') as f:
        pickle.dump(history.metrics_centralized, f)
    # Save files to wandb
    wandb.save('losses_distributed.pkl', policy="now")
    wandb.save('losses_centralized.pkl', policy="now")
    wandb.save('metrics_distributed_fit.pkl', policy="now")
    wandb.save('metrics_distributed.pkl', policy="now")
    wandb.save('metrics_centralized.pkl', policy="now")

def load_history_from_files(root_path: str):
    """Load an `history` object from files in the `root_path`."""
    history = History()
    # Restoring files from wandb
    wandb.restore('losses_distributed.pkl')
    wandb.restore('losses_centralized.pkl')
    wandb.restore('metrics_distributed_fit.pkl')
    wandb.restore('metrics_distributed.pkl')
    wandb.restore('metrics_centralized.pkl')
    with open(Path(root_path, 'losses_distributed.pkl'), 'rb') as f:
        history.losses_distributed = pickle.load(f)
    with open(Path(root_path, 'losses_centralized.pkl'), 'rb') as f:
        history.losses_centralized = pickle.load(f)
    with open(Path(root_path, 'metrics_distributed_fit.pkl'), 'rb') as f:
        history.metrics_distributed_fit = pickle.load(f)
    with open(Path(root_path, 'metrics_distributed.pkl'), 'rb') as f:
        history.metrics_distributed = pickle.load(f)
    with open(Path(root_path, 'metrics_centralized.pkl'), 'rb') as f:
        history.metrics_centralized = pickle.load(f)
    return history

def dump_server_state_to_file(server_state: Dict, root_path: str):
    """Dump the received `server_state` object to files in the `root_path`."""
    with open(Path(root_path, 'server_state.pkl'), 'wb') as f:
        pickle.dump(server_state, f)
    wandb.save('server_state.pkl', policy="now")

def load_server_state_from_file(root_path: str):
    """Load the `server_state` object from files in the `root_path`."""
    wandb.restore('server_state.pkl')
    with open(Path(root_path, 'server_state.pkl'), 'rb') as f:
        server_state = pickle.load(f)
    return server_state
        
    

class WandbServer(Server):
    """WandbServer extends the Flower Server to log metrics to Weights and Biases."""

    def __repr__(self) -> str:
        rep = f"WandbServer(client_manager={self._client_manager}, strategy={self.strategy}, strategy.evaluate_metrics_aggregation_fn={self.strategy.evaluate_metrics_aggregation_fn})"
        return rep
    
    # pylint: disable=too-many-locals
    def fit(self, num_rounds: int, timeout: Optional[float]) -> History:
        """Run federated averaging for a number of rounds."""
        history = History()

        # Initialize parameters
        log(INFO, "Initializing global parameters")
        self.parameters = self._get_initial_parameters(timeout=timeout)
        log(INFO, "Evaluating initial parameters")
        res = self.strategy.evaluate(0, parameters=self.parameters)
        if res is not None:
            log(
                INFO,
                "initial parameters (loss, other metrics): %s, %s",
                res[0],
                res[1],
            )
            history.add_loss_centralized(server_round=0, loss=res[0])
            history.add_metrics_centralized(server_round=0, metrics=res[1])

        # Run federated learning for num_rounds
        log(INFO, "FL starting")
        start_time = timeit.default_timer()

        for current_round in range(1, num_rounds + 1):
            # Train model and replace previous global model
            res_fit = self.fit_round(server_round=current_round, timeout=timeout)
            if res_fit:
                parameters_prime, fit_metrics, _ = res_fit  # fit_metrics_aggregated
                if parameters_prime:
                    self.parameters = parameters_prime
                history.add_metrics_distributed_fit(
                    server_round=current_round, metrics=fit_metrics
                )

            # Evaluate model using strategy implementation
            res_cen = self.strategy.evaluate(current_round, parameters=self.parameters)
            if res_cen is not None:
                loss_cen, metrics_cen = res_cen
                log(
                    INFO,
                    "fit progress: (%s, %s, %s, %s)",
                    current_round,
                    loss_cen,
                    metrics_cen,
                    timeit.default_timer() - start_time,
                )
                history.add_loss_centralized(server_round=current_round, loss=loss_cen)
                history.add_metrics_centralized(
                    server_round=current_round, metrics=metrics_cen
                )

            # Evaluate model on a sample of available clients
            res_fed = self.evaluate_round(server_round=current_round, timeout=timeout)
            if res_fed:
                loss_fed, evaluate_metrics_fed, _ = res_fed
                if loss_fed:
                    history.add_loss_distributed(
                        server_round=current_round, loss=loss_fed
                    )
                    history.add_metrics_distributed(
                        server_round=current_round, metrics=evaluate_metrics_fed
                    )
            # NOTE: Log to wandb
            log_history_to_wandb(history)

        # Bookkeeping
        end_time = timeit.default_timer()
        elapsed = end_time - start_time
        log(INFO, "FL finished in %s", elapsed)
        return history


class CheckpointWandbServer(Server):
    """CheckpointWandbServer extends the Flower Server to log metrics to Weights and Biases and checkpointing."""

    def __init__(
        self,
        *,
        client_manager: ClientManager,
        strategy: Optional[Strategy] = None,
        resume: bool = False,
        parameters: Parameters = None,
    ) -> None:
        self._client_manager: ClientManager = client_manager
        self.parameters: Parameters = Parameters(
            tensors=[], tensor_type="numpy.ndarray"
        ) if parameters is None else parameters
        self.strategy: Strategy = strategy if strategy is not None else FedAvg()
        self.max_workers: Optional[int] = None
        self.resume = resume

    def __repr__(self) -> str:
        rep = f"CheckpointWandbServer(client_manager={self._client_manager}, strategy={self.strategy}, strategy.evaluate_metrics_aggregation_fn={self.strategy.evaluate_metrics_aggregation_fn}, resume={self.resume})"
        return rep

    # pylint: disable=too-many-locals
    def fit(self, num_rounds: int, timeout: Optional[float]) -> History:
        """Run federated averaging for a number of rounds."""
        history = History()

        if not self.resume:
            # Initialize parameters
            log(INFO, "Initializing global parameters")
            self.parameters = self._get_initial_parameters(timeout=timeout)
            log(INFO, "Evaluating initial parameters")
            res = self.strategy.evaluate(0, parameters=self.parameters)
            if res is not None:
                log(
                    INFO,
                    "initial parameters (loss, other metrics): %s, %s",
                    res[0],
                    res[1],
                )
                history.add_loss_centralized(server_round=0, loss=res[0])
                history.add_metrics_centralized(server_round=0, metrics=res[1])

            # Run federated learning for num_rounds
            log(INFO, "FL starting")
            start_time = timeit.default_timer()
            start_round = 1
        else:
            log(INFO, "Resuming FL")
            # Load history from checkpoint
            history = load_history_from_files(wandb.run.dir)
            # Load server state from checkpoint
            server_state = load_server_state_from_file(wandb.run.dir)
            self.parameters = server_state['model_parameters']
            start_round = server_state['server_round']
            start_time = timeit.default_timer() + server_state['elapsed_time']
            
        for current_round in range(start_round, num_rounds + 1):
            # Train model and replace previous global model
            res_fit = self.fit_round(
                server_round=current_round,
                timeout=timeout,
            )
            if res_fit is not None:
                parameters_prime, fit_metrics, _ = res_fit  # fit_metrics_aggregated
                if parameters_prime:
                    self.parameters = parameters_prime
                history.add_metrics_distributed_fit(
                    server_round=current_round, metrics=fit_metrics
                )

            # Evaluate model using strategy implementation
            res_cen = self.strategy.evaluate(current_round, parameters=self.parameters)
            if res_cen is not None:
                loss_cen, metrics_cen = res_cen
                log(
                    INFO,
                    "fit progress: (%s, %s, %s, %s)",
                    current_round,
                    loss_cen,
                    metrics_cen,
                    timeit.default_timer() - start_time,
                )
                history.add_loss_centralized(server_round=current_round, loss=loss_cen)
                history.add_metrics_centralized(
                    server_round=current_round, metrics=metrics_cen
                )

            # Evaluate model on a sample of available clients
            res_fed = self.evaluate_round(server_round=current_round, timeout=timeout)
            if res_fed is not None:
                loss_fed, evaluate_metrics_fed, _ = res_fed
                if loss_fed is not None:
                    history.add_loss_distributed(
                        server_round=current_round, loss=loss_fed
                    )
                    history.add_metrics_distributed(
                        server_round=current_round, metrics=evaluate_metrics_fed
                    )
            # Log to wandb
            log_history_to_wandb(history)
            # Dump history to files in the wandb checkpoint directory
            dump_history_to_files(history, wandb.run.dir)
            # Save checkpoint
            server_state = {
                'server_round': current_round,
                'elapsed_time': timeit.default_timer() - start_time,
                'model_parameters': self.parameters,
            }
            dump_server_state_to_file(server_state, wandb.run.dir)

        # Bookkeeping
        end_time = timeit.default_timer()
        elapsed = end_time - start_time
        log(INFO, "FL finished in %s", elapsed)
        return history
