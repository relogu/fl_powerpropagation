from typing import Dict, Tuple
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import instantiate, call
import wandb
import flwr
from flwr.server.app import ServerConfig
from flwr.common import Scalar, NDArrays
import wandb_utils
from simple_client import CifarClient
from utils import get_device

import tensorboardX as tensorboard

@hydra.main(version_base=None, config_path="./conf", config_name="config")
def my_app(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    
    dataset_root_dir = cfg.dataset.local_root_dir if cfg.is_local else cfg.dataset.root_dir
    
    def get_config_fn(server_round: int):
        return {
            "partitions_root": dataset_root_dir+f'/lda/{cfg.fl_setting.num_total_clients}/{cfg.fl_setting.lda_concentration:.2f}/',
            "batch_size": cfg.fl_setting.batch_size,
            "learning_rate": cfg.fl_setting.lr,
            "epochs": cfg.fl_setting.local_epochs,
        }
    print(get_config_fn(0))
    
    
    def evaluate_fn(server_round: int, parameters: NDArrays, config: Dict[str, Scalar] = {}) -> Tuple[float, Dict[str, Scalar]]:
        client = CifarClient(
            client_id=0,
            init_model_fn=call(cfg.model.init_model_fn),
        )
        net = client.set_parameters(net=None, parameters=parameters)
        def get_test_loop(*args, **kwargs):
            return call(cfg.fl_setting.test_loop, *args, **kwargs)
        return call(
            cfg.fl_setting.evaluate_fn,
            partition_root=Path(get_config_fn(server_round)['partitions_root']),
            net=net,
            device=get_device(),
            batch_size=get_config_fn(server_round)['batch_size'],
            test_loop=get_test_loop,
        )
        

    strategy = instantiate(
        cfg.strategy.init,
        fraction_fit=(float(cfg.fl_setting.num_train_clients_per_round) / cfg.fl_setting.num_total_clients),
        fraction_evaluate=(
            float(cfg.fl_setting.num_eval_clients_per_round) / cfg.fl_setting.num_total_clients
        ) if cfg.fl_setting.local_eval else 0.0,
        min_fit_clients=cfg.fl_setting.num_train_clients_per_round,
        min_evaluate_clients=cfg.fl_setting.num_eval_clients_per_round if cfg.fl_setting.local_eval else 0,
        min_available_clients=cfg.fl_setting.num_total_clients,
        on_fit_config_fn=get_config_fn,
        on_evaluate_config_fn=get_config_fn,
        evaluate_fn=evaluate_fn if cfg.fl_setting.centralised_eval else None,
        fit_metrics_aggregation_fn=wandb_utils.aggregate_weighted_average,
        evaluate_metrics_aggregation_fn=wandb_utils.aggregate_weighted_average,
    )
    print(strategy)
    
    server = instantiate(
        cfg.server.init,
        strategy=strategy,
        resume=cfg.wandb.resume,
    )
    print(server)
    
    # Define client resources and ray configs
    client_resources = {
        "num_cpus": cfg.ray_config.cpus_per_client,
        "num_gpus": 0 if cfg.is_local else cfg.ray_config.gpus_per_client,
    }
    print(client_resources)
    ray_config = {"include_dashboard": cfg.ray_config.include_dashboard}
    print(ray_config)
    
    # Download the centralised dataset
    call(
        cfg.dataset.load_centralised,
        root_dir=dataset_root_dir,
    )
    # Create and store the partitions
    paths = call(
        cfg.dataset.create_lda_partitions,
        root_dir=dataset_root_dir,
        num_partitions=cfg.fl_setting.num_total_clients,
        concentration=cfg.fl_setting.lda_concentration,
    )
    
    # Get client_fn and start simulation
    def client_fn(client_id: str) -> CifarClient:
        client = CifarClient(
            client_id=client_id,
            init_model_fn=call(cfg.model.init_model_fn),
        )
        if cfg.client_training_fn.training_fn is not None:
            client._train_loop = call(cfg.client_training_fn.training_fn).__get__(client, CifarClient)
        return client
    print(call(cfg.client_training_fn.training_fn))
        
    if cfg.is_simulation:
        # Start a new wandb run to track this script
        wandb.init(
            # Set the wandb project where this run will be logged
            project=cfg.wandb.project,
            resume=cfg.wandb.resume,
            id=cfg.wandb.id if cfg.wandb.id is not None else wandb.util.generate_id(),
            settings=wandb.Settings(start_method="thread"),
            config=OmegaConf.to_container(
                cfg, resolve=True, throw_on_missing=True
            ),
            group=cfg.client_training_fn.name,
        )
        print(f"Weights&Biases run is named {wandb.run.name} with id {wandb.run.id}")
        # Start the simulation
        hist = flwr.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=cfg.fl_setting.num_total_clients,
            client_resources=client_resources,
            config=ServerConfig(num_rounds=cfg.num_rounds),
            server=server,
            ray_init_args=ray_config,
        )
        print(hist)
        wandb.finish()

if __name__ == "__main__":
    # srun -w mauao -c 11 --gres=gpu:1 --partition=interactive python main.py num_rounds=1 is_simulation=False
    # srun -w mauao -c 11 --gres=gpu:1 --partition=interactive python main.py num_rounds=1 is_simulation=False model=powerprop
    # srun -w ngongotaha -c 8 --gres=gpu:1 --partition=interactive python main.py num_rounds=1 is_simulation=True model=powerprop ray_config.gpus_per_client=0.25
    # srun -w mauao -c 11 --gres=gpu:1 --partition=interactive python main.py num_rounds=1 is_simulation=True model=powerprop client_training_fn=pruning
    # srun -w mauao -c 11 --gres=gpu:1 --partition=interactive python main.py num_rounds=1 is_simulation=True model=powerprop client_training_fn=iterative_pruning
    my_app()