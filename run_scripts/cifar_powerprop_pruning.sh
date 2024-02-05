#!/bin/bash
#SBATCH --cpus-per-task=5
#SBATCH --gres=gpu:rtx2080:1

cd /nfs-share/ag2411/project/fl_powerpropagation
# poetry run python -m project.main --config-name=cluster_cifar_powerprop_pruning
# poetry run python -m project.main --config-name=cluster_cifar_powerprop_pruning strategy=fedavgNZ task.fit_config.extra.noise=0 task.fit_config.run_config.learning_rate=0.01 task.sparsity=0.7
poetry run python -m project.main --config-name=cluster_cifar_powerprop_pruning strategy=fedavgNZ task.alpha=1.25 task.fit_config.extra.mask=true task.fit_config.extra.noise=0.1 task.fit_config.run_config.learning_rate=0.2 task.sparsity=0.9