#!/bin/bash
#SBATCH --cpus-per-task=5
#SBATCH --gres=gpu:rtx2080:1

cd /nfs-share/ag2411/project/fl_powerpropagation 

poetry run python -m project.main --config-name=cluster_cifar_powerprop_pruning task.alpha=1.0 task.sparsity=0.995 task.fit_config.run_config.learning_rate=0.5 task.fit_config.run_config.final_learning_rate=0.05 fed.num_rounds=700 strategy=fedavg dataset.lda_alpha=1000
# poetry run python -m project.main --config-name=cluster_cifar_powerprop_pruning task.alpha=1.0000001 task.sparsity=0.9 task.fit_config.run_config.learning_rate=0.1 task.fit_config.run_config.final_learning_rate=0.01 fed.num_rounds=200 strategy=fedavgNZ
# poetry run python -m project.main --config-name=cluster_cifar_powerprop_pruning strategy=fedavgNZ task.fit_config.extra.noise=0 task.fit_config.run_config.learning_rate=0.01 task.sparsity=0.7
# poetry run python -m project.main --config-name=cluster_cifar_powerprop_pruning strategy=fedavg task.alpha=1.25 task.fit_config.extra.mask=true task.fit_config.extra.noise=0.1 task.fit_config.run_config.learning_rate=0.3 task.sparsity=0.9