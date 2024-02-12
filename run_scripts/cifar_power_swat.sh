#!/bin/bash
#SBATCH --cpus-per-task=5
#SBATCH --gres=gpu:rtx2080:1

cd /nfs-share/ag2411/project/fl_powerpropagation
poetry run python -m project.main --config-name=cluster_cifar_power_swat task.alpha=1.25 task.sparsity=0.99 task.fit_config.run_config.learning_rate=0.5 task.fit_config.run_config.final_learning_rate=0.05 fed.num_rounds=700 strategy=fedavg dataset.lda_alpha=0.1
# poetry run python -m project.main --config-name=cluster_cifar_swat task.alpha=1.0 task.sparsity=0.0
