#!/bin/bash
#SBATCH --cpus-per-task=5
#SBATCH --gres=gpu:rtx2080:1

cd /nfs-share/ag2411/project/fl_powerpropagation
poetry run python -m project.main --config-name=cluster_cifar_zerofl task.alpha=1.0 task.sparsity=0.9 task.fit_config.run_config.learning_rate=0.3 task.fit_config.run_config.final_learning_rate=0.03 fed.num_rounds=700 strategy=fedavgNZ
# poetry run python -m project.main --config-name=cluster_cifar_swat task.alpha=1.0 task.sparsity=0.0
