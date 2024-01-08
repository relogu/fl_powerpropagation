#!/bin/bash
#SBATCH --partition=low_priority
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:rtx2080:1

poetry run python -m project.main --config-name=cluster_cifar_swat
