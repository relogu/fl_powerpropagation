#!/bin/bash
#SBATCH --cpus-per-task=5
#SBATCH --gres=gpu:rtx2080:1

cd /nfs-share/ag2411/project/fl_powerpropagation

poetry run python -m project.main --config-name=cluster_cifar_powerprop_a1
poetry run python -m project.main --config-name=cluster_cifar_powerprop_a2
poetry run python -m project.main --config-name=cluster_cifar_powerprop_a3
poetry run python -m project.main --config-name=cluster_cifar_powerprop_a4

