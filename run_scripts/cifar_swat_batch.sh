#!/bin/bash
#SBATCH --cpus-per-task=5
#SBATCH --gres=gpu:rtx2080:1

cd /nfs-share/ag2411/project/fl_powerpropagation
poetry run python -m project.main --config-name=cluster_cifar_swat_a2_s05
poetry run python -m project.main --config-name=cluster_cifar_swat_a2_s07
poetry run python -m project.main --config-name=cluster_cifar_swat_a2_s09
