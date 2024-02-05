#!/bin/bash
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:rtx2080:1

cd /nfs-share/ag2411/project/fl_powerpropagation
poetry run wandb agent fl_powerprop/sweeptry/dsru04xa