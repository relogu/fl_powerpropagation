#!/bin/bash
#SBATCH --cpus-per-task=5
#SBATCH --gres=gpu:rtx2080:1

cd /nfs-share/ag2411/project/fl_powerpropagation
# poetry run wandb agent fl_powerprop/sweeptry/dsru04xa
# poetry run wandb agent fl_powerprop/sweeptry/brnnq64c
poetry run wandb agent fl_powerprop/sweeptry/jgtn0hiv