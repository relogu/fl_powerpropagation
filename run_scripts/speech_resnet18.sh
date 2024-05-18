#!/bin/bash
#SBATCH --cpus-per-task=5
#SBATCH --gres=gpu:rtx2080:1

cd /nfs-share/ag2411/project/fl_powerpropagation

poetry run python -m project.main --config-name=cluster_speech_resnet18 task.alpha=1.0 task.sparsity=0.0 task.fit_config.run_config.learning_rate=0.5 task.fit_config.run_config.final_learning_rate=0.01 dataset.lda_alpha=1.0
