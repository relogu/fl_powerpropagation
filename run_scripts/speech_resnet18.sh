#!/bin/bash
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:rtx2080:1

cd /nfs-share/ag2411/project/fl_powerpropagation

poetry run python -m project.main --config-name=cluster_speech_resnet18 task.model_and_data=SPEECH_RESNET18 task.alpha=1.0 task.sparsity=0.0 task.mask=0.0 task.fit_config.dataloader_config.batch_size=32 task.fit_config.run_config.learning_rate=0.1 task.fit_config.run_config.final_learning_rate=0.001 dataset.lda_alpha=1.0 fed.num_rounds=500 
# strategy=fedavgNZ


# model_and_data: SPEECH_RESNET18
# model_and_data: SPEECH_PP
# model_and_data: SPEECH_PPSWAT
# model_and_data: SPEECH_ZERO