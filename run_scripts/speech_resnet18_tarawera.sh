#!/bin/bash
#SBATCH -w gpu-vm
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --partition=debug

#! Adding OpenSSL v1.1.1
export LD_LIBRARY_PATH=/opt/openssl-1.1.1o${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

#! Adding CUDA 12.4 to the PATH environment variables
export PATH=/usr/local/cuda-12.4/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

cd /nfs-share/ag2411/project/fl_powerpropagation

# model_and_data: SPEECH_RESNET18
# model_and_data: SPEECH_PP
# model_and_data: SPEECH_PPSWAT
# model_and_data: SPEECH_ZERO

# train_structure: SPEECH_RESNET18_FLASH

poetry run python -m project.main --config-name=cluster_speech_resnet18 fed.seed=5378 task.model_and_data=SPEECH_RESNET18 task.alpha=1.0 task.sparsity=0.99 task.mask=0.0 task.fit_config.dataloader_config.batch_size=16 task.fit_config.run_config.learning_rate=0.5 task.fit_config.run_config.final_learning_rate=0.001 dataset.lda_alpha=1.0 fed.num_rounds=500 wandb.setup.project=speech_tarawera fed.gpus_per_client=0.1 fed.cpus_per_client=2
# poetry run python -m project.main --config-name=cluster_speech_resnet18 fed.seed=5378 task.model_and_data=SPEECH_RESNET18 task.alpha=1.0 task.sparsity=0.999 task.mask=0.0 task.fit_config.dataloader_config.batch_size=16 task.fit_config.run_config.learning_rate=0.5 task.fit_config.run_config.final_learning_rate=0.001 dataset.lda_alpha=1.0 fed.num_rounds=500 wandb.setup.project=speech_tarawera fed.gpus_per_client=0.1 fed.cpus_per_client=2
