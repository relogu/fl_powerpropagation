#!/bin/bash
# shellcheck disable=SC2206
# THIS FILE IS GENERATED BY AUTOMATION SCRIPT! PLEASE REFER TO ORIGINAL SCRIPT!
# THIS FILE IS A TEMPLATE AND IT SHOULD NOT BE DEPLOYED TO PRODUCTION!
#SBATCH -w ngongotaha
#SBATCH --gres=gpu:1
#SBATCH --job-name=fl_powerprop_swat
#SBATCH -c 8

# Load modules or your own conda environment here
# module load pytorch/v1.4.0-gpu
source /nfs-share/ls985/anaconda3/bin/activate ray111-test
cd /nfs-share/ls985/fedssl_exits/fl_powerpropagation

# srun python main.py model=base

# srun python main.py model=powerprop

# srun python main.py -m model=powerprop model.alpha=1,2,4,8 client_training_fn=pruning,base,iterative_pruning
# srun python main.py model=powerprop client_training_fn=pruning

# srun python main.py -m model=powerprop model.init_model_fn.alpha=1,2,4 client_training_fn=pruning client_training_fn.training_fn.amount=0.6,0.8,0.9 fl_setting.local_epochs=1 fl_setting.lr=0.1

# srun python main.py -m model=powerprop model.init_model_fn.alpha=1,2,4 client_training_fn=base fl_setting.local_epochs=5,10 fl_setting.lr=0.1 ray_config.gpus_per_client=0.15

# srun python main.py -m model=powerprop model.init_model_fn.alpha=1,2,4 client_training_fn=iterative_pruning client_training_fn.training_fn.amount=0.1,0.2,0.5,0.6,0.8,0.9 fl_setting.local_epochs=5,10 fl_setting.lr=0.1 ray_config.gpus_per_client=0.25

# srun python main.py -m model=powerprop model.init_model_fn.alpha=2 client_training_fn=iterative_pruning client_training_fn.training_fn.amount=0.8,0.9 fl_setting.local_epochs=5,10 fl_setting.lr=0.1 ray_config.gpus_per_client=0.25

# srun python main.py -m model=powerprop model.init_model_fn.alpha=4 client_training_fn=iterative_pruning client_training_fn.training_fn.amount=0.1,0.2,0.5,0.6,0.8,0.9 fl_setting.local_epochs=5,10 fl_setting.lr=0.1 ray_config.gpus_per_client=0.25


# srun python main.py -m model=powerprop model.init_model_fn.alpha=1,2,4,8 client_training_fn=pruning,base,iterative_pruning fl_setting.local_epochs=1,5,10 fl_setting.lr=0.1,0.01,0.001

srun python main.py -m fl_setting.local_epochs=1 ray_config.gpus_per_client=0.1 model=swat model.init_model_fn.sparsity=0.1,0.2,0.5,0.6,0.8,0.9,0.99,0.999 model.init_model_fn.alpha=1 ray_config.gpus_per_client=0.3

# srun python main.py -m fl_setting.local_epochs=1 ray_config.gpus_per_client=0.1 model=swat model.init_model_fn.sparsity=0.1,0.2,0.5,0.6,0.8,0.9,0.99,0.999 model.init_model_fn.alpha=2 ray_config.gpus_per_client=0.3

# srun python main.py -m fl_setting.local_epochs=1 ray_config.gpus_per_client=0.1 model=swat model.init_model_fn.sparsity=0.1,0.2,0.5,0.6,0.8,0.9,0.99,0.999 model.init_model_fn.alpha=4 ray_config.gpus_per_client=0.3

