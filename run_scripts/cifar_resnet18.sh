#!/bin/bash
#SBATCH -w ngongotaha
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1

cd /nfs-share/ag2411/project/fl_powerpropagation

# Define arrays for each parameter
model_and_data_options=(
    "CIFAR_RESNET18"
    "CIFAR_SPARSYFED_RESNET18"
    "CIFAR_SPARSYFED_NO_ACT_RESNET18"
    "CIFAR_ZEROFL_RESNET18"
    "CIFAR_FLASH_RESNET18"
)

train_structure_options=(
    "CIFAR_RESNET18"
    "CIFAR_RESNET18_PRUNE"
    "CIFAR_RESNET18_FIXED_PRUNE"
)

alpha_options=(1.25)
sparsity_options=(0.9)
seed_options=(5378)

# Loop through all combinations
for model_and_data in "${model_and_data_options[@]}"; do
    for train_structure in "${train_structure_options[@]}"; do
        for alpha in "${alpha_options[@]}"; do
            for sparsity in "${sparsity_options[@]}"; do
                for seed in "${seed_options[@]}"; do
                    echo "Running combination: $model_and_data $train_structure $alpha $sparsity $seed"
                    poetry run python -m project.main \
                        --config-name=cifar_resnet18 \
                        fed.seed="$seed" \
                        task.model_and_data="$model_and_data" \
                        task.train_structure="$train_structure" \
                        task.alpha="$alpha" \
                        task.sparsity="$sparsity" \
                        fed.num_rounds=2
                done
            done
        done
    done
done

echo "All combinations have been executed."