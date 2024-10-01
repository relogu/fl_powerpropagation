#!/bin/bash
#SBATCH -w ngongotaha
#SBATCH --cpus-per-task=5
#SBATCH --gres=gpu:1

cd /nfs-share/ag2411/project/fl_powerpropagation 


# LDA1.0
# poetry run python -m project.main --config-name=cluster_cifar_power_swat fed.seed=5378 task.alpha=-1 task.sparsity=0.900 task.fit_config.run_config.learning_rate=0.5 task.fit_config.run_config.final_learning_rate=0.01 fed.num_rounds=500 dataset.num_classes=10 dataset.lda_alpha=1.0
# poetry run python -m project.main --config-name=cluster_cifar_power_swat fed.seed=5378 task.alpha=-1 task.sparsity=0.950 task.fit_config.run_config.learning_rate=0.5 task.fit_config.run_config.final_learning_rate=0.01 fed.num_rounds=500 dataset.num_classes=10 dataset.lda_alpha=1.0
# poetry run python -m project.main --config-name=cluster_cifar_power_swat fed.seed=5378 task.alpha=-1 task.sparsity=0.990 task.fit_config.run_config.learning_rate=0.5 task.fit_config.run_config.final_learning_rate=0.01 fed.num_rounds=500 dataset.num_classes=10 dataset.lda_alpha=1.0
# poetry run python -m project.main --config-name=cluster_cifar_power_swat fed.seed=5378 task.alpha=-1 task.sparsity=0.995 task.fit_config.run_config.learning_rate=0.5 task.fit_config.run_config.final_learning_rate=0.01 fed.num_rounds=500 dataset.num_classes=10 dataset.lda_alpha=1.0
# poetry run python -m project.main --config-name=cluster_cifar_power_swat fed.seed=5378 task.alpha=-1 task.sparsity=0.999 task.fit_config.run_config.learning_rate=0.5 task.fit_config.run_config.final_learning_rate=0.01 fed.num_rounds=500 dataset.num_classes=10 dataset.lda_alpha=1.0



# poetry run python -m project.main --config-name=cluster_cifar_power_swat fed.seed=5378 task.alpha=1.0 task.sparsity=0.95 task.fit_config.run_config.learning_rate=0.5 task.fit_config.run_config.final_learning_rate=0.01 fed.num_rounds=500 dataset.num_classes=10 dataset.lda_alpha=1.0 wandb.setup.project=ablation_alpha_sensitivity
# poetry run python -m project.main --config-name=cluster_cifar_power_swat fed.seed=5378 task.alpha=1.125 task.sparsity=0.95 task.fit_config.run_config.learning_rate=0.5 task.fit_config.run_config.final_learning_rate=0.01 fed.num_rounds=500 dataset.num_classes=10 dataset.lda_alpha=1.0 wandb.setup.project=ablation_alpha_sensitivity
# poetry run python -m project.main --config-name=cluster_cifar_power_swat fed.seed=5378 task.alpha=1.250 task.sparsity=0.95 task.fit_config.run_config.learning_rate=0.5 task.fit_config.run_config.final_learning_rate=0.01 fed.num_rounds=500 dataset.num_classes=10 dataset.lda_alpha=1.0 wandb.setup.project=ablation_alpha_sensitivity
# poetry run python -m project.main --config-name=cluster_cifar_power_swat fed.seed=5378 task.alpha=1.375 task.sparsity=0.95 task.fit_config.run_config.learning_rate=0.5 task.fit_config.run_config.final_learning_rate=0.01 fed.num_rounds=500 dataset.num_classes=10 dataset.lda_alpha=1.0 wandb.setup.project=ablation_alpha_sensitivity
# poetry run python -m project.main --config-name=cluster_cifar_power_swat fed.seed=5378 task.alpha=2.250 task.sparsity=0.95 task.fit_config.run_config.learning_rate=0.5 task.fit_config.run_config.final_learning_rate=0.01 fed.num_rounds=500 dataset.num_classes=10 dataset.lda_alpha=1.0 wandb.setup.project=ablation_alpha_sensitivity
poetry run python -m project.main --config-name=cluster_cifar_power_swat fed.seed=5378 task.alpha=4.0 task.sparsity=0.95 task.fit_config.run_config.learning_rate=0.5 task.fit_config.run_config.final_learning_rate=0.01 fed.num_rounds=500 dataset.num_classes=10 dataset.lda_alpha=1.0 wandb.setup.project=ablation_alpha_sensitivity
# poetry run python -m project.main --config-name=cluster_cifar_power_swat fed.seed=5378 task.alpha=2.000 task.sparsity=0.95 task.fit_config.run_config.learning_rate=0.5 task.fit_config.run_config.final_learning_rate=0.01 fed.num_rounds=500 dataset.num_classes=10 dataset.lda_alpha=1.0 wandb.setup.project=ablation_alpha_sensitivity
# poetry run python -m project.main --config-name=cluster_cifar_power_swat fed.seed=5378 task.alpha=3.000 task.sparsity=0.95 task.fit_config.run_config.learning_rate=0.5 task.fit_config.run_config.final_learning_rate=0.01 fed.num_rounds=500 dataset.num_classes=10 dataset.lda_alpha=1.0 wandb.setup.project=ablation_alpha_sensitivity
