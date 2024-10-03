#!/bin/bash
#SBATCH -w ngongotaha
#SBATCH --cpus-per-task=5
#SBATCH --gres=gpu:1

# Seeds 5378, 9421, 2035

cd /nfs-share/ag2411/project/fl_powerpropagation 

# poetry run python -m project.main --config-name=cluster_cifar_flash fed.seed=5378 task.sparsity=0.000 task.fit_config.run_config.learning_rate=0.5 task.fit_config.run_config.final_learning_rate=0.01 fed.num_rounds=700 strategy=fedavgFLASH dataset.num_classes=10 dataset.lda_alpha=1.0

# poetry run python -m project.main --config-name=cluster_cifar_flash fed.seed=9421 task.sparsity=0.900 task.fit_config.run_config.learning_rate=0.5 task.fit_config.run_config.final_learning_rate=0.01 fed.num_rounds=700 strategy=fedavgFLASH dataset.num_classes=100 dataset.lda_alpha=0.1
# poetry run python -m project.main --config-name=cluster_cifar_flash fed.seed=9421 task.sparsity=0.950 task.fit_config.run_config.learning_rate=0.5 task.fit_config.run_config.final_learning_rate=0.01 fed.num_rounds=700 strategy=fedavgFLASH dataset.num_classes=100 dataset.lda_alpha=0.1
# poetry run python -m project.main --config-name=cluster_cifar_flash fed.seed=9421 task.sparsity=0.990 task.fit_config.run_config.learning_rate=0.5 task.fit_config.run_config.final_learning_rate=0.01 fed.num_rounds=700 strategy=fedavgFLASH dataset.num_classes=100 dataset.lda_alpha=0.1
# poetry run python -m project.main --config-name=cluster_cifar_flash fed.seed=9421 task.sparsity=0.995 task.fit_config.run_config.learning_rate=0.5 task.fit_config.run_config.final_learning_rate=0.01 fed.num_rounds=700 strategy=fedavgFLASH dataset.num_classes=100 dataset.lda_alpha=0.1
# poetry run python -m project.main --config-name=cluster_cifar_flash fed.seed=9421 task.sparsity=0.999 task.fit_config.run_config.learning_rate=0.5 task.fit_config.run_config.final_learning_rate=0.01 fed.num_rounds=700 strategy=fedavgFLASH dataset.num_classes=100 dataset.lda_alpha=0.1


# poetry run python -m project.main --config-name=cluster_cifar_flash fed.seed=9421 task.sparsity=0.900 task.fit_config.run_config.learning_rate=0.5 task.fit_config.run_config.final_learning_rate=0.01 fed.num_rounds=700 strategy=fedavgFLASH dataset.num_classes=100 dataset.lda_alpha=1.0
# poetry run python -m project.main --config-name=cluster_cifar_flash fed.seed=9421 task.sparsity=0.950 task.fit_config.run_config.learning_rate=0.5 task.fit_config.run_config.final_learning_rate=0.01 fed.num_rounds=700 strategy=fedavgFLASH dataset.num_classes=100 dataset.lda_alpha=1.0
# poetry run python -m project.main --config-name=cluster_cifar_flash fed.seed=9421 task.sparsity=0.990 task.fit_config.run_config.learning_rate=0.5 task.fit_config.run_config.final_learning_rate=0.01 fed.num_rounds=700 strategy=fedavgFLASH dataset.num_classes=100 dataset.lda_alpha=1.0
# poetry run python -m project.main --config-name=cluster_cifar_flash fed.seed=9421 task.sparsity=0.995 task.fit_config.run_config.learning_rate=0.5 task.fit_config.run_config.final_learning_rate=0.01 fed.num_rounds=700 strategy=fedavgFLASH dataset.num_classes=100 dataset.lda_alpha=1.0
# poetry run python -m project.main --config-name=cluster_cifar_flash fed.seed=9421 task.sparsity=0.999 task.fit_config.run_config.learning_rate=0.5 task.fit_config.run_config.final_learning_rate=0.01 fed.num_rounds=700 strategy=fedavgFLASH dataset.num_classes=100 dataset.lda_alpha=1.0


# poetry run python -m project.main --config-name=cluster_cifar_flash fed.seed=9421 task.sparsity=0.900 task.fit_config.run_config.learning_rate=0.5 task.fit_config.run_config.final_learning_rate=0.01 fed.num_rounds=700 strategy=fedavgFLASH dataset.num_classes=100 dataset.lda_alpha=1000.0
# poetry run python -m project.main --config-name=cluster_cifar_flash fed.seed=9421 task.sparsity=0.950 task.fit_config.run_config.learning_rate=0.5 task.fit_config.run_config.final_learning_rate=0.01 fed.num_rounds=700 strategy=fedavgFLASH dataset.num_classes=100 dataset.lda_alpha=1000.0
# poetry run python -m project.main --config-name=cluster_cifar_flash fed.seed=9421 task.sparsity=0.990 task.fit_config.run_config.learning_rate=0.5 task.fit_config.run_config.final_learning_rate=0.01 fed.num_rounds=700 strategy=fedavgFLASH dataset.num_classes=100 dataset.lda_alpha=1000.0
# poetry run python -m project.main --config-name=cluster_cifar_flash fed.seed=9421 task.sparsity=0.995 task.fit_config.run_config.learning_rate=0.5 task.fit_config.run_config.final_learning_rate=0.01 fed.num_rounds=700 strategy=fedavgFLASH dataset.num_classes=100 dataset.lda_alpha=1000.0
# poetry run python -m project.main --config-name=cluster_cifar_flash fed.seed=9421 task.sparsity=0.999 task.fit_config.run_config.learning_rate=0.5 task.fit_config.run_config.final_learning_rate=0.01 fed.num_rounds=700 strategy=fedavgFLASH dataset.num_classes=100 dataset.lda_alpha=1000.0



# Continual window
# poetry run python -m project.main --config-name=cluster_cifar_flash fed.seed=5378 task.sparsity=0.950 task.fit_config.run_config.learning_rate=0.5 task.fit_config.run_config.final_learning_rate=0.01 fed.num_rounds=500 strategy=fedavgCFLASH dataset.num_classes=10 dataset.lda_alpha=1.0 wandb.setup.project=continual_learning task.eval_config.extra.window_training=True dataset=cluster_cifar_clustered_lda


# mask test
poetry run python -m project.main --config-name=cluster_cifar_flash fed.seed=2035 task.sparsity=0.950 task.fit_config.run_config.learning_rate=0.5 task.fit_config.run_config.final_learning_rate=0.01 fed.num_rounds=500 strategy=fedavgFLASH dataset.num_classes=10 dataset.lda_alpha=1000.0 wandb.setup.project=cleaning
