#!/bin/bash
#SBATCH --cpus-per-task=5
#SBATCH --gres=gpu:rtx2080:1

cd /nfs-share/ag2411/project/fl_powerpropagation 
# poetry run python -m project.main --config-name=cluster_cifar_zerofl task.alpha=1.25 task.sparsity=0.9 task.fit_config.run_config.learning_rate=0.5 task.fit_config.run_config.final_learning_rate=0.01 fed.num_rounds=700 strategy=fedavgNZ dataset.lda_alpha=1000.0
# poetry run python -m project.main --config-name=cluster_cifar_zerofl task.alpha=1.25 task.sparsity=0.95 task.fit_config.run_config.learning_rate=0.5 task.fit_config.run_config.final_learning_rate=0.01 fed.num_rounds=700 strategy=fedavgNZ dataset.lda_alpha=1000.0

# poetry run python -m project.main --config-name=cluster_cifar_zerofl fed.seed=5378 task.alpha=1.25 task.sparsity=0.9 task.fit_config.run_config.learning_rate=0.5 task.fit_config.run_config.final_learning_rate=0.01 fed.num_rounds=700 strategy=fedavgNZ dataset.num_classes=100 dataset.lda_alpha=1000.0
# poetry run python -m project.main --config-name=cluster_cifar_zerofl fed.seed=5378 task.alpha=1.0 task.sparsity=0.5 task.fit_config.run_config.learning_rate=0.5 task.fit_config.run_config.final_learning_rate=0.01 fed.num_rounds=700 strategy=fedavgNZ dataset.num_classes=100 dataset.lda_alpha=1000.0


# poetry run python -m project.main --config-name=cluster_cifar_zerofl fed.seed=5378 task.alpha=1.0 task.sparsity=0.999 task.fit_config.run_config.learning_rate=0.5 task.fit_config.run_config.final_learning_rate=0.01 fed.num_rounds=700 strategy=fedavgNZ dataset.num_classes=100 dataset.lda_alpha=1000.0
# poetry run python -m project.main --config-name=cluster_cifar_zerofl fed.seed=9421 task.alpha=1.0 task.sparsity=0.999 task.fit_config.run_config.learning_rate=0.5 task.fit_config.run_config.final_learning_rate=0.01 fed.num_rounds=700 strategy=fedavgNZ dataset.num_classes=100 dataset.lda_alpha=1000.0
# poetry run python -m project.main --config-name=cluster_cifar_zerofl fed.seed=2035 task.alpha=1.0 task.sparsity=0.999 task.fit_config.run_config.learning_rate=0.5 task.fit_config.run_config.final_learning_rate=0.01 fed.num_rounds=700 strategy=fedavgNZ dataset.num_classes=100 dataset.lda_alpha=1000.0
# poetry run python -m project.main --config-name=cluster_cifar_zerofl fed.seed=5378 task.alpha=1.0 task.sparsity=0.999 task.fit_config.run_config.learning_rate=0.5 task.fit_config.run_config.final_learning_rate=0.01 fed.num_rounds=700 strategy=fedavgNZ dataset.num_classes=100 dataset.lda_alpha=1.0
# poetry run python -m project.main --config-name=cluster_cifar_zerofl fed.seed=9421 task.alpha=1.0 task.sparsity=0.999 task.fit_config.run_config.learning_rate=0.5 task.fit_config.run_config.final_learning_rate=0.01 fed.num_rounds=700 strategy=fedavgNZ dataset.num_classes=100 dataset.lda_alpha=1.0
# poetry run python -m project.main --config-name=cluster_cifar_zerofl fed.seed=2035 task.alpha=1.0 task.sparsity=0.999 task.fit_config.run_config.learning_rate=0.5 task.fit_config.run_config.final_learning_rate=0.01 fed.num_rounds=700 strategy=fedavgNZ dataset.num_classes=100 dataset.lda_alpha=1.0
# CIFAR10

# poetry run python -m project.main --config-name=cluster_cifar_zerofl fed.seed=5378 task.alpha=1.0 task.sparsity=0.9 task.fit_config.run_config.learning_rate=0.5 task.fit_config.run_config.final_learning_rate=0.01 fed.num_rounds=700 strategy=fedavgNZ dataset.num_classes=10 dataset.lda_alpha=0.1
# poetry run python -m project.main --config-name=cluster_cifar_zerofl fed.seed=9421 task.alpha=1.0 task.sparsity=0.9 task.fit_config.run_config.learning_rate=0.5 task.fit_config.run_config.final_learning_rate=0.01 fed.num_rounds=700 strategy=fedavgNZ dataset.num_classes=10 dataset.lda_alpha=0.1
# poetry run python -m project.main --config-name=cluster_cifar_zerofl fed.seed=2035 task.alpha=1.0 task.sparsity=0.9 task.fit_config.run_config.learning_rate=0.5 task.fit_config.run_config.final_learning_rate=0.01 fed.num_rounds=700 strategy=fedavgNZ dataset.num_classes=10 dataset.lda_alpha=0.1
# poetry run python -m project.main --config-name=cluster_cifar_zerofl fed.seed=5378 task.alpha=1.0 task.sparsity=0.95 task.fit_config.run_config.learning_rate=0.5 task.fit_config.run_config.final_learning_rate=0.01 fed.num_rounds=700 strategy=fedavgNZ dataset.num_classes=10 dataset.lda_alpha=0.1
# poetry run python -m project.main --config-name=cluster_cifar_zerofl fed.seed=9421 task.alpha=1.0 task.sparsity=0.95 task.fit_config.run_config.learning_rate=0.5 task.fit_config.run_config.final_learning_rate=0.01 fed.num_rounds=700 strategy=fedavgNZ dataset.num_classes=10 dataset.lda_alpha=0.1
# poetry run python -m project.main --config-name=cluster_cifar_zerofl fed.seed=2035 task.alpha=1.0 task.sparsity=0.95 task.fit_config.run_config.learning_rate=0.5 task.fit_config.run_config.final_learning_rate=0.01 fed.num_rounds=700 strategy=fedavgNZ dataset.num_classes=10 dataset.lda_alpha=0.1
# poetry run python -m project.main --config-name=cluster_cifar_zerofl fed.seed=5378 task.alpha=1.0 task.sparsity=0.99 task.fit_config.run_config.learning_rate=0.5 task.fit_config.run_config.final_learning_rate=0.01 fed.num_rounds=700 strategy=fedavgNZ dataset.num_classes=10 dataset.lda_alpha=0.1
# poetry run python -m project.main --config-name=cluster_cifar_zerofl fed.seed=9421 task.alpha=1.0 task.sparsity=0.99 task.fit_config.run_config.learning_rate=0.5 task.fit_config.run_config.final_learning_rate=0.01 fed.num_rounds=700 strategy=fedavgNZ dataset.num_classes=10 dataset.lda_alpha=0.1
# poetry run python -m project.main --config-name=cluster_cifar_zerofl fed.seed=5378 task.alpha=1.0 task.sparsity=0.995 task.fit_config.run_config.learning_rate=0.5 task.fit_config.run_config.final_learning_rate=0.01 fed.num_rounds=700 strategy=fedavgNZ dataset.num_classes=10 dataset.lda_alpha=0.1
# poetry run python -m project.main --config-name=cluster_cifar_zerofl fed.seed=9421 task.alpha=1.0 task.sparsity=0.995 task.fit_config.run_config.learning_rate=0.5 task.fit_config.run_config.final_learning_rate=0.01 fed.num_rounds=700 strategy=fedavgNZ dataset.num_classes=10 dataset.lda_alpha=0.1
poetry run python -m project.main --config-name=cluster_cifar_zerofl fed.seed=5378 task.alpha=1.0 task.sparsity=0.999 task.fit_config.run_config.learning_rate=0.5 task.fit_config.run_config.final_learning_rate=0.01 fed.num_rounds=700 strategy=fedavgNZ dataset.num_classes=10 dataset.lda_alpha=0.1
poetry run python -m project.main --config-name=cluster_cifar_zerofl fed.seed=9421 task.alpha=1.0 task.sparsity=0.999 task.fit_config.run_config.learning_rate=0.5 task.fit_config.run_config.final_learning_rate=0.01 fed.num_rounds=700 strategy=fedavgNZ dataset.num_classes=10 dataset.lda_alpha=0.1


# poetry run python -m project.main --config-name=cluster_cifar_swat task.alpha=1.0 task.sparsity=0.0

