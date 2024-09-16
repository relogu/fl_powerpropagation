#!/bin/bash
#SBATCH -w ngongotaha
#SBATCH --cpus-per-task=5
#SBATCH --gres=gpu:1


cd /nfs-share/ag2411/project/fl_powerpropagation 

# SPECTRAL
# poetry run python -m project.main --config-name=cluster_cifar_powerprop_pruning fed.seed=5378 task.alpha=-1.0 task.sparsity=0.000 task.fit_config.run_config.learning_rate=0.5 task.fit_config.run_config.final_learning_rate=0.01 fed.num_rounds=700 strategy=fedavg dataset.num_classes=10 dataset.lda_alpha=1.0
# poetry run python -m project.main --config-name=cluster_cifar_powerprop_pruning fed.seed=5378 task.alpha=-1.0 task.sparsity=0.900 task.fit_config.run_config.learning_rate=0.5 task.fit_config.run_config.final_learning_rate=0.01 fed.num_rounds=700 strategy=fedavg dataset.num_classes=10 dataset.lda_alpha=1.0
# poetry run python -m project.main --config-name=cluster_cifar_powerprop_pruning fed.seed=5378 task.alpha=-1.0 task.sparsity=0.950 task.fit_config.run_config.learning_rate=0.5 task.fit_config.run_config.final_learning_rate=0.01 fed.num_rounds=700 strategy=fedavg dataset.num_classes=10 dataset.lda_alpha=1.0
# poetry run python -m project.main --config-name=cluster_cifar_powerprop_pruning fed.seed=5378 task.alpha=-1.0 task.sparsity=0.990 task.fit_config.run_config.learning_rate=0.5 task.fit_config.run_config.final_learning_rate=0.01 fed.num_rounds=700 strategy=fedavg dataset.num_classes=10 dataset.lda_alpha=1.0
# poetry run python -m project.main --config-name=cluster_cifar_powerprop_pruning fed.seed=5378 task.alpha=-1.0 task.sparsity=0.995 task.fit_config.run_config.learning_rate=0.5 task.fit_config.run_config.final_learning_rate=0.01 fed.num_rounds=700 strategy=fedavg dataset.num_classes=10 dataset.lda_alpha=1.0
# poetry run python -m project.main --config-name=cluster_cifar_powerprop_pruning fed.seed=5378 task.alpha=-1.0 task.sparsity=0.999 task.fit_config.run_config.learning_rate=0.5 task.fit_config.run_config.final_learning_rate=0.01 fed.num_rounds=700 strategy=fedavg dataset.num_classes=10 dataset.lda_alpha=1.0

# poetry run python -m project.main --config-name=cluster_cifar_powerprop_pruning fed.seed=5378 task.alpha=1.25 task.sparsity=0.9 task.fit_config.run_config.learning_rate=0.5 task.fit_config.run_config.final_learning_rate=0.01 fed.num_rounds=700 strategy=fedavgHNZ dataset.num_classes=10 dataset.lda_alpha=0.1

# poetry run python -m project.main --config-name=cluster_cifar_powerprop_pruning fed.seed=5378 task.alpha=1.0 task.sparsity=0.99 task.fit_config.run_config.learning_rate=0.5 task.fit_config.run_config.final_learning_rate=0.01 fed.num_rounds=700 strategy=fedavgFLASH dataset.num_classes=10 dataset.lda_alpha=1000.0

# poetry run python -m project.main --config-name=cluster_cifar_powerprop_pruning fed.seed=5378 task.alpha=1.25 task.sparsity=0.95 task.fit_config.run_config.learning_rate=0.5 task.fit_config.run_config.final_learning_rate=0.01 fed.num_rounds=700 strategy=fedavg dataset.num_classes=10 dataset.lda_alpha=1.0
# poetry run python -m project.main --config-name=cluster_cifar_powerprop_pruning fed.seed=5378 task.alpha=1.25 task.sparsity=0.95 task.fit_config.run_config.learning_rate=0.5 task.fit_config.run_config.final_learning_rate=0.01 fed.num_rounds=700 strategy=fedavg dataset.num_classes=10 dataset.lda_alpha=0.1



# poetry run python -m project.main --config-name=cluster_cifar_powerprop_pruning fed.seed=5378 task.alpha=1.0 task.sparsity=0.95 task.fit_config.run_config.learning_rate=0.5 task.fit_config.run_config.final_learning_rate=0.01 fed.num_rounds=700 strategy=fedavg dataset.num_classes=100 dataset.lda_alpha=1000.0
# poetry run python -m project.main --config-name=cluster_cifar_powerprop_pruning fed.seed=9421 task.alpha=1.0 task.sparsity=0.95 task.fit_config.run_config.learning_rate=0.5 task.fit_config.run_config.final_learning_rate=0.01 fed.num_rounds=700 strategy=fedavg dataset.num_classes=100 dataset.lda_alpha=1000.0
# poetry run python -m project.main --config-name=cluster_cifar_powerprop_pruning fed.seed=2035 task.alpha=1.0 task.sparsity=0.95 task.fit_config.run_config.learning_rate=0.5 task.fit_config.run_config.final_learning_rate=0.01 fed.num_rounds=700 strategy=fedavg dataset.num_classes=100 dataset.lda_alpha=1000.0
# poetry run python -m project.main --config-name=cluster_cifar_powerprop_pruning fed.seed=5378 task.alpha=1.25 task.sparsity=0.99 task.fit_config.run_config.learning_rate=0.5 task.fit_config.run_config.final_learning_rate=0.01 fed.num_rounds=700 strategy=fedavg dataset.num_classes=100 dataset.lda_alpha=1.0
# poetry run python -m project.main --config-name=cluster_cifar_powerprop_pruning fed.seed=9421 task.alpha=1.25 task.sparsity=0.99 task.fit_config.run_config.learning_rate=0.5 task.fit_config.run_config.final_learning_rate=0.01 fed.num_rounds=700 strategy=fedavg dataset.num_classes=100 dataset.lda_alpha=1.0
# poetry run python -m project.main --config-name=cluster_cifar_powerprop_pruning fed.seed=2035 task.alpha=1.25 task.sparsity=0.99 task.fit_config.run_config.learning_rate=0.5 task.fit_config.run_config.final_learning_rate=0.01 fed.num_rounds=700 strategy=fedavg dataset.num_classes=100 dataset.lda_alpha=1.0

# poetry run python -m project.main --config-name=cluster_cifar_powerprop_pruning fed.seed=5378 task.alpha=1.0 task.sparsity=0.999 task.fit_config.run_config.learning_rate=0.5 task.fit_config.run_config.final_learning_rate=0.01 fed.num_rounds=700 strategy=fedavg dataset.num_classes=10 dataset.lda_alpha=0.1
# poetry run python -m project.main --config-name=cluster_cifar_powerprop_pruning fed.seed=9421 task.alpha=1.0 task.sparsity=0.999 task.fit_config.run_config.learning_rate=0.5 task.fit_config.run_config.final_learning_rate=0.01 fed.num_rounds=700 strategy=fedavg dataset.num_classes=10 dataset.lda_alpha=0.1
# poetry run python -m project.main --config-name=cluster_cifar_powerprop_pruning fed.seed=2035 task.alpha=1.0 task.sparsity=0.999 task.fit_config.run_config.learning_rate=0.5 task.fit_config.run_config.final_learning_rate=0.01 fed.num_rounds=700 strategy=fedavg dataset.num_classes=10 dataset.lda_alpha=0.1

# LDA tests
# poetry run python -m project.main --config-name=cluster_cifar_powerprop_pruning task.alpha=1.0 task.sparsity=0.0 task.fit_config.run_config.learning_rate=0.5 task.fit_config.run_config.final_learning_rate=0.01 fed.num_rounds=200 strategy=fedavg dataset.lda_alpha=0.005 # dataset.num_classes=100
# poetry run python -m project.main --config-name=cluster_cifar_powerprop_pruning task.alpha=1.0 task.sparsity=0.0 task.fit_config.run_config.learning_rate=0.5 task.fit_config.run_config.final_learning_rate=0.01 fed.num_rounds=200 strategy=fedavg dataset.lda_alpha=0.001 # dataset.num_classes=100
# poetry run python -m project.main --config-name=cluster_cifar_powerprop_pruning task.alpha=1.0 task.sparsity=0.0 task.fit_config.run_config.learning_rate=0.5 task.fit_config.run_config.final_learning_rate=0.01 fed.num_rounds=200 strategy=fedavg dataset.lda_alpha=0.01 # dataset.num_classes=100
# poetry run python -m project.main --config-name=cluster_cifar_powerprop_pruning task.alpha=1.0 task.sparsity=0.0 task.fit_config.run_config.learning_rate=0.5 task.fit_config.run_config.final_learning_rate=0.01 fed.num_rounds=200 strategy=fedavg dataset.lda_alpha=0.05 # dataset.num_classes=100
# poetry run python -m project.main --config-name=cluster_cifar_powerprop_pruning task.alpha=1.0 task.sparsity=0.0 task.fit_config.run_config.learning_rate=0.5 task.fit_config.run_config.final_learning_rate=0.01 fed.num_rounds=200 strategy=fedavg dataset.lda_alpha=0.1 # dataset.num_classes=100
# poetry run python -m project.main --config-name=cluster_cifar_powerprop_pruning task.alpha=1.0 task.sparsity=0.0 task.fit_config.run_config.learning_rate=0.5 task.fit_config.run_config.final_learning_rate=0.01 fed.num_rounds=200 strategy=fedavg dataset.lda_alpha=0.5 # dataset.num_classes=100
# poetry run python -m project.main --config-name=cluster_cifar_powerprop_pruning task.alpha=1.0 task.sparsity=0.0 task.fit_config.run_config.learning_rate=0.5 task.fit_config.run_config.final_learning_rate=0.01 fed.num_rounds=200 strategy=fedavg dataset.lda_alpha=1.0 # dataset.num_classes=100
# poetry run python -m project.main --config-name=cluster_cifar_powerprop_pruning task.alpha=1.0 task.sparsity=0.0 task.fit_config.run_config.learning_rate=0.5 task.fit_config.run_config.final_learning_rate=0.01 fed.num_rounds=200 strategy=fedavg dataset.lda_alpha=1000.0 # dataset.num_classes=100


# poetry run python -m project.main --config-name=cluster_cifar_powerprop_pruning task.alpha=1.25 task.sparsity=0.95 task.fit_config.run_config.learning_rate=0.5 task.fit_config.run_config.final_learning_rate=0.01 fed.num_rounds=300 strategy=fedavg dataset.num_classes=10 dataset.lda_alpha=1000.0
# poetry run python -m project.main --config-name=cluster_cifar_powerprop_pruning task.alpha=1.25 task.sparsity=0.95 task.fit_config.run_config.learning_rate=0.5 task.fit_config.run_config.final_learning_rate=0.01 fed.num_rounds=300 strategy=fedavg dataset.num_classes=10 dataset.lda_alpha=1000.0
# poetry run python -m project.main --config-name=cluster_cifar_powerprop_pruning task.alpha=1.0 task.sparsity=0.95 task.fit_config.run_config.learning_rate=0.5 task.fit_config.run_config.final_learning_rate=0.01 fed.num_rounds=300 strategy=fedavg dataset.num_classes=10 dataset.lda_alpha=1000.0 strategy=fedavgNZ
