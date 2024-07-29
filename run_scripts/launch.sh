#!/bin/bash
# Add your code for the job manager

poetry run python -m project.main --config-name=local_cifar_powerprop_pruning fed.seed=5378 task.alpha=1.0 task.sparsity=0.9 task.fit_config.run_config.learning_rate=0.5 task.fit_config.run_config.final_learning_rate=0.01 fed.num_rounds=300 dataset.num_classes=10 dataset.lda_alpha=0.1 strategy=fedavg
poetry run python -m project.main --config-name=local_cifar_powerprop_pruning fed.seed=5378 task.alpha=1.25 task.sparsity=0.9 task.fit_config.run_config.learning_rate=0.5 task.fit_config.run_config.final_learning_rate=0.01 fed.num_rounds=300 dataset.num_classes=10 dataset.lda_alpha=0.1 strategy=fedavg
poetry run python -m project.main --config-name=local_cifar_swat_pruning fed.seed=5378 task.alpha=1.0 task.sparsity=0.9 task.fit_config.run_config.learning_rate=0.5 task.fit_config.run_config.final_learning_rate=0.01 fed.num_rounds=300 dataset.num_classes=10 dataset.lda_alpha=0.1 strategy=fedavgNZ
