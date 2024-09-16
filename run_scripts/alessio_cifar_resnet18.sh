#!/bin/bash

cd /home/aguastella/fl_powerpropagation

# da lanciare
# PARTIZIONAMENTO PER LDA=1000.0
# poetry run python -m project.main --config-name=local_cifar_swat_pruning fed.seed=5378 task.alpha=1.0 task.sparsity=0.95 task.fit_config.run_config.learning_rate=0.5 task.fit_config.run_config.final_learning_rate=0.01 fed.num_rounds=600 dataset.num_classes=10 dataset.lda_alpha=1000.0
# poetry run python -m project.main --config-name=local_cifar_powerprop_pruning fed.seed=5378 task.alpha=1.0 task.sparsity=0.95 task.fit_config.run_config.learning_rate=0.5 task.fit_config.run_config.final_learning_rate=0.01 fed.num_rounds=600 dataset.num_classes=10 dataset.lda_alpha=1000.0
# poetry run python -m project.main --config-name=local_cifar_powerprop_pruning fed.seed=5378 task.alpha=1.25 task.sparsity=0.95 task.fit_config.run_config.learning_rate=0.5 task.fit_config.run_config.final_learning_rate=0.01 fed.num_rounds=600 dataset.num_classes=10 dataset.lda_alpha=1000.0

# PARTIZIONAMENTO PER LDA=1.0
# poetry run python -m project.main --config-name=local_cifar_swat_pruning fed.seed=5378 task.alpha=1.0 task.sparsity=0.95 task.fit_config.run_config.learning_rate=0.5 task.fit_config.run_config.final_learning_rate=0.01 fed.num_rounds=600 dataset.num_classes=10 dataset.lda_alpha=1.0

# PARTIZIONAMENTO PER LDA=0.1
# poetry run python -m project.main --config-name=local_cifar_swat_pruning fed.seed=5378 task.alpha=1.0 task.sparsity=0.95 task.fit_config.run_config.learning_rate=0.5 task.fit_config.run_config.final_learning_rate=0.01 fed.num_rounds=600 dataset.num_classes=10 dataset.lda_alpha=0.1
# poetry run python -m project.main --config-name=local_cifar_powerprop_pruning fed.seed=5378 task.alpha=1.0 task.sparsity=0.95 task.fit_config.run_config.learning_rate=0.5 task.fit_config.run_config.final_learning_rate=0.01 fed.num_rounds=600 dataset.num_classes=10 dataset.lda_alpha=0.1
poetry run python -m project.main --config-name=local_cifar_powerprop_pruning fed.seed=5378 task.alpha=1.25 task.sparsity=0.95 task.fit_config.run_config.learning_rate=0.5 task.fit_config.run_config.final_learning_rate=0.01 fed.num_rounds=600 dataset.num_classes=10 dataset.lda_alpha=0.1