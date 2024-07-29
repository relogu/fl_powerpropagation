#!/bin/bash
# #SBATCH --cpus-per-task=5
# #SBATCH --gres=gpu:rtx2080:1

cd /home/aguastella/fl_powerpropagation

# poetry run python -m project.main --config-name=alessio_speech_resnet18 fed.seed=9421 task.model_and_data=SPEECH_PPSWAT task.alpha=1.25 task.sparsity=0.9 task.mask=0.0 task.fit_config.dataloader_config.batch_size=16 task.fit_config.run_config.learning_rate=0.1 task.fit_config.run_config.final_learning_rate=0.001 dataset.lda_alpha=0.1 fed.num_rounds=500 
# poetry run python -m project.main --config-name=alessio_speech_resnet18 fed.seed=9421 task.model_and_data=SPEECH_PPSWAT task.alpha=1.25 task.sparsity=0.95 task.mask=0.0 task.fit_config.dataloader_config.batch_size=16 task.fit_config.run_config.learning_rate=0.1 task.fit_config.run_config.final_learning_rate=0.001 dataset.lda_alpha=0.1 fed.num_rounds=500 
# poetry run python -m project.main --config-name=alessio_speech_resnet18 fed.seed=9421 task.model_and_data=SPEECH_PPSWAT task.alpha=1.25 task.sparsity=0.99 task.mask=0.0 task.fit_config.dataloader_config.batch_size=16 task.fit_config.run_config.learning_rate=0.1 task.fit_config.run_config.final_learning_rate=0.001 dataset.lda_alpha=0.1 fed.num_rounds=500 
# poetry run python -m project.main --config-name=alessio_speech_resnet18 fed.seed=9421 task.model_and_data=SPEECH_PPSWAT task.alpha=1.25 task.sparsity=0.995 task.mask=0.0 task.fit_config.dataloader_config.batch_size=16 task.fit_config.run_config.learning_rate=0.1 task.fit_config.run_config.final_learning_rate=0.001 dataset.lda_alpha=0.1 fed.num_rounds=500 
# poetry run python -m project.main --config-name=alessio_speech_resnet18 fed.seed=9421 task.model_and_data=SPEECH_PPSWAT task.alpha=1.25 task.sparsity=0.999 task.mask=0.0 task.fit_config.dataloader_config.batch_size=16 task.fit_config.run_config.learning_rate=0.1 task.fit_config.run_config.final_learning_rate=0.001 dataset.lda_alpha=0.1 fed.num_rounds=500 

# poetry run python -m project.main --config-name=local_cifar_powerprop_pruning fed.seed=5378 task.alpha=1.0 task.sparsity=0.9 task.fit_config.run_config.learning_rate=0.1 task.fit_config.run_config.final_learning_rate=0.1 fed.num_rounds=700 dataset.num_classes=10 dataset.lda_alpha=1.0 strategy=fedavg
# poetry run python -m project.main --config-name=local_cifar_powerprop_pruning fed.seed=5378 task.alpha=1.25 task.sparsity=0.9 task.fit_config.run_config.learning_rate=0.1 task.fit_config.run_config.final_learning_rate=0.01 fed.num_rounds=700 dataset.num_classes=10 dataset.lda_alpha=1.0 strategy=fedavg
poetry run python -m project.main --config-name=local_cifar_swat_pruning fed.seed=5378 task.alpha=1.25 task.sparsity=0.9 task.fit_config.run_config.learning_rate=0.5 task.fit_config.run_config.final_learning_rate=0.01 fed.num_rounds=700 dataset.num_classes=10 dataset.lda_alpha=1.0 strategy=fedavgNZ








# poetry run python -m project.main --config-name=alessio_speech_resnet18 task.alpha=1.0 task.sparsity=0.9 task.fit_config.dataloader_config.batch_size=16 task.fit_config.run_config.learning_rate=0.1 task.fit_config.run_config.final_learning_rate=0.001 dataset.lda_alpha=0.1
# poetry run python -m project.main --config-name=alessio_speech_resnet18 task.alpha=1.0 task.sparsity=0.995 task.fit_config.dataloader_config.batch_size=16 task.fit_config.run_config.learning_rate=0.1 task.fit_config.run_config.final_learning_rate=0.001 dataset.lda_alpha=1.0
# poetry run python -m project.main --config-name=alessio_speech_resnet18 task.alpha=1.0 task.sparsity=0.999 task.fit_config.dataloader_config.batch_size=16 task.fit_config.run_config.learning_rate=0.1 task.fit_config.run_config.final_learning_rate=0.001 dataset.lda_alpha=1.0


# PARTIZIONAMENTO PER LDA=1000.0
# poetry run python -m project.task.speech_resnet18.dataset_preparation --config-name=alessio_speech_resnet18 dataset.lda_alpha=1000.0

# poetry run python -m project.main --config-name=alessio_speech_resnet18 task.alpha=1.0 task.sparsity=0.0 task.fit_config.dataloader_config.batch_size=16 task.fit_config.run_config.learning_rate=0.1 task.fit_config.run_config.final_learning_rate=0.001 dataset.lda_alpha=1000.0
# poetry run python -m project.main --config-name=alessio_speech_resnet18 task.alpha=1.0 task.sparsity=0.9 task.fit_config.dataloader_config.batch_size=16 task.fit_config.run_config.learning_rate=0.1 task.fit_config.run_config.final_learning_rate=0.001 dataset.lda_alpha=1000.0
# poetry run python -m project.main --config-name=alessio_speech_resnet18 task.alpha=1.0 task.sparsity=0.95 task.fit_config.dataloader_config.batch_size=16 task.fit_config.run_config.learning_rate=0.1 task.fit_config.run_config.final_learning_rate=0.001 dataset.lda_alpha=1000.0
# poetry run python -m project.main --config-name=alessio_speech_resnet18 task.alpha=1.0 task.sparsity=0.995 task.fit_config.dataloader_config.batch_size=16 task.fit_config.run_config.learning_rate=0.1 task.fit_config.run_config.final_learning_rate=0.001 dataset.lda_alpha=1000.0
# poetry run python -m project.main --config-name=alessio_speech_resnet18 task.alpha=1.0 task.sparsity=0.999 task.fit_config.dataloader_config.batch_size=16 task.fit_config.run_config.learning_rate=0.1 task.fit_config.run_config.final_learning_rate=0.001 dataset.lda_alpha=1000.0

# PARTIZIONAMENTO PER LDA=0.1
# poetry run python -m project.task.speech_resnet18.dataset_preparation --config-name=alessio_speech_resnet18 dataset.lda_alpha=0.1

# poetry run python -m project.main --config-name=alessio_speech_resnet18 task.alpha=1.0 task.sparsity=0.0 task.fit_config.dataloader_config.batch_size=16 task.fit_config.run_config.learning_rate=0.1 task.fit_config.run_config.final_learning_rate=0.001 dataset.lda_alpha=0.1
# poetry run python -m project.main --config-name=alessio_speech_resnet18 task.alpha=1.0 task.sparsity=0.9 task.fit_config.dataloader_config.batch_size=16 task.fit_config.run_config.learning_rate=0.1 task.fit_config.run_config.final_learning_rate=0.001 dataset.lda_alpha=0.1
# poetry run python -m project.main --config-name=alessio_speech_resnet18 task.alpha=1.0 task.sparsity=0.95 task.fit_config.dataloader_config.batch_size=16 task.fit_config.run_config.learning_rate=0.1 task.fit_config.run_config.final_learning_rate=0.001 dataset.lda_alpha=0.1
# poetry run python -m project.main --config-name=alessio_speech_resnet18 task.alpha=1.0 task.sparsity=0.995 task.fit_config.dataloader_config.batch_size=16 task.fit_config.run_config.learning_rate=0.1 task.fit_config.run_config.final_learning_rate=0.001 dataset.lda_alpha=0.1
# poetry run python -m project.main --config-name=alessio_speech_resnet18 task.alpha=1.0 task.sparsity=0.999 task.fit_config.dataloader_config.batch_size=16 task.fit_config.run_config.learning_rate=0.1 task.fit_config.run_config.final_learning_rate=0.001 dataset.lda_alpha=0.1
