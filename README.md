# PowerPropagation in Federated Learning using Flower

This repository contains the codebase for experimenting with the PowerPropagation reparametrisation proposed in [this paper](https://arxiv.org/abs/2110.00296).
The code base leverages Hydra for the configuration of the experiments.
The code has been tested with ResNet18 training of CIFAR-10 partitioned using LDA, with 100 clients and $\alpha=0.1$.
