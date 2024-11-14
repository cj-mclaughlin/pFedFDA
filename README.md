# Source Code for pFedFDA: Personalized Federated Learning via Feature Distribution Adaptation

PyTorch implementation of pFedFDA: Personalized Federated Learning via Feature Distribution Adaptation (NeurIPS 2024). 

## Data Setup

Folder `data/` contains scripts for generating non-IID client partitions, and for generating corrupted versions of CIFAR datasets.

We require that an environment variable `DATA_PATH` is defined as the root folder for all dataset installations.

A simple shell script `data/partition.sh` is provided to generate the CIFAR10/100 partitions.

Instructions and code for setting up EMNIST, partitioned by writers, can be found in `data/utils/emnist_setup.py`.

Corrupted CIFAR datasets can be generated with `python generate_corruptions.py`.

## Launch Script

We provide an example launch script in `launch.sh`, which runs our method and FedAvg on CIFAR10-Dir(0.5). 

The launch script can be modified to run other configurations.

## Environment Details

We run our code using PyTorch 2.1 with CUDA 12. We provide a reference conda environment in `environment.yml`.

## Citation

```
@inproceedings{
mclaughlin2024personalized,
title={Personalized Federated Learning via Feature Distribution Adaptation},
author={Connor Mclaughlin and Lili Su},
booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
year={2024},
url={https://openreview.net/forum?id=Wl2optQcng}
}
```
