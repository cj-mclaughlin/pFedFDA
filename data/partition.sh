#!/bin/bash

# cifar10
python generate_data.py dirichlet --dataset cifar10 --num_clients 100 --dirichlet_alpha 0.1 --train_prop 0.8 --name cifar10_c100_dir01
python generate_data.py dirichlet --dataset cifar10 --num_clients 100 --dirichlet_alpha 0.5 --train_prop 0.8 --name cifar10_c100_dir05

# cifar100
python generate_data.py dirichlet --dataset cifar100 --num_clients 100 --dirichlet_alpha 0.1 --train_prop 0.8 --name cifar100_c100_dir01
python generate_data.py dirichlet --dataset cifar100 --num_clients 100 --dirichlet_alpha 0.5 --train_prop 0.8 --name cifar100_c100_dir05

# # tinyimagenet
# python generate_data.py dirichlet --dataset tinyimagenet --num_clients 100 --dirichlet_alpha 0.1 --train_prop 0.8 --name tinyimagenet_c100_dir01
# python generate_data.py dirichlet --dataset tinyimagenet --num_clients 100 --dirichlet_alpha 0.5 --train_prop 0.8 --name tinyimagenet_c100_dir05