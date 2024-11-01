import argparse
import torch
import numpy as np
from data.utils.loader import get_base_dataset
from servers.server_fedavg import ServerFedAvg
from servers.server_fedfda import ServerFedFDA
from servers.server_local import ServerLocal
from models import model_dict

import warnings
warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser()
    # dataset arguments
    parser.add_argument("--dataset", default="cifar10", 
                        choices=["cifar10", "cifar100", "digit5", "tinyimagenet", "emnist"],
                        type=str)
    parser.add_argument("--num_classes", default=10, type=int)
    parser.add_argument("--partition_path", default="cifar10_c100_dir05", type=str, help="name of partition folder")
    parser.add_argument("--augmented", action="store_true", help="whether or not to augment the first 50 clients (Only for CIFAR)")
    # generic training hyperparameters
    parser.add_argument("--global_rounds", default=200, type=int)
    parser.add_argument("--local_epochs", default=5, type=int)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--momentum", default=0.5, type=float)
    parser.add_argument("--wd", default=5e-4, type=float)
    parser.add_argument("--batch_size", default=50, type=int)
    parser.add_argument("--eval_gap", default=200, type=float, help="Rounds Between Model Evaluation")
    parser.add_argument("--train_prop", default=1.0, type=float, help="Proportion of Training Data To Use")
    # FL/Server Setup
    parser.add_argument("--method", default="pFedFDA", type=str)
    parser.add_argument("--num_clients", default=100, type=int)
    parser.add_argument("--sampling_prob", default=0.3, type=float, help="Client Participation Probability")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", type=str)
    # model architecture
    parser.add_argument("--model_name", default="cnn", type=str, help="Model Architecture", choices=["cnn", "resnet18"])   
    # method-specific hyperparameters
    parser.add_argument("--p_epochs", default=5, type=int, help="Number of Personalization Epochs") 
    parser.add_argument("--single_beta", action="store_true", help="if we should only use a single beta term (pFedFDA)") 
    parser.add_argument("--local_beta", action="store_true", help="if we should only use only local statistics (pFedFDA)")
    # logging/saving
    parser.add_argument("--exp_name", default="baseline", type=str, help="save file prefix") 
    args = parser.parse_args()

    # numpy seed (ensures repeatable subsampling)
    np.random.seed(0)

    # ensure arguments are correct
    if args.dataset in ["mnist", "emnist", "fmnist"]:
        in_channels = 1
        if args.model_name == "cnn":
            args.model_name = "emnistnet"
    else:
        in_channels = 3
        if args.model_name == "cnn":
            args.model_name = "cifarnet"

    if args.dataset == "emnist":
        args.batch_size = 16
        
    if args.dataset == "tinyimagenet":
        args.model_name = "resnet18"
    
    args.model = model_dict[args.model_name](num_classes=args.num_classes, in_channels=in_channels)
    args.base_dataset = get_base_dataset(args)
    return args

def main(args):
    if args.method == "FedAvg":
        server = ServerFedAvg(args)
    elif args.method == "Local":
        server = ServerLocal(args)
    elif args.method == "pFedFDA":
        server = ServerFedFDA(args)
    else:
        raise NotImplementedError

    print(f"Training {args.method}...")
    
    server.train()
    print(f"Method took ({np.mean(server.train_times):.2f}, {np.std(server.train_times):.2f}) seconds per training iteration")
    print(f"Method took ({np.sum(server.round_times):.2f}) total seconds")

if __name__ == "__main__":
    args = parse_args()
    main(args)