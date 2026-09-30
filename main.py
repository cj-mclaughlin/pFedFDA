import argparse
import os
import torch
import numpy as np
from data.utils.loader import get_base_dataset
from servers.server_fedavg import ServerFedAvg
from servers.server_fedfda import ServerFedFDA
from servers.server_local import ServerLocal
from models import model_dict, CLIP_MODELS

# Only the first NUM_TRAIN_CLIENTS clients participate in training; the remaining clients are held out for evaluation
NUM_TRAIN_CLIENTS = 50

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
    parser.add_argument("--global_rounds", default=10, type=int)
    parser.add_argument("--local_epochs", default=1, type=int)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--momentum", default=0.5, type=float)
    parser.add_argument("--wd", default=5e-4, type=float)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--eval_gap", default=200, type=float, help="Rounds Between Model Evaluation")
    parser.add_argument("--train_prop", default=1.0, type=float, help="Proportion of Training Data To Use")
    
    # FL/Server Setup
    parser.add_argument("--method", default="pFedFDA", type=str)
    parser.add_argument("--num_clients", default=100, type=int)
    parser.add_argument("--sampling_prob", default=1.0, type=float, help="Client Participation Probability")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", type=str)
    
    # model architecture (Added CLIP)
    parser.add_argument("--model_name", default="clip", type=str, help="Model Architecture", choices=["cnn", "resnet18"] + CLIP_MODELS)   
    
    # method-specific hyperparameters
    parser.add_argument("--p_epochs", default=1, type=int, help="Number of Personalization Epochs") 
    parser.add_argument("--single_beta", action="store_true", help="if we should only use a single beta term (pFedFDA)") 
    parser.add_argument("--local_beta", action="store_true", help="if we should only use only local statistics (pFedFDA)")
    
    # --- NEW EXPERIMENTAL FLAGS ---
    parser.add_argument("--clip_arch", default="ViT-B/32", type=str, help="CLIP architecture to load")
    parser.add_argument("--normalize_features", action="store_true", help="L2 normalize CLIP features before FDA")
    parser.add_argument("--eval_ood_clients", action="store_true", help="Evaluate generalization on new, corrupted clients post-training")
    parser.add_argument("--zero_shot", action="store_true", help="Bypass FL training and run zero-shot CLIP baseline")
    parser.add_argument("--goodness_of_fit", action="store_true", help="Compute normality metrics (e.g., Mardia's) on local feature sets")

    # ProbVLM adapter training (--model_name probvlm)
    parser.add_argument("--probvlm_lr", default=1e-4, type=float, help="Adam learning rate for the ProbVLM adapter")
    parser.add_argument("--probvlm_T1", default=1.0, type=float, help="weight of the L1 term in the ProbVLM loss")
    parser.add_argument("--probvlm_T2", default=1e-4, type=float, help="weight of the generalized Gaussian NLL term in the ProbVLM loss")
    parser.add_argument("--probvlm_cross_lambda", default=1e-4, type=float, help="weight of the cross-modal ProbVLM loss terms")
    
    # logging/saving
    parser.add_argument("--exp_name", default="baseline", type=str, help="save file prefix") 
    parser.add_argument("--checkpoint_path", default=None, type=str, help="checkpoint file (default: results/<exp_name>/checkpoint_latest.pt)")
    parser.add_argument("--from_checkpoint", default=True, action=argparse.BooleanOptionalAction, help="skip training; load global model + all Gaussian statistics from --checkpoint_path (--no-from_checkpoint to train)")
    args = parser.parse_args()

    if args.checkpoint_path is None:
        args.checkpoint_path = f"results/{args.exp_name}/checkpoint_latest.pt"
    if args.from_checkpoint:
        assert args.method == "pFedFDA", "--from_checkpoint is only supported for pFedFDA"

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
        if args.model_name not in CLIP_MODELS:
            args.model_name = "resnet18"
    
    # Load model (requires model_dict to handle "clip" initialization)
    args.model = model_dict[args.model_name](
        num_classes=args.num_classes, 
        in_channels=in_channels, 
        args=args # Pass args to handle normalize_features / clip_arch
    ).cpu()
    
    args.base_dataset = get_base_dataset(args)
    return args

def main(args):
    # Phase 3: Zero-Shot Baseline Bypass
    if args.zero_shot:
        assert args.model_name in CLIP_MODELS, "Zero-shot evaluation requires the CLIP model."
        print(f"Running Zero-Shot CLIP evaluation on {args.dataset}...")
        # TODO: Implement zero-shot evaluation logic
        # e.g., evaluate_zeroshot(args.model, args.base_dataset, args.device)
        return

    # Standard Server Initialization
    if args.method == "FedAvg":
        server = ServerFedAvg(args)
    elif args.method == "Local":
        server = ServerLocal(args)
    elif args.method == "pFedFDA":
        server = ServerFedFDA(args)
    else:
        raise NotImplementedError

    if args.normalize_features:
        print("Feature normalization is ENABLED.")

    out_dir = os.path.dirname(args.checkpoint_path) or "."
    assert args.num_clients > NUM_TRAIN_CLIENTS, f"need more than {NUM_TRAIN_CLIENTS} clients to hold some out"

    # Restrict the server to the training population
    all_clients = server.clients
    server.clients = all_clients[:NUM_TRAIN_CLIENTS]
    server.num_clients = NUM_TRAIN_CLIENTS

    if args.from_checkpoint:
        # Skip training: restore global model + server/client distribution estimates
        server.load_checkpoint(args.checkpoint_path)
        ptest_acc, ptest_loss, ptest_acc_std = server.evaluate_checkpoint()
        print(f"Checkpoint Test Loss [{ptest_loss:.4f}]\t Test Acc [{ptest_acc:.2f} (±{ptest_acc_std:.2f})]")
    else:
        # Phase 1: Standard Training & In-Distribution Feature Fit
        print(f"Training {args.method} with model {args.model_name} on clients 0-{NUM_TRAIN_CLIENTS - 1}...")
        server.train()

        print(f"Method took ({np.mean(server.train_times):.2f}, ±{np.std(server.train_times):.2f}) seconds per training iteration")
        print(f"Method took ({np.sum(server.round_times):.2f}) total seconds")

    if args.method == "pFedFDA":
        # Optional: Compute goodness of fit on the training clients
        server.compute_goodness_of_fit(save_path=os.path.join(out_dir, "gaussian_fit_2d.png"))

    # Restore the full population for evaluation
    server.clients = all_clients
    server.num_clients = len(all_clients)

    # Phase 2: Evaluate the training population, the held-out clients, and corrupted versions of the held-out clients
    if args.method == "pFedFDA":
        heldout_ids = range(NUM_TRAIN_CLIENTS, len(all_clients))
        server.evaluate_populations({
            "train": all_clients[:NUM_TRAIN_CLIENTS],
            "heldout": all_clients[NUM_TRAIN_CLIENTS:],
            "heldout_corrupted": server.corrupted_clients(heldout_ids),
        }, out_dir)

if __name__ == "__main__":
    args = parse_args()
    main(args)