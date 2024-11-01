import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_clients", required=True, type=int)
    parser.add_argument("--train_prop", required=True, type=float)
    parser.add_argument("--num_samples", required=False, default=500, type=int)
    parser.add_argument("--name", required=True, type=str)
    parser.add_argument("--dataset_split", required=False, default="train", type=str)
    #----------------- partition specific parameters -------------------------------
    parser.add_argument("--dirichlet_alpha", required=False, default=0.1, type=float)
    parser.add_argument("--num_shards", required=False, default=2, type=int)
    parser.add_argument("--dataset",required=False, default="cifar10", type=str)
    
    return parser.parse_known_args()[0]