import os
import ujson
import numpy as np
import random
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm 
import math 

random.seed(42)
np.random.seed(42)

BATCH_SIZE = 50 
TRAIN_PROP = 0.8
MIN_SIZE = BATCH_SIZE / TRAIN_PROP  # at least one training batch per client

def separate_data(data, num_classes, partition, args):
    num_clients = args.num_clients
    dir_alpha = args.dirichlet_alpha
    num_shards = args.num_shards

    X = [[] for _ in range(num_clients)]
    y = [[] for _ in range(num_clients)]
    statistic = [[] for _ in range(num_clients)]

    dataset_content, dataset_label = data

    dataidx_map = {}

    #------------------------------------------------------------------------------------------------------------
    if partition == "dirichlet":
        """
        Client data distributions ~ Dirichlet(alpha)
        """
        ind, labels = data
        if not isinstance(labels, np.ndarray):
            labels = np.array(labels)
        num_samples = labels.shape[0]

        min_size = 0
        while min_size < MIN_SIZE:
            idx_batch = [[] for _ in range(num_clients)]
            # for each class in the dataset
            for k in range(num_classes):
                idx_k = np.where(labels == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(
                    np.repeat(dir_alpha, num_clients))
                # Balance
                proportions = np.array(
                    [p * (len(idx_j) < num_samples / num_clients) for p, idx_j in
                    zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in
                            zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(num_clients):
            dataidx_map[j] = np.array(idx_batch[j])
    #------------------------------------------------------------------------------------------------------------
    elif partition == "dirichlet_limited":
        """
        Client data distributions ~ Dirichlet(alpha), 
        and total client data volume is set to a specified number
        """
        dir_return = None
        samples_per_client = args.num_samples
        if samples_per_client > len(data[0]) / num_clients:
            samples_per_client = math.floor(len(data[0])/num_classes)
        
        while dir_return == None:
            dir_return = dirichlet_limited(samples_per_client, dataset_label, args)
            samples_per_client -= 50
            if samples_per_client < 0:
                print("Invalid parameters, could not partition samples.")
                raise NotImplementedError
        
        idx_batch = dir_return
    
        for j in range(num_clients):
            dataidx_map[j] = idx_batch[j]
    #------------------------------------------------------------------------------------------------------------
    elif partition == "shards":
        """
        simplified implementation of
        https://github.com/lgcollins/FedRep/blob/main/utils/sampling.py
        """
        idxs_dict = {}
        ind, labels = data
        count = len(labels)
        idxs_dict = {c: ind[labels == c] for c in np.unique(labels)}

        shard_per_class = int(num_shards * num_clients / num_classes)
        samples_per_user = int(count / num_clients)

        if num_shards > num_classes:
            print("More shards than available classes!")
            raise NotImplementedError
        
        for label in idxs_dict.keys():
            x = idxs_dict[label]
            num_leftover = len(x) % shard_per_class
            leftover = x[-num_leftover:] if num_leftover > 0 else []
            x = np.array(x[:-num_leftover]) if num_leftover > 0 else np.array(x)
            x = x.reshape((shard_per_class, -1))
            x = list(x)

            for i, idx in enumerate(leftover):
                x[i] = np.concatenate([x[i], [idx]])
            idxs_dict[label] = x

        rand_set_all = list(range(num_classes)) * shard_per_class
        random.shuffle(rand_set_all)
        rand_set_all = np.array(rand_set_all).reshape((num_clients, -1))

        # divide and assign
        for i in range(num_clients):
            rand_set_label = rand_set_all[i]
            rand_set = []
            for label in rand_set_label:
                idx = np.random.choice(len(idxs_dict[label]), replace=False)
                rand_set.append(idxs_dict[label].pop(idx))
            
            dataidx_map[i] = np.concatenate(rand_set)
    #------------------------------------------------------------------------------------------------------------
    else:
        raise NotImplementedError

    # assign data
    for client in tqdm(range(num_clients)):
        idxs = dataidx_map[client]
        X[client] = dataset_content[idxs]
        y[client] = dataset_label[idxs]
        for i in np.unique(y[client]):
            statistic[client].append((int(i), int(sum(y[client]==i))))
    del data
    return X, y, statistic

def dirichlet_limited(samples_per_client, dataset_label, args):
    n_data_per_clnt = samples_per_client
    num_clients = args.num_clients
    num_classes = args.num_classes

    clnt_data_list = (np.ones(num_clients) * n_data_per_clnt).astype(int)
    cls_priors = np.random.dirichlet(alpha=[args.dirichlet_alpha] * num_classes, size=num_clients)
    prior_cumsum = np.cumsum(cls_priors, axis=1)
    idx_list = [np.where(dataset_label == i)[0] for i in range(num_classes)]
    cls_amount = [len(idx_list[i]) for i in range(num_classes)]

    idx_batch = {cid: [] for cid in range(num_clients)}
    while(np.sum(clnt_data_list) != 0):
        curr_clnt = np.random.randint(num_clients)
        if clnt_data_list[curr_clnt] <= 0:
            continue
        clnt_data_list[curr_clnt] -= 1
        curr_prior = prior_cumsum[curr_clnt]
        num_fail_iters = 0
        while True:
            cls_label = np.argmax(np.random.uniform() <= curr_prior)
            # Redraw class label if trn_y is out of that class
            if cls_amount[cls_label] <= 0:
                num_fail_iters += 1
                if num_fail_iters >= 1000:
                    print("Samples per client cannot be satisfied.")
                    print("We will try to adjust the local data volume!")
                    return None
                continue
            cls_amount[cls_label] -= 1
            idx_batch[curr_clnt].append(idx_list[cls_label][cls_amount[cls_label]])
            break

    return idx_batch

def save_file(X, y, statistic, partition, args):
    config = {
        'num_clients': args.num_clients, 
        'num_classes': args.num_classes,
        'partition': partition, 
        'label_distribution': statistic, 
        'dirichlet_alpha': args.dirichlet_alpha, 
        'num_shards': args.num_shards,
    }
    partition_path = Path("partition") / f"{args.name}"
    if not os.path.exists(partition_path):
        os.mkdir(partition_path)

    config_path = Path("partition") / f"{args.name}.json"
    with open(config_path, 'w') as f:
        ujson.dump(config, f)

    # save train/test sample indices for each client
    for client_idx in range(args.num_clients):
        X_train, X_test, _, _ = train_test_split(
            X[client_idx], y[client_idx], train_size=args.train_prop, shuffle=True, random_state=42)
        save_path = Path(partition_path) / f"client_{client_idx}.npz"
        np.savez(save_path, train=X_train, test=X_test)