from abc import ABC, abstractmethod
from copy import deepcopy
import numpy as np
from data.utils.loader import get_augmented_partition, get_partition, get_digit5, get_office10
from torch.utils.data import DataLoader, Subset
import torch
import os

class Client(ABC):
    def __init__(self, args, client_idx):
        self.D = args.model.D
        self.model = deepcopy(args.model)
        self.model_path = f"results/{args.exp_name}/client_{client_idx}.ckpt"
        self.partition_path = args.partition_path
        self.client_idx = client_idx
        self.batch_size = args.batch_size
        self.local_epochs = args.local_epochs
        self.num_classes = args.num_classes
        self.device = args.device
        self.lr = args.lr
        self.model_name = args.model_name
        self.momentum = args.momentum
        self.wd = args.wd
        self.loss = torch.nn.CrossEntropyLoss()
        self.last_acc = -1
        self.last_loss = -1
        self.train_prop = args.train_prop
        self.val_prop = 1 - args.train_prop
        self.setup_dataset(args) 
        self.label_distribution = list(self.get_label_distribution()[1].cpu().numpy())
        
    def save_model(self):
        torch.save(self.model.state_dict(), self.model_path)

    def partial_dataset(self, dataset):
        """
        allocate training samples for validation 
        (also used for down-sampling the training set)
        """
        num_samples = len(dataset)
        num_keep = int(self.train_prop * num_samples)
        indices = [i for i in range(len(dataset))]
        train_indices = np.random.choice(indices, size=num_keep, replace=False)
        indices = list(set(indices) - set(train_indices))  # set of remaining samples
        num_val = int(self.val_prop * len(indices))
        val_indices = np.random.choice(indices, size=num_val, replace=False)
        train_dataset = Subset(dataset, indices=train_indices)
        val_dataset = Subset(dataset, indices=val_indices)
        return train_dataset, val_dataset

    def setup_dataset(self, args):
        if args.dataset == "emnist":
            self.setup_emnist_dataset(args)
        elif args.dataset == "digit5":
            self.setup_digit5_dataset(args)
        elif args.dataset == "office10":
            self.setup_office10_dataset(args)
        elif args.augmented and self.client_idx < 50:
            self.setup_augmented_dataset(args)
            print("Loaded augmented dataset for client", self.client_idx)
        else:
            indices = np.load(f"data/partition/{self.partition_path}/client_{self.client_idx}.npz")
            train_dataset, test_dataset = get_partition(args, indices)
            self.trainset, self.valset = self.partial_dataset(train_dataset)
            self.testset = test_dataset
            self.num_train = len(self.trainset)
            self.num_test = len(self.testset)
            self.num_val = len(self.valset)
            self.augmentation = "None"
            self.severity = 0
    
    def setup_digit5_dataset(self, args):
        datasets = [
            "MNIST",
            "SVHN",
            "USPS",
            "SynthDigits",
            "MNIST_M"
        ]
        client_paths = []
        client_datasets = []
        for dataset in datasets:
            dataset_path = f"data/partition/Digit5_{dataset}_{self.partition_path}/"
            listdir = os.listdir(f"data/partition/Digit5_{dataset}_{self.partition_path}/")
            client_paths += [f"{dataset_path}{listdir[i]}" for i in range(len(listdir))]
            client_datasets += [dataset]*len(listdir)
        indices = np.load(client_paths[self.client_idx])
        train_dataset, test_dataset = get_digit5(client_datasets[self.client_idx], indices=indices)
        self.trainset, self.valset = self.partial_dataset(train_dataset)
        self.testset = test_dataset
        self.num_train = len(self.trainset)
        self.num_test = len(self.testset)
        self.num_val = len(self.valset)
        self.augmentation = "None"
        self.severity = 0

    def setup_office10_dataset(self, args):
        datasets = [
            "amazon",
            "caltech",
            "dslr",
            "webcam"
        ]
        client_paths = []
        client_datasets = []
        for dataset in datasets:
            dataset_path = f"data/partition/Office10_{dataset}_{self.partition_path}/"
            listdir = os.listdir(f"data/partition/Office10_{dataset}_{self.partition_path}/")
            client_paths += [f"{dataset_path}{listdir[i]}" for i in range(len(listdir))]
            client_datasets += [dataset]*len(listdir)
        indices = np.load(client_paths[self.client_idx])
        train_dataset, test_dataset = get_office10(client_datasets[self.client_idx], indices=indices)
        self.domain = datasets.index(client_datasets[self.client_idx])
        self.trainset, self.valset = self.partial_dataset(train_dataset)
        self.testset = test_dataset
        self.num_train = len(self.trainset)
        self.num_test = len(self.testset)
        self.num_val = len(self.valset)
        self.augmentation = "None"
        self.severity = 0

    def setup_augmented_dataset(self, args, augmentation=None, severity=None):
        indices = np.load(f"data/partition/{self.partition_path}/client_{self.client_idx}.npz")
        augmentations = [
            "motion_blur",
            "defocus_blur",
            "gaussian_noise",
            "shot_noise",
            "impulse_noise",
            "frost",
            "fog",
            "jpeg_compression",
            "brightness",
            "contrast"
        ]
        # by default, select one augmentation and severity level
        # assumes 100 total clients and 50 clients for augmentation
        if augmentation is None:
            self.augmentation = augmentations[self.client_idx % 10]
            self.severity = (self.client_idx % 5)+1
        # otherwise, use the given augmentation/severity pair
        else:
            self.augmentation = augmentation
            self.severity = severity
        train_dataset, test_dataset = get_augmented_partition(args, indices, self.augmentation, self.severity)
        self.trainset, self.valset = self.partial_dataset(train_dataset)
        self.testset = test_dataset
        self.num_train = len(self.trainset)
        self.num_val = len(self.valset)
        self.num_test = len(self.testset)

    def setup_emnist_dataset(self, args):
        indices = self.client_idx
        train_dataset, test_dataset = get_partition(args, indices)
        self.trainset, self.valset = self.partial_dataset(train_dataset)
        self.testset = test_dataset
        self.num_train = len(self.trainset)
        self.num_val = len(self.valset)
        self.num_test = len(test_dataset)
        self.augmentation = "None"
        self.severity = 0

    def set_model(self, global_model):
        self.model = deepcopy(global_model)

    def load_train_data(self, batch_size=None, drop_last=True):
        if batch_size is None:
            batch_size = min(self.batch_size, self.num_train)
        dataloader = DataLoader(self.trainset, batch_size=batch_size, shuffle=True, drop_last=drop_last)
        return dataloader

    def load_val_data(self, batch_size=None):
        if batch_size is None:
            batch_size = min(self.batch_size, self.num_val)
        dataloader = DataLoader(self.valset, batch_size=batch_size, shuffle=False, drop_last=False)
        return dataloader

    def load_test_data(self, batch_size=None):
        if batch_size is None:
            batch_size = min(self.batch_size, self.num_test)
        dataloader = DataLoader(self.testset, batch_size=batch_size, shuffle=False, drop_last=False)
        return dataloader
    
    def get_label_distribution(self, split="train"):
        if split == "train":
            loader = self.load_train_data(drop_last=False)
        else:
            loader = self.load_test_data()
        labels = []
        for i, (x, y) in enumerate(loader):
            labels.append(y.cpu().numpy())
        labels = np.concatenate(labels)
        label_counts = np.bincount(labels, minlength=self.num_classes)
        label_counts = torch.Tensor(label_counts).to(self.device)
        priors = np.bincount(labels, minlength=self.num_classes) / float(len(labels))
        priors = torch.Tensor(priors).float().to(self.device)
        return label_counts, priors
    
    def compute_feats(self, split="train"):
        with torch.no_grad():
            if split == "train":
                loader = self.load_train_data()
            else:
                loader = self.load_test_data()
            feats = []
            labels = []
            self.model.eval()
            self.model.to(self.device)
            for i, (x, y) in enumerate(loader):
                x = x.to(self.device)
                feat, _ = self.model.forward(x, return_feat=True)
                feats.append(feat)
                labels.append(y.cpu().numpy())
            feats = torch.cat(feats, axis=0)
            labels = np.concatenate(labels, axis=0)
        return feats, labels
    
    @abstractmethod
    def train(self):
        pass

    def evaluate(self):
        with torch.no_grad():
            total_loss = 0
            num_correct = 0
            num_samples = 0
            self.model.to(self.device)
            self.model.eval()
            dataloader = self.load_test_data()
            y_pred = []
            y_true = []
            for i, (x, y) in enumerate(dataloader):
                x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                loss = self.loss(output, y)
                total_loss += loss
                num_correct += torch.sum(torch.argmax(output, dim=1) == y)
                num_samples += x.size(0)
                y_pred.append(torch.argmax(output, dim=1).cpu().numpy())
                y_true.append(y.cpu().numpy())
            y_pred = np.concatenate(y_pred)
            y_true = np.concatenate(y_true)
            avg_loss = total_loss / len(dataloader)
            accuracy = (num_correct / num_samples) * 100
            self.last_acc = accuracy.cpu().numpy().item()
            self.model.to("cpu")
            return accuracy, avg_loss
    
    def evaluate_personalized(self):
        print("Evaluating without set personalization method.")
        return self.evaluate()