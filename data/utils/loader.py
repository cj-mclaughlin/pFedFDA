import torch
from torch.utils.data import Dataset, ConcatDataset, Subset
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10, CIFAR100, ImageFolder
from torchvision import transforms as t
import os
from pathlib import Path
import pandas as pd
from PIL import Image
import numpy as np
import json

DOWNLOAD = True
DATA_PATH = os.environ["DATA_PATH"]

class DatasetFromSubset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)

def get_base_dataset(args):
    dataset_str = args.dataset
    data_path = Path(os.environ["DATA_PATH"]) / dataset_str
    if dataset_str == "cifar10":
        return load_cifar10(data_path)
    elif dataset_str == 'cifar100':
        return load_cifar100(data_path)
    elif dataset_str == 'mnist':
        return load_mnist(data_path)
    elif dataset_str == 'fmnist':
        return load_fmnist(data_path)
    elif dataset_str == "tinyimagenet":
        return load_tinyimagenet(data_path)
    else:
        return None
    
def get_partition(args, indices=None):
    base_dataset = args.base_dataset
    if args.dataset in ["emnist", "fmnist", "mnist"]:
        train_transform = t.Compose([
            t.ToTensor(),
            t.Normalize((0.5,), (0.5,))
        ])
        test_transform = t.Compose([
            t.ToTensor(),
            t.Normalize((0.5,), (0.5,))
        ])
    else:
        train_transform = t.Compose([
            t.ToTensor(),
            t.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        test_transform = t.Compose([
            t.ToTensor(),
            t.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    if args.dataset in ["cifar10", "cifar100", "mnist", "fmnist", "tinyimagenet"]:
        train_set = DatasetFromSubset(Subset(base_dataset, indices["train"]), transform=train_transform)
        test_set = DatasetFromSubset(Subset(base_dataset, indices["test"]), transform=test_transform)
    elif args.dataset == "emnist":
        with open(Path(DATA_PATH) / "emnist/client_names.json") as f:
            client_names = json.load(f)
        client_name = client_names[indices]
        train_set = EMNIST_WRITER(Path(DATA_PATH) / "emnist" / "train", client_name=client_name, transform=train_transform, split="train")
        test_set = EMNIST_WRITER(Path(DATA_PATH) / "emnist" / "test", client_name=client_name, transform=train_transform, split="test")
    return train_set, test_set

def get_augmented_partition(args, indices, augmentation, severity):
    transform = t.Compose([
        t.ToTensor(),
        t.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    full_X = np.load(Path(DATA_PATH) / f"{args.dataset}c" / f"{augmentation}_{severity}.npy")
    full_y = np.load(Path(DATA_PATH) / f"{args.dataset}c" / f"{augmentation}_{severity}_labels.npy")
    X_tr = full_X[indices["train"]]
    y_tr = full_y[indices["train"]]
    X_te = full_X[indices["test"]]
    y_te = full_y[indices["test"]]
    train_set = CIFARC(X_tr, y_tr, transform=transform)
    test_set = CIFARC(X_te, y_te, transform=transform)
    return train_set, test_set

def load_mnist(data_path, download=DOWNLOAD):
    train_set = MNIST(data_path, train=True, transform=None, download=download)
    test_set = MNIST(data_path, train=False, transform=None, download=download)
    full_set = ConcatDataset([train_set, test_set])
    return full_set

def load_fmnist(data_path, download=DOWNLOAD):
    train_set = FashionMNIST(data_path, train=True, transform=None, download=download)
    test_set = FashionMNIST(data_path, train=False, transform=None, download=download)
    full_set = ConcatDataset([train_set, test_set])
    return full_set

def load_cifar10(data_path, download=DOWNLOAD):
    train_set = CIFAR10(data_path, train=True, transform=None, download=download)
    test_set = CIFAR10(data_path, train=False, transform=None, download=download)
    full_set = ConcatDataset([train_set, test_set])
    return full_set
    
def load_cifar100(data_path, download=DOWNLOAD):
    train_set = CIFAR100(data_path, train=True, transform=None, download=download)
    test_set = CIFAR100(data_path, train=False, transform=None, download=download)
    full_set = ConcatDataset([train_set, test_set])
    return full_set

def load_tinyimagenet(data_path):
    train_set = ImageFolder(root=data_path / "train")
    test_set = TinyImagenetVal(root=data_path / "val", class_to_idx=train_set.class_to_idx)
    full_set = ConcatDataset([train_set, test_set])
    return full_set

class TinyImagenetVal(Dataset):
    # extract contents https://image-net.org/data/tiny-imagenet-200.zip to root
    def __init__(self, root, class_to_idx):
        df = pd.read_csv(root/"val_annotations.txt", sep="\t", header=None)
        self.image_paths = str(root) + "/images/" + df[0]
        self.targets = df[1].apply(lambda r: class_to_idx[r])
        
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, index):
        image = Image.open(os.path.join(self.image_paths[index])).convert("RGB")
        label = self.targets[index]
        return image, label
    
class EMNIST_WRITER(Dataset):
    def __init__(self, root, client_name, transform=None, split="train", preload=False):
        suffix = "tr" if split == "train" else "te"
        self.transform = transform
        self.X_path = root / f"{client_name}_X_{suffix}.npy"
        self.y_path = root / f"{client_name}_y_{suffix}.npy"
        self.y = np.load(self.y_path)
        self.preload = preload
        if self.preload:
            self.X = np.load(self.X_path)

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, index):
        if self.preload:
            image = self.X[index]
        else:
            image = np.load(self.X_path)[index]
        if self.transform is not None:
            image = self.transform(image)
        label = self.y[index]
        return image, label

class CIFARC(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = torch.LongTensor(targets)
        self.transform = transform
        
    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        
        if self.transform is not None:
            x = self.transform(x)
        return x, y
    
    def __len__(self):
        return len(self.data)
    
class DigitsDataset(Dataset):
    # https://github.com/yuetan031/FedPCL/blob/main/lib/data_utils.py
    def __init__(self, data_path, channels, start_idx, parts=1, train=True, transform=None):
        if train:
            all_images, all_labels = [], []
            for part in range(start_idx*parts, (start_idx+1)*parts):
                images, labels = np.load(os.path.join(data_path, 'partitions/train_part{}.pkl'.format(part)), allow_pickle=True)
                all_images.append(images)
                all_labels.append(labels)
            self.images = np.concatenate(all_images, axis=0)
            self.targets = np.concatenate(all_labels, axis=0)
        else:
            self.images, self.targets = np.load(os.path.join(data_path, 'test.pkl'), allow_pickle=True)

        self.transform = transform
        self.channels = channels
        self.targets = self.targets.astype(np.int64).squeeze()

        self.images, self.targets = torch.from_numpy(self.images), torch.from_numpy(self.targets)

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        image = self.images[idx]
        label = int(self.targets[idx])
        if self.channels == 1:
            image = Image.fromarray(image.numpy(), mode='L')
        elif self.channels == 3:
            image = Image.fromarray(image.numpy(), mode='RGB')
        else:
            raise ValueError("{} channel is not allowed.".format(self.channels))

        if self.transform is not None:
            image = self.transform(image)

        return image, label

def get_digit5(dataset, indices=None):
    if dataset in ["MNIST", "USPS"]:
        transform = t.Compose([
            t.Resize((32, 32)),
            t.Grayscale(num_output_channels=3),
            t.ToTensor(),
            t.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        channels = 1
    else:
        transform = t.Compose([
            t.Resize([32, 32]),
            t.ToTensor(),
            t.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        channels = 3

    trainset = DigitsDataset(data_path=Path(DATA_PATH) / f"digit_dataset/{dataset}", parts=10, start_idx=0, channels=channels, train=True, transform=transform)
    testset = DigitsDataset(data_path=Path(DATA_PATH) / f"digit_dataset/{dataset}", parts=10, start_idx=0, channels=channels, train=False, transform=transform)
    
    if indices:
        dataset = ConcatDataset([trainset, testset])
        trainset = DatasetFromSubset(Subset(dataset, indices["train"]), transform=None)
        testset = DatasetFromSubset(Subset(dataset, indices["test"]), transform=None)

    return trainset, testset

class OfficeDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.paths, self.text_labels = np.load(data_path, allow_pickle=True)
        label_dict={'back_pack':0, 'bike':1, 'calculator':2, 'headphones':3, 'keyboard':4, 'laptop_computer':5, 'monitor':6, 'mouse':7, 'mug':8, 'projector':9}
        self.targets = [label_dict[text] for text in self.text_labels]
        self.transform = transform
        self.data_path = data_path

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        img_path = os.path.join(DATA_PATH, self.paths[idx])
        label = self.targets[idx]
        image = Image.open(img_path)

        if len(image.split()) != 3:
            image = t.Grayscale(num_output_channels=3)(image)

        if self.transform is not None:
            image = self.transform(image)

        return image, label
    
def get_office10(dataset, indices=None):
    transform = t.Compose([
        t.Resize([32, 32]),
        t.ToTensor(),
        t.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    trainset = OfficeDataset(data_path=Path(DATA_PATH) / f"office_caltech_10/{dataset}_train.pkl", transform=transform)
    testset = OfficeDataset(data_path=Path(DATA_PATH) / f"office_caltech_10/{dataset}_test.pkl",  transform=transform)

    if indices:
        dataset = ConcatDataset([trainset, testset])
        trainset = DatasetFromSubset(Subset(dataset, indices["train"]), transform=None)
        testset = DatasetFromSubset(Subset(dataset, indices["test"]), transform=None)
    return trainset, testset