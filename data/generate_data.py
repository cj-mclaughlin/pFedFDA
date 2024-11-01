import numpy as np
import os
import sys
import random
import torchvision
from pathlib import Path
from torch.utils.data import ConcatDataset
from utils.loader import DigitsDataset, OfficeDataset, TinyImagenetVal
from utils.partition import separate_data, save_file
from utils.parser import parse_args
import os

random.seed(42)
np.random.seed(42)

DOWNLOAD = True

# environment variable pointing to stored datasets, e.g. ImageNet
DATA_PATH = os.environ["DATA_PATH"]

# allocate data to users
# partition will be saved to args.partition_path
def generate_dataset(args, partition):
    # Setup directory
    if not os.path.exists(Path(os.curdir)/"partition"):
        os.makedirs(Path(os.curdir)/"partition")

    # Load dataset
    args, dataset = load_dataset(args)
    NUM_CLASSES = args.num_classes

    # Partition the data for clients, keeping track of labels and indices
    if isinstance(dataset, ConcatDataset):
        indices = np.arange(len(dataset))
        labels = np.concatenate([dataset.datasets[i].targets for i in range(len(dataset.datasets))])
    else:
        indices = np.arange(len(dataset))
        labels = np.array(dataset.targets)

    X, y, statistic = separate_data((indices, labels), NUM_CLASSES, partition, args)
    
    # save indices to disk for later use
    save_file(X, y, statistic, partition, args)

def load_dataset(args, download=DOWNLOAD):
    FULL_PATH = Path(DATA_PATH) / f"{args.dataset}"
    if args.dataset == "cifar10":
        args.num_classes = 10
        train_dataset = torchvision.datasets.CIFAR10(
            root=FULL_PATH, train=True, download=download, transform=None)
        test_dataset = torchvision.datasets.CIFAR10(
            root=FULL_PATH, train=False, download=download, transform=None)
        
    elif args.dataset == "mnist":
        args.num_classes = 10
        train_dataset = torchvision.datasets.MNIST(
            root=FULL_PATH, train=True, download=download, transform=None)
        test_dataset = torchvision.datasets.MNIST(
            root=FULL_PATH, train=False, download=download, transform=None)

    elif args.dataset == "fmnist":
        args.num_classes = 10
        train_dataset = torchvision.datasets.FashionMNIST(
            root=FULL_PATH, train=True, download=download, transform=None)
        test_dataset = torchvision.datasets.FashionMNIST(
            root=FULL_PATH, train=False, download=download, transform=None)
    
    elif args.dataset == "cifar100":
        args.num_classes = 100
        train_dataset = torchvision.datasets.CIFAR100(
            root=FULL_PATH, train=True, download=download, transform=None)
        test_dataset = torchvision.datasets.CIFAR100(
            root=FULL_PATH, train=False, download=download, transform=None)

    elif args.dataset == "tinyimagenet":
        args.num_classes = 200
        train_dataset = torchvision.datasets.ImageFolder(
            root=FULL_PATH / "train", transform=None
        )
        test_dataset = TinyImagenetVal(root=FULL_PATH / "val", class_to_idx=train_dataset.class_to_idx)

    elif "Digit5" in args.dataset:
        if args.dataset == "Digit5_MNIST":
            args.num_classes = 10
            train_dataset = DigitsDataset(data_path=Path(DATA_PATH) / f"digit_dataset/MNIST", start_idx=0, parts=10, channels=1, train=True, transform=None)
            test_dataset = DigitsDataset(data_path=Path(DATA_PATH) / f"digit_dataset/MNIST", start_idx=0, channels=1, train=False, transform=None)
        
        elif args.dataset == "Digit5_SVHN":
            args.num_classes = 10
            train_dataset = DigitsDataset(data_path=Path(DATA_PATH) / f"digit_dataset/SVHN", start_idx=0, parts=10, channels=3, train=True, transform=None)
            test_dataset = DigitsDataset(data_path=Path(DATA_PATH) / f"digit_dataset/SVHN", start_idx=0, channels=3, train=False, transform=None)

        elif args.dataset == "Digit5_USPS":
            args.num_classes = 10
            train_dataset = DigitsDataset(data_path=Path(DATA_PATH) / f"digit_dataset/USPS", start_idx=0, parts=10, channels=1, train=True, transform=None)
            test_dataset = DigitsDataset(data_path=Path(DATA_PATH) / f"digit_dataset/USPS", start_idx=0, channels=1, train=False, transform=None)

        elif args.dataset == "Digit5_SynthDigits":
            args.num_classes = 10
            train_dataset = DigitsDataset(data_path=Path(DATA_PATH) / f"digit_dataset/SynthDigits", start_idx=0, parts=10, channels=3, train=True, transform=None)
            test_dataset = DigitsDataset(data_path=Path(DATA_PATH) / f"digit_dataset/SynthDigits", start_idx=0, channels=3, train=False, transform=None)

        elif args.dataset == "Digit5_MNIST_M":
            args.num_classes = 10
            train_dataset = DigitsDataset(data_path=Path(DATA_PATH) / f"digit_dataset/MNIST_M", start_idx=0, parts=10, channels=3, train=True, transform=None)
            test_dataset = DigitsDataset(data_path=Path(DATA_PATH) / f"digit_dataset/MNIST_M", start_idx=0, channels=3, train=False, transform=None)

    elif "Office10" in args.dataset:
        if args.dataset == "Office10_amazon":
            args.num_classes = 10
            train_dataset = OfficeDataset(data_path=Path(DATA_PATH) / f"office_caltech_10/amazon_train.pkl", transform=None)
            test_dataset = OfficeDataset(data_path=Path(DATA_PATH) / f"office_caltech_10/amazon_test.pkl", transform=None)
        
        elif args.dataset == "Office10_caltech":
            args.num_classes = 10
            train_dataset = OfficeDataset(data_path=Path(DATA_PATH) / f"office_caltech_10/caltech_train.pkl", transform=None)
            test_dataset = OfficeDataset(data_path=Path(DATA_PATH) / f"office_caltech_10/caltech_test.pkl", transform=None)

        elif args.dataset == "Office10_dslr":
            args.num_classes = 10
            train_dataset = OfficeDataset(data_path=Path(DATA_PATH) / f"office_caltech_10/dslr_train.pkl",transform=None)
            test_dataset = OfficeDataset(data_path=Path(DATA_PATH) / f"office_caltech_10/dslr_test.pkl", transform=None)

        elif args.dataset == "Office10_webcam":
            args.num_classes = 10
            train_dataset = OfficeDataset(data_path=Path(DATA_PATH) / f"office_caltech_10/webcam_train.pkl", transform=None)
            test_dataset = OfficeDataset(data_path=Path(DATA_PATH) / f"office_caltech_10/webcam_test.pkl", transform=None)

    dataset = ConcatDataset([train_dataset, test_dataset])
    return args, dataset

if __name__ == "__main__":
    partition = sys.argv[1]
    args = parse_args()
    generate_dataset(args, partition)