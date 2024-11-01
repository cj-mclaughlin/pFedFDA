from .resnet_gn import resnet8, resnet18, resnet50
from .cnn import CIFARNet, EMNISTNet

model_dict = {
    "cifarnet": CIFARNet,
    "emnistnet": EMNISTNet,
    "resnet8": resnet8,
    "resnet18": resnet18,
    "resnet50": resnet50
}