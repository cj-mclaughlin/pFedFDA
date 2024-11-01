from torch import nn
import torch.nn.functional as F

"""
Small CNN Architectures taken from
https://github.com/JianXu95/FedPAC/blob/main/models/cnn.py
"""

class CIFARNet(nn.Module):
    def __init__(self, num_classes=10, in_channels=3):
        super(CIFARNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 5, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.flat_size = 64 * 3 * 3
        self.linear = nn.Linear(self.flat_size, 128)
        self.fc = nn.Linear(128, num_classes)
        self.D = 128
        self.cls = num_classes

    def forward(self, x, return_feat=False):
        x = self.pool(F.leaky_relu(self.conv1(x)))
        x = self.pool(F.leaky_relu(self.conv2(x)))
        x = self.pool(F.leaky_relu(self.conv3(x)))
        x = x.view(-1, self.flat_size)
        x = F.leaky_relu(self.linear(x))
        out = self.fc(x)
        if return_feat:
            return x, out
        return out

class EMNISTNet(nn.Module):
    def __init__(self, num_classes=62, in_channels=1):
        super(EMNISTNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 5, padding=1)
        self.flat_size = 32 * 5 * 5
        self.linear = nn.Linear(self.flat_size, 128)
        self.fc = nn.Linear(128, num_classes)
        self.D = 128
        self.cls = num_classes

    def forward(self, x, return_feat=False):
        x = self.pool(F.leaky_relu(self.conv1(x)))
        x = self.pool(F.leaky_relu(self.conv2(x)))
        x = x.view(-1, self.flat_size)
        x = F.leaky_relu(self.linear(x))
        out = self.fc(x)
        if return_feat:
            return x, out
        return out

