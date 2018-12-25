"""
Definition of Net-12 and Net-48 models

Inspired by
Zhang, K., Zhang, Z., Li, Z. and Qiao, Y., 2016. Joint face detection and alignment using 
multitask cascaded convolutional networks. 
IEEE Signal Processing Letters, 23(10), pp.1499-1503.

Zuzeng Lin, 2018
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def init_xavier(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
        nn.init.constant_(m.bias, 0.1)


class Net12(nn.Module):
    def __init__(self, is_train=False):
        # define layers
        super(Net12, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 10, kernel_size=3, stride=1),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(10, 16, kernel_size=3, stride=1),
            nn.PReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1),
            nn.PReLU()
        )
        self.fc1_classes = nn.Conv2d(32, 1, kernel_size=1, stride=1)
        self.fc1_bbox = nn.Conv2d(32, 4, kernel_size=1, stride=1)
        self.apply(init_xavier)

    def forward(self, x):
        # forward propration
        # CNN feature extraction
        x = self.features(x)
        # regression and classification tasks
        classes = torch.sigmoid(self.fc1_classes(x))
        bbox = self.fc1_bbox(x)
        return classes, bbox


class Net48(nn.Module):

    def __init__(self, is_train=False):
        # define layers
        super(Net48, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=2, stride=1),
            nn.PReLU()
        )
        self.fc1 = nn.Linear(128*2*2, 256)
        self.prelu1 = nn.PReLU()
        self.fc2_classes = nn.Linear(256, 1)
        self.fc2_bbox = nn.Linear(256, 4)
        self.fc2_landmarks = nn.Linear(256, 10)
        # init weight with xavier
        self.apply(init_xavier)

    def forward(self, x):
        # forward propration
        # CNN feature extraction
        x = self.features(x)
        x = x.view(x.size(0), -1)
        # perceptron
        x = self.fc1(x)
        x = self.prelu1(x)
        # regression and classification tasks
        classes = torch.sigmoid(self.fc2_classes(x))
        bbox = self.fc2_bbox(x)
        landmark = self.fc2_landmarks(x)
        return classes, bbox, landmark
