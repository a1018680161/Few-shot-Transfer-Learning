#!/usr/bin/python
# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.utils.weight_norm import WeightNorm


class CNNEncoder2d(nn.Module):
    """docstring for ClassName"""

    def __init__(self, feature_dim):
        super(CNNEncoder2d, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=0),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=0),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_dim, momentum=1, affine=True),
            nn.ReLU())
        self.admp = nn.AdaptiveMaxPool2d((5, 5))

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.admp(out)
        # out = out.view(out.size(0),-1)
        return out  # 64


class RelationNetwork2d(nn.Module):
    """docstring for RelationNetwork"""

    def __init__(self, input_size, hidden_size):
        super(RelationNetwork2d, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(input_size * 2, input_size, kernel_size=3, padding=1),
            nn.BatchNorm2d(input_size, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(input_size, input_size, kernel_size=3, padding=1),
            nn.BatchNorm2d(input_size, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = torch.sigmoid(self.fc2(out))
        return out


class CNNEncoder1d(nn.Module):
    """docstring for ClassName"""

    def __init__(self, feature_dim):
        super(CNNEncoder1d, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=10, padding=0, stride=3),
            nn.BatchNorm1d(64, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool1d(2))
        self.layer2 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, padding=0),
            nn.BatchNorm1d(64, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool1d(2))
        self.layer3 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64, momentum=1, affine=True),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv1d(64, feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(feature_dim, momentum=1, affine=True),
            nn.ReLU())
        self.admp = nn.AdaptiveMaxPool1d(25)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.admp(out)
        # out = out.view(out.size(0),-1)
        return out  # 64


class RelationNetwork1d(nn.Module):
    """docstring for RelationNetwork"""

    def __init__(self, input_size, hidden_size):
        super(RelationNetwork1d, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(input_size * 2, input_size, kernel_size=3, padding=1),
            nn.BatchNorm1d(input_size, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool1d(2))
        self.layer2 = nn.Sequential(
            nn.Conv1d(input_size, input_size, kernel_size=3, padding=1),
            nn.BatchNorm1d(input_size, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool1d(2))
        self.fc1 = nn.Linear(input_size * 6, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = torch.sigmoid(self.fc2(out))
        return out


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        n = m.weight.size(1)
        m.weight.data.normal_(0, 0.01)
        m.bias.data = torch.ones(m.bias.data.size())


class Classifier1d(nn.Module):
    """docstring for ClassName"""

    def __init__(self, input_size, out_size):
        super(Classifier1d, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(input_size, input_size, kernel_size=3, padding=1),
            nn.BatchNorm1d(input_size, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool1d(2))
        self.layer2 = nn.Sequential(
            nn.Conv1d(input_size, input_size, kernel_size=3, padding=1),
            nn.BatchNorm1d(input_size, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool1d(2))
        self.fc1 = nn.Linear(input_size * 6, out_size)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        return out


class CNN1d(nn.Module):
    """docstring for ClassName"""

    def __init__(self, feature_dim):
        super(CNN1d, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=10, padding=0, stride=3),
            nn.BatchNorm1d(64, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool1d(2))
        self.layer2 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, padding=0),
            nn.BatchNorm1d(64, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool1d(2))
        self.layer3 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64, momentum=1, affine=True),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv1d(64, feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(feature_dim, momentum=1, affine=True),
            nn.ReLU())
        self.admp = nn.AdaptiveMaxPool1d(25)
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.admp(out)
        out = out.view(out.size(0),-1)
        return out


class distLinear(nn.Module):
    def __init__(self, indim, outdim):
        super(distLinear, self).__init__()
        self.L = nn.Linear( indim, outdim, bias = False)
        self.class_wise_learnable_norm = True  #See the issue#4&8 in the github
        if self.class_wise_learnable_norm:
            WeightNorm.apply(self.L, 'weight', dim=0) #split the weight update component to direction and norm

        if outdim <=200:
            self.scale_factor = 2; #a fixed scale factor to scale the output of cos value into a reasonably large input for softmax, for to reproduce the result of CUB with ResNet10, use 4. see the issue#31 in the github
        else:
            self.scale_factor = 10; #in omniglot, a larger scale factor is required to handle >1000 output classes.

    def forward(self, x):
        x_norm = torch.norm(x, p=2, dim =1).unsqueeze(1).expand_as(x)
        x_normalized = x.div(x_norm+ 0.00001)
        if not self.class_wise_learnable_norm:
            L_norm = torch.norm(self.L.weight.data, p=2, dim =1).unsqueeze(1).expand_as(self.L.weight.data)
            self.L.weight.data = self.L.weight.data.div(L_norm + 0.00001)
        cos_dist = self.L(x_normalized) #matrix product by forward function, but when using WeightNorm, this also multiply the cosine distance by a class-wise learnable norm, see the issue#4&8 in the github
        scores = self.scale_factor* (cos_dist)

        return scores