import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

class ResNet(nn.Module):
    def __init__(self,n):
        super().__init__()
        self.linear1 = nn.Linear(n,n)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(n,n)
        self.relu2 = nn.ReLU()
    def forward(self,x):
        id = x
        out = x
        out = self.linear1(out)
        out = self.relu1(out)
        out = self.linear2(out)+id
        out = self.relu2(out)
        return out

class model(nn.Module):
    def __init__(self):
        super().__init__()
        self.Linear1 = nn.Linear(6,32)
        self.resblock1 = ResNet(32)
        self.Linear2 = nn.Linear(32,128)
        self.resblock2 = ResNet(128)
        self.resblock3 = ResNet(128)
        self.Linear3 = nn.Linear(128,32)
        self.resblock4 = ResNet(32)
        self.Linear4 = nn.Linear(32,5)
    def forward(self,x):
        out = x
        out = self.Linear1(out)
        out = F.relu(out)
        out = self.resblock1(out)
        out = self.Linear2(out)
        out = F.relu(out)
        out = self.resblock2(out)
        out = self.resblock3(out)
        out = self.Linear3(out)
        out = F.relu(out)
        out = self.resblock4(out)
        out = self.Linear4(out)
        return out


mod2 = model()
mod2.load_state_dict(torch.load('/mnt/scratch/liwenrui/state_dict2.pt'))
