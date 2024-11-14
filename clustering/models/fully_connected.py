import torch
from torch import nn

from . import Cnn


class Model(nn.Module):
    def __init__(self, d_model: int = 256):
        super().__init__()
        self.cnn = Cnn(d_model)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(d_model * 3, 2 * d_model)
        self.fc2 = nn.Linear(2 * d_model, 2 * d_model)
        self.out = nn.Linear(2 * d_model, 1)

    def forward(self, x):
        x = self.cnn(x)
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        out = self.out(x)
        return out
