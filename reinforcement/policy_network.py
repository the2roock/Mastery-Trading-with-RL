import os

import torch
from torch import nn


class Model(nn.Module):
    def __init__(self, input_shape: int, output_shape: int = 1, d_model: int = 256):
        super().__init__()
        self.input_shape = input_shape
        self.d_model = d_model
        self.output_shape = output_shape

        self.fc1 = nn.Linear(self.input_shape, self.d_model)
        self.fc2 = nn.Linear(self.d_model, self.d_model)
        self.fc3 = nn.Linear(self.d_model, self.output_shape)

    def forward(self, x):
        x = nn.functional.selu(self.fc1(x))
        x = nn.functional.selu(self.fc2(x))
        out = self.fc3(x)
        return out


def load_trained() -> Model:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "static", "PolicyNetwork-v1.pkl")
    model = torch.load(path, map_location=device)
    return model