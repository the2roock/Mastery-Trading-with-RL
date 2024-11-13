import torch
from torch import nn


class BiLSTM(nn.Module):    
    def __init__(self, input_shape: int = 4, hidden_shape: int = 64, layers: int = 2):
        super().__init__()

        self.bi_lstm = nn.LSTM(input_shape, hidden_shape, num_layers=layers, batch_first=True, bidirectional=True)

        self.fc1 = nn.Linear(2*hidden_shape, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.out = nn.Linear(256, 1)

    def forward(self, x):
        h, c = self.bi_lstm(x)
        x1 = torch.relu(h[:, -1, :])

        x2 = torch.relu(self.fc1(x1))
        x3 = torch.relu(self.fc2(x2))

        out = self.out(x3)
        return out
