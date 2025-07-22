import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

class CharLSTMGenerator(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super(CharLSTMGenerator, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, hidden=None):
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out)
        return out, hidden
