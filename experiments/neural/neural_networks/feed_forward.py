import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleFeedforwardModel(nn.Module):
    def __init__(self, num_symbols, hidden_dim=64):
        super(SimpleFeedforwardModel, self).__init__()
        self.fc1 = nn.Linear(num_symbols, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_symbols)

    def forward(self, x):
        # x is one-hot encoded
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)
        return logits
