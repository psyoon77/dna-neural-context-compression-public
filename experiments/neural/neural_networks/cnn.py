import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNModel(nn.Module):
    def __init__(self, num_symbols=5, num_filters=64, kernel_size=3):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=num_symbols, out_channels=num_filters, kernel_size=kernel_size)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(num_filters, num_symbols)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, embedding_dim)
        x = x.permute(0, 2, 1)  # (batch_size, embedding_dim, sequence_length)
        x = F.relu(self.conv1(x))  # (batch_size, num_filters, L_out)
        x = self.pool(x).squeeze(2)  # (batch_size, num_filters)
        logits = self.fc(x)  # (batch_size, num_symbols)
        return logits

