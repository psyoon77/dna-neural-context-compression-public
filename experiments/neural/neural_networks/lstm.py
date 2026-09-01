import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMModel(nn.Module):
    def __init__(self, num_symbols=5, hidden_dim=64, num_layers=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(num_symbols, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_symbols)

    def forward(self, x):
        # x shape: (batch_size, sequence_length)  # (batch_size, sequence_length, embedding_dim)
        output, _ = self.lstm(x)  # output: (batch_size, sequence_length, hidden_dim)
        output = output[:, -1, :]  # Take the output at the last time step
        logits = self.fc(output)   # (batch_size, num_symbols)
        return logits
