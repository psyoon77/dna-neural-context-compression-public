import torch
from torch.utils.data import Dataset, DataLoader

class DNASequenceDataset(Dataset):
    def __init__(self, data, num_symbols: int = 5): # A, G, T, C, EOF
        # data is a list or array of integer-encoded tokens
        self.data = data

    def __len__(self):
        return len(self.data) - 1

    def __getitem__(self, idx):
        input_token = self.data[idx]
        target_token = self.data[idx + 1]
        return torch.tensor([input_token], dtype=torch.long), torch.tensor(target_token, dtype=torch.long)

    def symbol_to_onehot(self, symbol, num_symbols):
        onehot = [0] * num_symbols
        onehot[symbol] = 1
        return onehot

def char_to_symbol(char):
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3, '$': 4} # $ for EOF
    return mapping[char.upper()]

def symbol_to_char(symbol):
    mapping = {0: 'A', 1: 'C', 2: 'G', 3: 'T', 4: '$'} # $ for EOF
    return mapping[symbol]

# Example usage
# Assuming 'sequence' is a list of integer-encoded DNA bases
# dataset = DNASequenceDataset(sequence)
# dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
