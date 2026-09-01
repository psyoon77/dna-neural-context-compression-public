import os
import random
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from dna_sequence_dataset import char_to_symbol

class DNASequenceDataset(Dataset):
    def __init__(self, directory_path: str, num_symbols: int = 5, sampling_ratio=1.0):
        self.inputs = []
        self.targets = []
        self.num_symbols = num_symbols
        self.sampling_ratio = sampling_ratio

        # Directory contains files with DNA sequences
        for filename in tqdm(os.listdir(directory_path)):
            filepath = os.path.join(directory_path, filename)
            if os.path.isfile(filepath):
                if os.path.splitext(filepath)[1] == "":  # Make sure no extension
                    with open(filepath, 'r') as file:
                        sequence = file.read().strip() + '$'  # Add $ to the end
                        symbols = [char_to_symbol(c) for c in sequence]
                        self.inputs.extend(symbols[:-1])   # All symbols except the last
                        self.targets.extend(symbols[1:])   # All symbols except the first

        assert len(self.inputs) == len(self.targets)
        print(f'Loaded {len(self.inputs)} symbols from {directory_path}')

        # 샘플링 비율 적용
        if self.sampling_ratio < 1.0:
            sample_size = int(len(self.inputs) * self.sampling_ratio)
            sampled_indices = random.sample(range(len(self.inputs)), sample_size)
            self.inputs = [self.inputs[i] for i in sampled_indices]
            self.targets = [self.targets[i] for i in sampled_indices]
            print(f'Sampled {len(self.inputs)} symbols (Sampling ratio: {self.sampling_ratio})')
            
        # Convert inputs and targets to tensors
        self.inputs = torch.tensor(self.inputs, dtype=torch.long)
        self.targets = torch.tensor(self.targets, dtype=torch.long)

        # Precompute one-hot encodings
        self.input_onehot = torch.nn.functional.one_hot(self.inputs, num_classes=num_symbols).float()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.input_onehot[idx], self.targets[idx]
