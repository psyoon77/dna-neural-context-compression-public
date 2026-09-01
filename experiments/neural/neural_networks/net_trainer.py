from dna_sequence_dataset import DNASequenceDataset, char_to_symbol
from feed_forward import SimpleFeedforwardModel
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch import optim


num_symbols = 5  # A, G, T, C, EOF
num_epochs = 10

# Assuming 'sequence' is a string of DNA bases (e.g., 'ACGTACGT...')
# TODO: Read dna file
sequence_symbols = [char_to_symbol(c) for c in sequence]

dataset = DNASequenceDataset(sequence_symbols)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

model = SimpleFeedforwardModel(num_symbols)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(num_epochs):
    for inputs, targets in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

# Save the trained model
torch.save(model, 'dna_feedforward_model.pth')
