import os
import torch
import torch.nn as nn
import torch.optim as optim
from dataloaders.lstm import DNASequenceDataset
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from cnn import CNNModel


torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

num_symbols = 5
num_epochs = 10

if torch.cuda.is_available():
    gpu_count = torch.cuda.device_count()
    print(f"Number of GPUs available: {gpu_count}")
    for i in range(gpu_count):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("No GPUs available")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

data_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/DNACorpus'

print(f"Loading data from {data_path}")

dataloader  = DNASequenceDataset(data_path, sequence_length=3, num_symbols=5, sampling_ratio=1/8) # 600M is large enough, use 1/8 of the data
dataloader = torch.utils.data.DataLoader(dataloader, batch_size=131072, shuffle=True, pin_memory=True, num_workers=96, prefetch_factor=16)  # num of batches

model = CNNModel(num_symbols).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1.0)
scaler = GradScaler()  # Mixed precision scaler

for epoch in tqdm(range(num_epochs), desc='Epochs'):
    total_loss = 0
    for inputs, targets in tqdm(dataloader, desc='Batches', leave=False):
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)

        optimizer.zero_grad()

        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
    average_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch + 1}, Loss: {average_loss:.4f}")


# save model
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': epoch,
    'loss': average_loss,
}, 'cnn_model.pth')
