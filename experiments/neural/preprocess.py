import os
import sys
import glob
import subprocess
from concurrent.futures import ThreadPoolExecutor

def process_data():
    """
    Processes the DNA corpus using kmer_onehot.py.
    """
    # Set the path of kmer_onehot.py relative to the current script
    kmer_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'kmer_onehot.py')

    if not os.path.exists(kmer_script):
        print(f"Error: cannot find kmer_onehot.py at {kmer_script}")
        return

    # Find all files in DNACorpus directory with 4-character names
    files = glob.glob('DNACorpus/????')
    if not files:
        print("Error: no matching files found in DNACorpus directory")
        return

    print(f"Found {len(files)} files to process")

    # Data processing function
    def process_file(file):
        try:
            output_file = f"{file}.oneshot.csv"
            print(f"Processing file: {file}")
            with open(file, "r", encoding="ascii") as source, open(
                output_file, "w", encoding="ascii"
            ) as destination:
                subprocess.run(
                    [sys.executable, kmer_script, "10"],
                    stdin=source,
                    stdout=destination,
                    check=True,
                )
            print(f"Processed: {file}")
        except Exception as e:
            print(f"Error processing {file}: {str(e)}")

    # Begin processing
    print("Starting data processing...")
    with ThreadPoolExecutor() as executor:
        executor.map(process_file, files)

    output_files = glob.glob('DNACorpus/*.oneshot.csv')
    print(f"Processed {len(output_files)} files")
    print("Data processing completed.")

if __name__ == "__main__":
    # Supply a local DNACorpus directory, then enable this step if preprocessing
    # is required for a new experiment.
    # process_data()

    # Part 2: PyTorch Data Loader
    import torch
    from torch.utils.data import Dataset, DataLoader
    import pandas as pd
    import numpy as np
    from typing import Tuple

    class DNASequenceDataset(Dataset):
        def __init__(self, data_dir: str, sequence_length: int = 10):
            """
            Initialize DNA sequence dataset.

            Args:
                data_dir: Path to the directory containing the .oneshot.csv files.
                sequence_length: DNA sequence length (default 10).
            """
            self.sequence_length = sequence_length

            # Get all processed CSV files
            csv_files = glob.glob(os.path.join(data_dir, '*.oneshot.csv'))
            print(f"Found {len(csv_files)} CSV files")

            # Read and merge all CSV files
            all_data = []
            for file in csv_files:
                try:
                    # Read the CSV file
                    data = pd.read_csv(file, header=None)
                    all_data.append(data)
                except Exception as e:
                    print(f"Error reading {file}: {str(e)}")

            if all_data:
                self.data = pd.concat(all_data, ignore_index=True)
                print(f"Total samples: {len(self.data)}")
            else:
                self.data = pd.DataFrame()
                print("No data loaded.")

            # DNA base mapping dictionary
            self.nucleotide_map = {
                'A': 0,
                'C': 1,
                'G': 2,
                'T': 3
            }

        def __len__(self) -> int:
            return len(self.data)

        def _encode_sequence(self, sequence: str) -> torch.Tensor:
            """
            Convert DNA sequences to one-hot encoding.

            Args:
                sequence: DNA sequence string.

            Returns:
                torch.Tensor: The one-hot encoded tensor.
            """
            # Remove commas and create a list of nucleotides
            nucleotides = sequence.replace(',', '')

            # Create a one-hot encoding matrix
            encoding = torch.zeros((4, self.sequence_length), dtype=torch.float32)

            # One-hot encoding for each position
            for i, nucleotide in enumerate(nucleotides):
                if nucleotide in self.nucleotide_map:
                    encoding[self.nucleotide_map[nucleotide], i] = 1.0

            return encoding

        def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """
            Get a sample of the data.

            Args:
                idx: Sample index.

            Returns:
                tuple: (first sequence tensor, second sequence tensor, label)
            """
            row = self.data.iloc[idx]

            # Extract two sequences and the label
            seq1 = ''.join(row[:self.sequence_length].astype(str))
            seq2 = ''.join(row[self.sequence_length:-1].astype(str))
            label = row.iloc[-1]

            # Encode sequences
            seq1_tensor = self._encode_sequence(seq1)
            seq2_tensor = self._encode_sequence(seq2)

            return seq1_tensor, seq2_tensor, torch.tensor(label, dtype=torch.long)

    # Function to create data loaders
    def create_data_loaders(
            data_dir: str,
            batch_size: int = 32,
            val_split: float = 0.2,
            num_workers: int = 2,
            sequence_length: int = 10
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Create training and validation data loaders.

        Args:
            data_dir: Path to data directory.
            batch_size: Batch size.
            val_split: Validation set ratio.
            num_workers: Number of data loading threads.
            sequence_length: DNA sequence length.

        Returns:
            tuple: (Training Data Loader, Validation Data Loader)
        """
        # Create the complete dataset
        full_dataset = DNASequenceDataset(data_dir, sequence_length)

        # Check if dataset is empty
        if len(full_dataset) == 0:
            print("No data available to create data loaders.")
            return None, None

        # Calculate split sizes
        val_size = int(len(full_dataset) * val_split)
        train_size = len(full_dataset) - val_size

        # Randomly split into training and validation sets
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size]
        )

        # Create Data Loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

        return train_loader, val_loader

    # Usage example
    data_dir = 'DNACorpus'

    train_loader, val_loader = create_data_loaders(
        data_dir=data_dir,
        batch_size=32,
        val_split=0.2,
        num_workers=2
    )

    if val_loader is not None:
        # Iterate over the validation loader and print a batch
        for batch in val_loader:
            print(batch)
            break
