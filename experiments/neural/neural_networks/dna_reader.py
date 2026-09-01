import os

def read_dna_files(directory):
    sequence = ''
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath):
            with open(filepath, 'r') as f:
                data = f.read().strip()
                sequence += data
    return sequence

# Usage
sequence = read_dna_files('DNACorpus')
