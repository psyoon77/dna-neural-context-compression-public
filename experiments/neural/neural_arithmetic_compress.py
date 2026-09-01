"""Historical neural arithmetic-encoder prototype; see this directory's README."""

import argparse
import contextlib
import os
import sys
from pathlib import Path

# Reuse the pinned arithmetic-coding core without mixing project additions into
# the third-party source tree.
REFERENCE_PYTHON = (
    Path(__file__).resolve().parents[2]
    / "third_party"
    / "reference-arithmetic-coding"
    / "python"
)
sys.path.insert(0, str(REFERENCE_PYTHON))

from neural_networks.feed_forward import SimpleFeedforwardModel
from neural_networks.cnn import CNNModel
from neural_networks.lstm import LSTMModel

import torch
from tqdm import tqdm
from neural_network_frequency_table import NeuralNetworkFrequencyTable, NeuralNetworkModel
import arithmeticcoding

import random
import numpy as np

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

def compress(inp, bitout, model):
    inp.seek(0, os.SEEK_END)
    total_bytes = inp.tell()
    inp.seek(0)  # 파일 포인터를 다시 처음으로 이동
    
    freqs = NeuralNetworkFrequencyTable(model)
    enc = arithmeticcoding.ArithmeticEncoder(32, bitout)
    with tqdm(total=total_bytes, desc="Compressing", unit="B", unit_scale=True) as pbar:
        while True:
            # Read one byte
            symbol = inp.read(1)
            if len(symbol) == 0:
                break
            symbol = symbol[0]
            # Encode the symbol using the current frequencies
            enc.write(freqs, symbol)
            # Update the context and frequencies
            freqs.update_context(symbol)
            pbar.update(1)  # 1바이트 처리

    # Handle EOF
    enc.write(freqs, 256)  # EOF symbol
    enc.finish()

def decompress(bitin, out, model):
    freqs = NeuralNetworkFrequencyTable(model)
    dec = arithmeticcoding.ArithmeticDecoder(32, bitin)
    while True:
        symbol = dec.read(freqs)
        if symbol == 256:  # EOF symbol
            break
        out.write(bytes((symbol,)))
        # Update the context and frequencies
        freqs.update_context(symbol)

# NeuralNetworkFrequencyTable class as above

# Modify main to accept model parameters
def main(inputfile, outputfile, model_path):
    # Load the neural network model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    if "feed" in model_path:
        model = SimpleFeedforwardModel(5)
        model.load_state_dict(checkpoint)
    elif "cnn" in model_path:
        model = CNNModel(5)
        model.load_state_dict(checkpoint["model_state_dict"])
    elif "lstm" in model_path:
        model = LSTMModel(5)
        model.load_state_dict(checkpoint)
    else:
        raise ValueError("Invalid model type")

    model = NeuralNetworkModel(model, device=device)
    # Perform file compression
    with open(inputfile, "rb") as inp, \
            contextlib.closing(arithmeticcoding.BitOutputStream(open(outputfile, "wb"))) as bitout:
        compress(inp, bitout, model)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Compress a file using adaptive arithmetic coding with a neural network model")
    argparser.add_argument("InputFile", help="Input file to compress")
    argparser.add_argument("OutputFile", help="Output compressed file")
    argparser.add_argument("ModelPath", help="Path to the neural network model")
    args = argparser.parse_args()

    main(args.InputFile, args.OutputFile, args.ModelPath)
