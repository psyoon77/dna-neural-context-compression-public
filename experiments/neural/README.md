# Neural experiment notes

This directory preserves the project-specific neural context-model experiments.
It is research code, not a released codec or Python package.

## Contents

- `kmer_onehot.py` converts DNA windows into categorical one-hot rows.
- `preprocess.py` contains local-corpus preprocessing and dataset utilities.
- `neural_networks/` contains the FFNN, CNN, and LSTM models and training code.
- `neural_network_frequency_table.py` adapts model probabilities to the pinned
  arithmetic-coding interface.
- `neural_arithmetic_compress.py` is the historical neural encoder prototype.

## Important limitation

The historical neural encoder mixes a five-symbol DNA prediction model with a
byte-oriented arithmetic-coding interface. A matching, tested decompression
path was not completed. Consequently, this code is retained to document the
experiment but is not presented as a verified lossless codec.

The dependency-free fixed, adaptive, and PPM reference codecs under
`third_party/` have automated byte-for-byte round-trip coverage. That test does
not validate the neural integration.

## Data and checkpoints

The DNA corpus, trained checkpoints, and generated compressed files are not
included. They are local inputs or generated artifacts and are ignored by the
repository's `.gitignore`.

Training scripts default to an available CUDA device and otherwise use CPU.
They still require experiment-specific review of dataset paths, sampling,
memory use, and hyperparameters before execution.
