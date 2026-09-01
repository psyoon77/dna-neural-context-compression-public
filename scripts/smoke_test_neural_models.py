#!/usr/bin/env python3
"""Exercise each archived neural model with a small CPU input."""

from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
NEURAL_EXPERIMENT = ROOT / "experiments" / "neural"
sys.path.insert(0, str(NEURAL_EXPERIMENT))

from neural_networks.cnn import CNNModel  # noqa: E402
from neural_networks.feed_forward import SimpleFeedforwardModel  # noqa: E402
from neural_networks.lstm import LSTMModel  # noqa: E402


def main() -> None:
    torch.manual_seed(0)
    batch_size = 2
    sequence_length = 8
    symbol_count = 5
    symbol_ids = torch.arange(sequence_length).repeat(batch_size, 1) % symbol_count
    model_input = torch.nn.functional.one_hot(
        symbol_ids, num_classes=symbol_count
    ).float()

    checks = (
        (
            "FFNN",
            SimpleFeedforwardModel(symbol_count),
            (batch_size, sequence_length, symbol_count),
        ),
        ("CNN", CNNModel(symbol_count), (batch_size, symbol_count)),
        ("LSTM", LSTMModel(symbol_count), (batch_size, symbol_count)),
    )

    for name, model, expected_shape in checks:
        model.eval()
        with torch.inference_mode():
            output = model(model_input)
        if tuple(output.shape) != expected_shape:
            raise RuntimeError(
                f"{name}: expected output shape {expected_shape}, got {tuple(output.shape)}"
            )
        if not torch.isfinite(output).all():
            raise RuntimeError(f"{name}: output contains a non-finite value")
        print(f"{name}: OK {tuple(output.shape)}")

    print("Neural model smoke test: OK")


if __name__ == "__main__":
    main()
