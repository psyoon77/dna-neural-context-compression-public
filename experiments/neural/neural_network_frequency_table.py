"""Experimental bridge between five-symbol neural models and arithmetic coding."""

from arithmeticcoding import FrequencyTable
import torch
import torch.nn as nn
from neural_networks.feed_forward import SimpleFeedforwardModel
from neural_networks.cnn import CNNModel
from neural_networks.lstm import LSTMModel

from neural_networks.dna_sequence_dataset import char_to_symbol

def symbol_to_idx(symbol):
    if symbol == 256:
        return 4
    return char_to_symbol(chr(symbol))

class NeuralNetworkModel:
    # The model should be deterministic: disable dropout and set random seeds
    def __init__(self, model, num_symbols=5, device=None):
        self.num_symbols = num_symbols
        self.model = model
        self.model.eval()
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)

    def predict(self, context):
        # Convert context to tensor
        if isinstance(self.model, SimpleFeedforwardModel):
            context_tensor = torch.zeros((1, len(context), self.num_symbols), device=self.device)
            for i, symbol in enumerate(context):
                symbol_idx = symbol_to_idx(symbol)
                context_tensor[0, i, symbol_idx] = 1.0
            with torch.no_grad():
                output = self.model(context_tensor)
                probabilities = torch.softmax(output[0, -1], dim=0).cpu().numpy()
            return probabilities
        elif isinstance(self.model, CNNModel):
            # Prepare input for CNN
            if len(context) <= 2:
                return [1.0 / self.num_symbols] * self.num_symbols
            context_tensor = torch.zeros((1, len(context), self.num_symbols), device=self.device)
            for i, symbol in enumerate(context):
                symbol_idx = symbol_to_idx(symbol)
                context_tensor[0, i, symbol_idx] = 1.0
            with torch.no_grad():
                output = self.model(context_tensor)
                probabilities = torch.softmax(output[0], dim=0).cpu().numpy()
            return probabilities

        elif isinstance(self.model, LSTMModel):
            # Prepare input for LSTM
            if len(context) <= 2:
                return [1.0 / self.num_symbols] * self.num_symbols
            context_tensor = torch.zeros((1, len(context), self.num_symbols), device=self.device)
            for i, symbol in enumerate(context):
                symbol_idx = symbol_to_idx(symbol)
                context_tensor[0, i, symbol_idx] = 1.0
            with torch.no_grad():
                output = self.model(context_tensor)
                probabilities = torch.softmax(output[0], dim=0).cpu().numpy()
            return probabilities


class NeuralNetworkFrequencyTable(FrequencyTable):
    def __init__(self, model):
        self.model = model  # The neural network model
        self.symbol_limit = 257  # Number of symbols (256 bytes + EOF)
        self.probabilities = [1.0 / self.symbol_limit] * self.symbol_limit  # Initial probabilities
        self.total_freq = 0  # Will be   set after scaling
        self._update_frequencies()
        self.context = []

    def update_context(self, symbol):
        # Update the context with the recently encoded/decoded symbol
        self.context.append(symbol)
        # Optionally limit the context size
        N = 100  # For example, use the last 100 symbols
        self.context = self.context[-N:]
        # Update probabilities based on the new context
        self._update_probabilities()

    def _update_probabilities(self):
        # Use the neural network to predict the next symbol probabilities
        self.probabilities = self.model.predict(self.context)
        # Convert probabilities to integer frequencies
        self._update_frequencies()

    def _update_frequencies(self):
        # Get maximum total frequency from the encoder if necessary
        max_total = 1 << 15  # Example maximum total frequency
        # Scale probabilities to frequencies without exceeding max_total
        scale = max_total - self.symbol_limit  # Leave room to ensure each freq >= 1
        freqs = [max(int(p * scale), 1) for p in self.probabilities]
        self.frequencies = freqs
        self.total_freq = sum(self.frequencies)
        # Re-normalize if total exceeds max_total
        if self.total_freq > max_total:
            factor = max_total / self.total_freq
            self.frequencies = [max(int(freq * factor), 1) for freq in self.frequencies]
            self.total_freq = sum(self.frequencies)
        # Recompute cumulative frequencies
        self.cumulative = [0]
        for freq in self.frequencies:
            self.cumulative.append(self.cumulative[-1] + freq)


    # Implement the FrequencyTable interface methods
    def get_symbol_limit(self):
        return self.symbol_limit

    def get(self, symbol):
        symbol = symbol_to_idx(symbol)
        return self.frequencies[symbol]

    def get_total(self):
        return self.total_freq

    def get_low(self, symbol):
        symbol = symbol_to_idx(symbol)
        return self.cumulative[symbol]

    def get_high(self, symbol):
        symbol = symbol_to_idx(symbol)
        return self.cumulative[symbol + 1]

    # No need to implement set() and increment(), since frequencies are determined by the neural network
    def set(self, symbol, freq):
        raise NotImplementedError("set() not supported in NeuralNetworkFrequencyTable")

    def increment(self, symbol):
        raise NotImplementedError("increment() not supported in NeuralNetworkFrequencyTable")


# class NeuralNetworkFrequencyTable(FrequencyTable):
#     def __init__(self, model, sequence_length):
#         self.model = model
#         self.symbol_limit = num_symbols
#         self.sequence_length = sequence_length
#         self.context = [0] * sequence_length  # Initialize with zeros or any default symbol
#         # ... rest of the initialization

#     def update_context(self, symbol):
#         self.context.append(symbol)
#         self.context = self.context[-self.sequence_length:]  # Keep only the last N symbols
#         self._update_probabilities()

#     def _update_probabilities(self):
#         # Convert the context to a tensor
#         input_sequence = torch.tensor([self.context], dtype=torch.long, device=self.device)
#         with torch.no_grad():
#             logits = self.model(input_sequence)
#             probabilities = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
#         self.probabilities = probabilities
#         self._update_frequencies()
