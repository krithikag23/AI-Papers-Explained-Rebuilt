"""
Mini Transformer Encoder Stack.

Stacks multiple EncoderLayer blocks together.
"""

import numpy as np
from encoder_layer import EncoderLayer

class TransformerEncoder:
    def __init__(self, num_layers: int, d_model: int, num_heads: int, d_ff: int):
        self.layers = [
            EncoderLayer(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ]

    def forward(self, X: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        """
        Args:
            X: shape (seq_len, d_model)

        Returns:
            output: shape (seq_len, d_model)
        """
        for layer in self.layers:
            X = layer.forward(X, mask)
        return X
