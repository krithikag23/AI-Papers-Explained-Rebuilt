"""
Demo: Passing random embeddings through a stacked Transformer encoder.

Run:
python demo_transformer_encoder.py
"""

import numpy as np
from transformer_encoder import TransformerEncoder

def main():
    np.random.seed(0)

    seq_len = 6
    d_model = 8
    num_heads = 2
    d_ff = 16
    num_layers = 2

    # Random embeddings (pretend token embeddings)
    X = np.random.randn(seq_len, d_model).astype(np.float32)

    encoder = TransformerEncoder(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
    )

    output = encoder.forward(X)

    print("Input shape:", X.shape)
    print("Output shape:", output.shape)

    print("\nSample output (rounded):")
    print(np.round(output, 3))
