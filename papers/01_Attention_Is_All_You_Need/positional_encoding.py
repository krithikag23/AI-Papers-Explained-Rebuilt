"""
Sinusoidal Positional Encoding (from 'Attention Is All You Need').

This file implements fixed positional encodings using sine/cosine waves,
so Transformers can capture token order without recurrence.
"""

import numpy as np

def sinusoidal_positional_encoding(seq_len: int, d_model: int) -> np.ndarray:
    """
    Create sinusoidal positional encoding matrix.

    Args:
        seq_len: number of positions (sequence length)
        d_model: embedding dimension

    Returns:
        PE: shape (seq_len, d_model)
    """
    positions = np.arange(seq_len)[:, None]          # (seq_len, 1)
    dims = np.arange(d_model)[None, :]               # (1, d_model)

    # Compute the angle rates (10000^(2i/d_model))
    angle_rates = 1 / np.power(10000, (2 * (dims // 2)) / np.float32(d_model))
    angles = positions * angle_rates                 # (seq_len, d_model)

    PE = np.zeros((seq_len, d_model), dtype=np.float32)

    # Apply sin to even indices (0,2,4,...)
    PE[:, 0::2] = np.sin(angles[:, 0::2])

    # Apply cos to odd indices (1,3,5,...)
    PE[:, 1::2] = np.cos(angles[:, 1::2])

    return PE
