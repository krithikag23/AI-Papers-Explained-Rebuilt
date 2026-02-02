"""
Multi-Head Attention (from 'Attention Is All You Need') implemented in NumPy.

This file builds on scaled dot-product attention and shows how multiple heads
can attend to different subspaces of the representation.
"""

import numpy as np
from attention import scaled_dot_product_attention

def split_heads(X: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Split last dimension into (num_heads, depth).

    Args:
        X: shape (seq_len, d_model)
        num_heads: number of heads

    Returns:
        shape (num_heads, seq_len, depth)
    """
    seq_len, d_model = X.shape
    assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
    depth = d_model // num_heads

    # reshape to (seq_len, num_heads, depth) then transpose
    return X.reshape(seq_len, num_heads, depth).transpose(1, 0, 2)

