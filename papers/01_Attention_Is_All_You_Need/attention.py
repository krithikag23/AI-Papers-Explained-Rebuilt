"""
Scaled Dot-Product Attention (from 'Attention Is All You Need').

This implementation focuses on clarity and correctness.
It is meant for learning + reproducibility, not maximum speed.
"""

import numpy as np


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Stable softmax for numerical safety.
    """
    x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def scaled_dot_product_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute Scaled Dot-Product Attention.

    Args:
        Q: Queries, shape (seq_len_q, d_k)
        K: Keys,    shape (seq_len_k, d_k)
        V: Values,  shape (seq_len_k, d_v)
        mask: Optional mask matrix broadcastable to (seq_len_q, seq_len_k)
              mask values should be:
                - 0 for allowed positions
                - 1 for blocked positions

    Returns:
        output: Attention output, shape (seq_len_q, d_v)
        weights: Attention weights, shape (seq_len_q, seq_len_k)
    """
    d_k = Q.shape[-1]

    # Step 1: similarity scores
    scores = (Q @ K.T) / np.sqrt(d_k)  # shape: (seq_len_q, seq_len_k)

    # Step 2: apply mask (if provided)
    if mask is not None:
        scores = scores - 1e9 * mask  # large negative => softmax ~ 0

    # Step 3: normalize into attention weights
    weights = softmax(scores, axis=-1)

    # Step 4: weighted sum of values
    output = weights @ V  # shape: (seq_len_q, d_v)

    return output, weights
