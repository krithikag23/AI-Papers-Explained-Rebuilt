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
