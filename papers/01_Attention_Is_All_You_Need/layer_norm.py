"""
Layer Normalization (used in Transformer blocks).

This implementation normalizes across feature dimensions
and includes learnable scale (gamma) and shift (beta).
"""

import numpy as np

class LayerNorm:
    def __init__(self, d_model: int, eps: float = 1e-6):
        self.gamma = np.ones((d_model,), dtype=np.float32)
        self.beta = np.zeros((d_model,), dtype=np.float32)
        self.eps = eps

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Args:
            X: shape (seq_len, d_model)

        Returns:
            normalized output, same shape as X
        """
        mean = X.mean(axis=-1, keepdims=True)
        variance = X.var(axis=-1, keepdims=True)

        X_norm = (X - mean) / np.sqrt(variance + self.eps)
        return self.gamma * X_norm + self.beta
