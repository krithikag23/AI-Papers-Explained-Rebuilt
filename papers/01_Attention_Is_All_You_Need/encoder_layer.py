"""
Transformer Encoder Layer (Attention + FFN + Residual + LayerNorm).
"""

import numpy as np
from multi_head_attention import MultiHeadAttention
from layer_norm import LayerNorm
from feed_forward import FeedForward

class EncoderLayer:
    def __init__(self, d_model: int, num_heads: int, d_ff: int):
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)

    def forward(self, X: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        """
        Args:
            X: shape (seq_len, d_model)

        Returns:
            output: shape (seq_len, d_model)
        """
        # Multi-head self-attention + residual
        attn_output, _ = self.mha.forward(X, mask)
        X = self.norm1(X + attn_output)

        # Feed-forward network + residual
        ffn_output = self.ffn.forward(X)
        X = self.norm2(X + ffn_output)

        return X
