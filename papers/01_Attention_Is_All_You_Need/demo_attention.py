"""
Tiny demo for Scaled Dot-Product Attention.

Run:
python demo_attention.py
"""

import numpy as np
from attention import scaled_dot_product_attention


def main():
    np.random.seed(42)

    seq_len = 4
    d_k = 3
    d_v = 3

    Q = np.random.randn(seq_len, d_k)
    K = np.random.randn(seq_len, d_k)
    V = np.random.randn(seq_len, d_v)

    output, weights = scaled_dot_product_attention(Q, K, V)

    print("Attention Weights (rows sum to 1):")
    print(np.round(weights, 3))
    print("\nRow sums:", np.round(weights.sum(axis=-1), 3))

    print("\nAttention Output:")
    print(np.round(output, 3))

if __name__ == "__main__":
    main()
