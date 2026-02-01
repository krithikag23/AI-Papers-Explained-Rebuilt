import numpy as np
from attention import scaled_dot_product_attention


def main():
    seq_len = 6
    d_model = 8

    PE = sinusoidal_positional_encoding(seq_len, d_model)

    print("Positional Encoding Matrix (rounded):")
    print(np.round(PE, 3))

    print("\nShape:", PE.shape)
