# Attention Is All You Need — Summary

## Paper Overview
The paper *“Attention Is All You Need”* introduces the **Transformer architecture**, a novel neural network model that relies entirely on attention mechanisms, removing the need for recurrence (RNNs) and convolution (CNNs) in sequence modeling tasks.
This work fundamentally changed how natural language processing systems are built and laid the foundation for modern large language models.

---

## Core Problem Addressed
Before Transformers, sequence-to-sequence tasks such as machine translation primarily used RNNs or LSTMs. These models had key limitations:

- Sequential processing limited parallelism
- Difficulty capturing long-range dependencies
- High training time for long sequences

The paper addresses the question:
**Can we model sequence dependencies without recurrence or convolution?**