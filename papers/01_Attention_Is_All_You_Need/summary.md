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
Can we model sequence dependencies without recurrence or convolution?


---


## Key Idea
The Transformer architecture uses **self-attention** to directly model relationships between all tokens in a sequence, regardless of their distance.

Instead of processing tokens one by one, the model:
- Looks at the entire sequence at once
- Assigns attention weights to determine which tokens are most relevant
- Processes sequences in parallel, improving efficiency

---

## Transformer Architecture (High-Level)
The model follows an **Encoder–Decoder** structure:
### Encoder
- Stack of identical layers
- Each layer contains:
  - Multi-head self-attention
  - Position-wise feed-forward network
  - Residual connections and layer normalization


### Decoder 
- Similar stacked structure
- Includes:
  - Masked self-attention
  - Encoder–decoder attention
  - Feed-forward layers

---
## Why Attention Matters
