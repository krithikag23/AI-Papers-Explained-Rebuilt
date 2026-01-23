# Positional Encoding — Intuition and Purpose

## Why Positional Encoding Is Needed
The Transformer architecture processes all tokens **in parallel** using self-attention.
While this enables efficiency, it introduces a problem:
> Self-attention alone has **no sense of word order**.

For example:
- “The cat chased the dog”
- “The dog chased the cat”

Contain the same words, but very different meanings.
Without position information, the model cannot distinguish between them.

---

## Core Idea
**Positional encoding injects information about token order** into the input embeddings.
Instead of learning order through recurrence (RNNs), the Transformer:
- Adds a positional signal to each token embedding
- Allows attention to consider *where* a token appears in the sequence

---

## How Positional Encoding Is Applied
For each token:
final_input = token_embedding + positional_encoding

This ensures that:
- Token identity and position are both preserved
- Attention mechanisms can use position-aware information

---
## Sinusoidal Positional Encoding
The original paper uses **fixed sinusoidal functions** to encode position.

For position `pos` and dimension `i`:
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

---
