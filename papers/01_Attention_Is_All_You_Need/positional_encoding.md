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
