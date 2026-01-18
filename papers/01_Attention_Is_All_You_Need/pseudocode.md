# Attention Is All You Need — Pseudocode Explanation

This document presents a **high-level pseudocode walkthrough** of the Transformer model.
The goal is to understand *how data flows through the architecture*, not implementation details.

---
## Notation
- X : Input token embeddings
- Q, K, V : Query, Key, Value matrices
- d_k : Dimension of key vectors
- h : Number of attention heads

---
## Key Takeaway
The Transformer replaces sequential computation with:
