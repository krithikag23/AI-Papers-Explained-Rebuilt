# Attention Is All You Need — Mathematical Intuition

## Goal of Self-Attention

In a sentence, different words depend on each other in different ways.
Self-attention answers the question:

> “For a given word, which other words in the sequence are most relevant to it?”

Instead of processing tokens sequentially (like RNNs), the Transformer computes
relationships **between all tokens at once** using vector projections and similarity scores.

---
