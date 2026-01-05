# Attention Is All You Need — Mathematical Intuition

## Goal of Self-Attention

In a sentence, different words depend on each other in different ways.
Self-attention answers the question:

> “For a given word, which other words in the sequence are most relevant to it?”

Instead of processing tokens sequentially (like RNNs), the Transformer computes
relationships **between all tokens at once** using vector projections and similarity scores.

---
## Step 1 — Representing Tokens as Vectors

Each token in the sentence is first converted into an embedding vector.
From each embedding, the model computes three different vectors:
- **Q — Query** → What information this token is looking for 
- **K — Key** → What information this token contains  
- **V — Value** → The actual content to pass forward
-   
They are obtained by simple linear projections:
