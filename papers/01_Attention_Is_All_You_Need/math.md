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

They are obtained by simple linear projections:
> Q = XWᵩ 
> K = XWₖ 
> V = XWᵥ  
(Where `X` is the token embedding matrix, and `W` are learned weight matrices.)

---
## Step 2 — Finding Relevance Using Similarity
To see how much one word should attend to another, we compare:
> **Query of token i vs Key of token j**

The similarity is computed using a dot product:
> Q · Kᵀ


This gives a score — higher score → stronger relationship.

Example intuition:
- In *“The cat sat on the mat”* 
  The word **“cat”** may attend more to **“sat”** than to **“the.”**

---

## Step 3 — Scaling to Stabilize Training
Dot products grow with vector dimension.
To avoid very large values, scores are scaled by:

> √dₖ  (dimension of key vectors)

So the formula becomes:

> `Attention Score = (Q · Kᵀ) / √dₖ`

This prevents extremely sharp gradients and improves learning stability.

---
## Step 4 — Softmax → Convert Scores to Probabilities
The scores are converted into weights using **softmax**, so they:
- are positive
- sum to 1
- form a probability distribution of importance
  
Tokens with higher relevance receive higher attention weight.
---

## Step 5 — Weighted Sum of Values
Finally, attention output is computed as:
> **Weighted sum of V (value vectors)**

Meaning:
- The model aggregates information from important words
- Irrelevant tokens contribute very little
  
This creates a **context-aware representation** of each token.
---

## Why Multi-Head Attention?
Instead of learning one attention pattern, the model learns many:
- One head may focus on grammar
- Another on long-range meaning
- Another on positional relationships

Each head captures a different perspective, improving expressiveness.
---

## Key Intuition (Plain English)
Self-attention allows the model to:
- Look at **all words at once**
- Decide **which words matter the most**
- Combine information in a **context-aware way**
  
No recurrence. No convolution.

Just **parallel relationships + learned importance weights**.
---

## What This Math Enables
- Better long-range dependency handling  
- Massive training parallelism  
