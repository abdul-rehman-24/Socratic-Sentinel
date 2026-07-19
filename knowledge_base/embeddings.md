
---
id: embeddings
title: "Embeddings and Representation Learning"
topic_group: "advanced"
prerequisites: ["perceptrons_mlp", "backpropagation"]
difficulty: intermediate
chunk_version: 1
tags: ["embeddings", "word-embeddings", "representation-learning", "latent-space", "word2vec"]
---
# Embeddings and Representation Learning

## Overview

An **embedding** is a learned, dense, lower-dimensional vector representation of a
discrete input — most commonly a word, but also applicable to categories, users, items, or
any other discrete entity a network needs to process numerically. Rather than representing
each discrete item with an arbitrary index or a sparse one-hot vector, an embedding places
it as a point in a continuous vector space, where the position itself is learned to be
meaningful.

Embeddings are foundational to how virtually all modern NLP and many other deep learning
systems represent discrete, categorical input data — including as the very first step
inside Transformer architectures (see `transformers#positional-encoding`).

---

## Chunk: the-one-hot-problem

**ID**: `embeddings#the-one-hot-problem`

Before embeddings, the standard way to represent a discrete category (like a word from a
vocabulary) numerically was a **one-hot vector** — a vector as long as the vocabulary
size, with a 1 in the position corresponding to that word and 0 everywhere else:

```
Vocabulary: ["cat", "dog", "car", "apple"]

"cat"   → [1, 0, 0, 0]
"dog"   → [0, 1, 0, 0]
"car"   → [0, 0, 1, 0]
```

**Two major problems with one-hot encoding:**

1. **No notion of similarity.** The dot product (and every other standard similarity
   measure) between any two distinct one-hot vectors is always exactly 0 — "cat" and "dog"
   are represented as being just as different from each other as "cat" and "car," even
   though cat and dog are semantically far more similar. One-hot encoding carries no
   information about relationships between items.
2. **Dimensionality scales with vocabulary size.** A vocabulary of 50,000 words requires
   50,000-dimensional one-hot vectors — extremely sparse (only one non-zero entry) and
   computationally wasteful, especially as the first layer of a network multiplying
   against such a vector.

**Key insight:** Embeddings solve both problems: they use a small, fixed number of
dimensions (e.g., 100–300, regardless of vocabulary size) and — critically — the
*position* of each item in that space is learned such that semantically similar items end
up close together.

---

## Chunk: the-embedding-layer

**ID**: `embeddings#the-embedding-layer`

Mechanically, an embedding is implemented as a lookup table — a matrix E of shape
(vocabulary_size × embedding_dimension), where row i is the embedding vector for the item
with index i:

```
embedding(word_index) = E[word_index]      (simply selects the corresponding row)
```

Multiplying a one-hot vector by the embedding matrix E is mathematically equivalent to
just selecting the corresponding row directly — in practice this is implemented as a fast
lookup rather than an actual matrix multiplication, but the underlying trainable object is
still just a standard weight matrix.

**How it's trained:** The embedding matrix E is a set of learnable parameters, exactly like
the weight matrices in any other layer discussed in `perceptrons_mlp.md`. Gradients flow
back to E through backpropagation (`backpropagation.md`) based on whatever downstream task
the embeddings are being used for — the embedding values are updated via the same gradient
descent update rule as every other weight in the network.

**Key insight:** There's no special "embedding-learning algorithm" distinct from ordinary
backpropagation and gradient descent — the embedding matrix is simply a weight matrix
whose specific structure (one full row of parameters per discrete input item) makes it
naturally interpretable as a lookup table producing a dense vector per item.

---

## Chunk: what-embeddings-learn-to-capture

**ID**: `embeddings#what-embeddings-learn-to-capture`

When trained on a task that requires understanding relationships between words (such as
predicting a word from its surrounding context), the resulting embedding space often
captures meaningful semantic and even relational structure — purely as a side effect of
optimizing the training objective, with no explicit instruction to do so.

**A classic empirical observation:** In well-trained word embeddings, vector arithmetic on
embeddings can approximate meaningful relationships, e.g.:

```
embedding("king") - embedding("man") + embedding("woman") ≈ embedding("queen")
```

**Why this emerges:** Words that appear in similar contexts tend to be pushed to similar
regions of the embedding space during training, since the network benefits from
representing them similarly if they play similar predictive roles. Words related by a
consistent relationship (like gender, or country-to-capital) tend to be separated by
similar vector offsets throughout the space, because that consistent contextual pattern
gets encoded consistently across many such word pairs during training.

**Important caveat:** This structure is a *learned, emergent* property of the training
data and objective — not a guaranteed or explicitly enforced property. The quality and
nature of what an embedding space captures depends entirely on what data and task it was
trained on.

---

## Chunk: pretrained-vs-task-specific-embeddings

**ID**: `embeddings#pretrained-vs-task-specific-embeddings`

Similar to the transfer learning strategies covered in `transfer_learning.md`, embeddings
can either be:

**Trained from scratch, jointly with the rest of the network:** The embedding matrix E
starts randomly initialized and is updated via backpropagation alongside every other
weight in the network, specifically for the task at hand.

**Pretrained on a separate, often much larger, dataset, then reused:** Embeddings like
Word2Vec or GloVe are trained once on massive text corpora (capturing general-purpose word
relationships), then loaded into a new model as a starting point — either frozen (never
updated, exactly like `transfer_learning#feature-extraction`) or fine-tuned further on the
new, smaller, task-specific dataset (exactly like `transfer_learning#fine-tuning`).

**Contextual embeddings (a further evolution):** Classical embeddings assign exactly one
fixed vector to each word, regardless of context — but words like "bank" (a financial
institution vs. a riverbank) have different meanings depending on surrounding words.
Transformer-based models (`transformers.md`) produce **contextual embeddings**, where a
word's representation is dynamically computed using self-attention over its surrounding
context (`attention_mechanism#self-attention`) — so the same word can have a different
embedding vector in different sentences, depending on what it actually means there.

---

## Socratic Question Seeds

1. "Two one-hot vectors for different words always have a dot product of exactly 0, no
   matter how semantically similar or different the words are. What information is
   completely absent from a one-hot representation that an embedding is specifically
   designed to capture?"
2. "An embedding matrix E is described as being trained through ordinary backpropagation
   and gradient descent, just like any other weight matrix. What would the gradient
   ∂L/∂E[word_index] actually be updating, in terms of what that specific row of the
   matrix represents?"
3. "If embedding("king") - embedding("man") + embedding("woman") ends up close to
   embedding("queen"), and this relationship was never explicitly programmed into the
   training process, what does that suggest about how the training objective indirectly
   shaped the geometry of the embedding space?"
4. "A classical (non-contextual) embedding assigns the word 'bank' exactly one fixed
   vector, regardless of whether the sentence is about a river or about money. What
   architectural feature of a Transformer allows it to instead produce a different vector
   for 'bank' depending on its surrounding context?"
5. "Pretrained word embeddings, once loaded into a new model, can either be kept frozen or
   fine-tuned further. Based on what you know about the trade-offs between feature
   extraction and fine-tuning in transfer learning, when might freezing them be the safer
   choice?"

---

## References

- Mikolov, T. et al. (2013) — *Efficient Estimation of Word Representations in Vector
  Space* (Word2Vec)
- Pennington, J., Socher, R. & Manning, C. (2014) — *GloVe: Global Vectors for Word
  Representation*
- Peters, M. et al. (2018) — *Deep Contextualized Word Representations* (ELMo)
- Goodfellow, Bengio & Courville — *Deep Learning* (MIT Press), Chapter 15.4
