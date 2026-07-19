---
id: attention_mechanism
title: "The Attention Mechanism"
topic_group: "advanced"
prerequisites: ["rnn_fundamentals", "activation_functions"]
difficulty: advanced
chunk_version: 1
tags: ["attention", "self-attention", "query-key-value", "context-vector", "sequence-models"]
---

# The Attention Mechanism

## Overview

Even with gating (LSTM/GRU, see `lstm_gru.md`), recurrent architectures must compress an
entire sequence's history into a single fixed-size hidden state — a bottleneck that gets
worse as sequences grow longer. **Attention** solves this by letting the network, at every
step, directly look back at *all* previous positions in the sequence and decide, freshly
each time, which ones matter most for the current computation — rather than relying on a
single compressed summary carried forward step by step.

Attention was originally introduced to improve sequence-to-sequence models (like machine
translation), and later became the sole mechanism (no recurrence at all) underlying
Transformers, covered in `transformers.md`.

---

## Chunk: the-bottleneck-problem

**ID**: `attention_mechanism#the-bottleneck-problem`

In a standard encoder-decoder RNN (e.g., translating a sentence from one language to
another), the encoder compresses the entire input sentence into one final hidden state,
which the decoder must then use to generate the entire output sequence:

```
Input:  "The cat sat on the mat" → [encoder RNN] → single fixed-size vector
Output: single fixed-size vector → [decoder RNN] → "Le chat s'est assis sur le tapis"
```

**Why this is a bottleneck:** No matter how long or complex the input sentence is, all of
its information must be squeezed into that one fixed-size vector. For long sentences, this
mirrors the same fixed-size-memory limitation discussed in
`rnn_fundamentals#limitations-of-vanilla-rnns` — early words risk being diluted or lost
entirely by the time the encoder reaches the end of the sentence.

**Key insight:** Attention removes this bottleneck by giving the decoder access to *every*
encoder hidden state, not just the final one — and lets it learn which ones to focus on
for each word it generates.

---

## Chunk: query-key-value

**ID**: `attention_mechanism#query-key-value`

Modern attention (as used in Transformers) is formalized using three learned projections
of the input: **Query (Q)**, **Key (K)**, and **Value (V)**.

An intuitive analogy: imagine a search engine. Your search term is the **query**. Every
document in the database has a **key** (something to match against). Once you find which
documents match well, you retrieve their **values** (the actual content).

```
Q = X · W_Q      (what am I looking for?)
K = X · W_K      (what does each position offer, for matching?)
V = X · W_V      (what does each position actually contain?)
```

**Attention score computation:**
```
Attention(Q, K, V) = softmax(Q·Kᵀ / √d_k) · V
```

Step by step:
1. `Q·Kᵀ` computes a similarity score between every query and every key — how relevant
   is each position to the current one being processed.
2. Dividing by `√d_k` (the square root of the key dimension) rescales these scores to
   prevent them from growing too large in magnitude, which would push the softmax into a
   saturated, near-one-hot region — connecting back to
   `activation_functions#softmax` and the saturation issue described in
   `activation_functions#sigmoid`.
3. `softmax(...)` (see `activation_functions#softmax`) converts these raw scores into a
   probability distribution — attention **weights** — that sum to 1 across all positions.
4. Multiplying by `V` produces a weighted sum of values, where positions with higher
   attention weight contribute more to the output.

**Key insight:** Every output position is a learned weighted combination of *all* input
positions' values, with the weights themselves computed dynamically based on how relevant
each position is — determined fresh for every query, not fixed in advance.

---

## Chunk: self-attention

**ID**: `attention_mechanism#self-attention`

In **self-attention**, the queries, keys, and values all come from the *same* sequence —
each position attends to every other position (including itself) within one input, rather
than one sequence attending to a separate sequence (as in the original encoder-decoder
attention setup).

This lets a model directly capture relationships between any two positions in a sequence,
regardless of how far apart they are — a sentence's first word can directly attend to its
last word in a single computation, with no dependency on information having survived a
long chain of intermediate recurrent steps.

**Contrast with RNN sequential dependency:** As noted in
`rnn_fundamentals#limitations-of-vanilla-rnns`, an RNN must pass information through every
intermediate time step to connect two distant positions — subject to the vanishing
gradient risk from `rnn_fundamentals#vanishing-exploding-in-rnns` at every step along the
way. Self-attention instead computes a **direct** connection between any two positions in
one step, with no intermediate compounding.

---

## Chunk: multi-head-attention

**ID**: `attention_mechanism#multi-head-attention`

Rather than computing a single attention distribution, **multi-head attention** runs
several independent attention computations ("heads") in parallel, each with its own
learned Q, K, V projection matrices:

```
head_i = Attention(Q·W_Q_i, K·W_K_i, V·W_V_i)
MultiHead(Q,K,V) = Concat(head_1, ..., head_h) · W_O
```

**Why multiple heads:** A single attention head must learn one particular notion of
"relevance" between positions. Different heads can specialize in capturing different types
of relationships within the same sequence — for example, one head might learn to track
grammatical subject-verb relationships while another tracks broader topical similarity.
Concatenating the heads and projecting with W_O combines these different learned
perspectives into a single output representation.

---

## Socratic Question Seeds

1. "An encoder-decoder RNN compresses an entire input sentence into one fixed-size vector
   before the decoder ever sees it. What happens to information about the very first word
   of a long sentence by the time that compression reaches the last word?"

2. "In the attention formula softmax(Q·Kᵀ/√d_k)·V, what would happen to the attention
   weights if you skipped the division by √d_k and the raw scores from Q·Kᵀ were very
   large in magnitude, given how softmax behaves on large inputs?"

3. "Self-attention lets any two positions in a sequence connect directly, in one
   computation. An RNN connects two distant positions only by passing information through
   every position in between. How does this difference relate to the vanishing gradient
   problem covered earlier for RNNs?"

4. "If a single attention head can only learn one notion of 'what's relevant,' what
   might a model lose by using just one head instead of several running in parallel?"

5. "The Value (V) vectors carry the actual content being retrieved, while Query (Q) and
   Key (K) only determine how much of each Value to retrieve. What would happen to the
   attention mechanism if Q and K were somehow identical for every position?"

---

## References

- Bahdanau, D., Cho, K. & Bengio, Y. (2014) — *Neural Machine Translation by Jointly
  Learning to Align and Translate* (original attention mechanism)
- Vaswani, A. et al. (2017) — *Attention Is All You Need* (self-attention, multi-head
  attention, Transformer architecture)
- Luong, M-T. et al. (2015) — *Effective Approaches to Attention-based Neural Machine
  Translation*
- Alammar, J. — *The Illustrated Transformer* (jalammar.github.io)
