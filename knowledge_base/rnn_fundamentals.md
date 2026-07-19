---
id: rnn_fundamentals
title: "Recurrent Neural Networks (RNNs)"
topic_group: "advanced"
prerequisites: ["perceptrons_mlp", "backpropagation", "activation_functions"]
difficulty: intermediate
chunk_version: 1
tags: ["rnn", "sequence-models", "recurrent", "hidden-state", "bptt", "time-series"]
---

# Recurrent Neural Networks (RNNs)

## Overview

A standard MLP assumes every input is independent and fixed-size — it has no concept of
"order" or "what came before." **Recurrent Neural Networks (RNNs)** are designed for
**sequential data** — text, time series, audio — where the order of inputs matters and
sequence length can vary.

The core idea: an RNN maintains a **hidden state** that acts as a memory, updated at every
time step as it reads through the sequence, carrying forward information from all
previous time steps.

---

## Chunk: hidden-state-recurrence

**ID**: `rnn_fundamentals#hidden-state-recurrence`

At each time step t, an RNN takes the current input x_t and the previous hidden state
h_(t-1), and produces a new hidden state h_t:

```
h_t = tanh(W_hh · h_(t-1) + W_xh · x_t + b_h)
y_t = W_hy · h_t + b_y            (optional output at this time step)
```

**Key insight — parameter sharing across time:** The same weight matrices
(W_hh, W_xh, W_hy) are reused at every single time step, regardless of sequence length.
This is conceptually similar to how a CNN filter reuses the same weights across spatial
positions (see `cnn_fundamentals#parameter-sharing`) — here, the sharing happens across
*time* instead of *space*. This is what lets an RNN handle sequences of any length with a
fixed number of parameters.

**The hidden state as memory:** h_t is a compressed summary of everything the network has
seen up through time step t. It is not raw storage of previous inputs — it's a learned,
fixed-size representation that the network must decide what to keep and what to discard
at every step.

---

## Chunk: unrolling-the-network

**ID**: `rnn_fundamentals#unrolling-the-network`

Although an RNN is often drawn as a single cell with a loop (recurrence) back into itself,
for training purposes it is conceptually "unrolled" across time — one copy of the cell per
time step, connected in sequence:

```
x_1 → [RNN cell] → h_1 → [RNN cell] → h_2 → [RNN cell] → h_3 → ...
          ↑                  ↑                  ↑
        x_1                x_2                x_3
```

Each unrolled copy shares the exact same weights (per `rnn_fundamentals#hidden-state-recurrence`).
Unrolling turns the recurrent structure into an ordinary, if very deep, computational
graph — which is what makes standard backpropagation applicable to it, described next.

---

## Chunk: backpropagation-through-time

**ID**: `rnn_fundamentals#backpropagation-through-time`

**Backpropagation Through Time (BPTT)** is standard backpropagation (see
`backpropagation#chain-rule`) applied to the unrolled RNN graph.

The loss at each time step depends on the hidden state, which in turn depends on every
previous hidden state through the recurrence. So computing ∂L/∂W_hh requires summing
gradient contributions from every time step it was used in:

```
∂L/∂W_hh = Σ_t ∂L_t/∂W_hh
```

And critically, because h_t depends on h_(t-1), which depends on h_(t-2), and so on, the
chain rule must be applied repeatedly through every intermediate hidden state back to the
start of the sequence — exactly the kind of repeated chain-rule application described in
`backpropagation#chunk-chain-rule`, except now the "depth" being backpropagated through is
the sequence length, not the number of layers.

**Truncated BPTT:** For very long sequences, backpropagating all the way to time step 1 is
computationally expensive and memory-intensive. In practice, gradients are often only
propagated back a fixed number of steps (a "truncation window"), trading exact gradients
for tractability.

---

## Chunk: vanishing-exploding-in-rnns

**ID**: `rnn_fundamentals#vanishing-exploding-in-rnns`

RNNs are especially prone to the vanishing/exploding gradient problem described generally
in `backpropagation#vanishing-gradients` — but here, the "depth" isn't layers, it's
**sequence length**.

Because the same weight matrix W_hh is multiplied repeatedly at every time step during
BPTT, the gradient flowing back to early time steps contains a product of T copies of W_hh
(and the derivative of tanh at each step):

```
∂h_t/∂h_1 ≈ ∏_(k=2)^t  W_hhᵀ · diag(tanh'(z_k))
```

- If the dominant eigenvalue of W_hh is < 1, this product shrinks toward zero as T grows —
  **vanishing gradients**. The network effectively "forgets" information from early time
  steps; it cannot learn long-range dependencies (e.g., connecting a subject at the start
  of a long sentence to a verb 30 words later).
- If the dominant eigenvalue is > 1, this product grows unboundedly — **exploding
  gradients**, often visible as loss suddenly spiking to NaN during training.

**Why this is worse than in a standard deep feedforward net:** In a deep MLP, different
layers have different weight matrices, so the compounding effect can vary layer to layer.
In an RNN, it's the *exact same* matrix multiplied at every step — a small deviation from
eigenvalue 1 compounds identically and predictably across the entire sequence length,
making both vanishing and exploding gradients more severe and more common in RNNs.

**Practical mitigation:** Gradient clipping (rescaling the gradient if its norm exceeds a
threshold) is the standard fix for exploding gradients. Vanishing gradients in RNNs are
harder to patch after the fact — the deeper architectural fix is covered in `lstm_gru.md`.

---

## Chunk: limitations-of-vanilla-rnns

**ID**: `rnn_fundamentals#limitations-of-vanilla-rnns`

Beyond vanishing/exploding gradients, plain (vanilla) RNNs have structural limitations:

- **Sequential computation:** Each time step depends on the previous one's output, so
  RNNs cannot be parallelized across the time dimension during training — every step must
  wait for the one before it to finish. This makes RNNs slow on modern hardware compared
  to architectures that can process a whole sequence at once (see `transformers.md`).
- **Limited effective memory:** Even without severe vanishing gradients, a single fixed-
  size hidden state must compress the entire history seen so far — long sequences
  inevitably lose fine-grained information from early time steps.

These limitations are the direct motivation for the gated architectures (LSTM, GRU)
covered next, and eventually for attention-based architectures that abandon recurrence
entirely.

---

## Socratic Question Seeds

1. "If the same weight matrix W_hh gets multiplied into the hidden state at every single
   time step, what happens to a gradient that has to travel back through 50 time steps,
   compared to one that only travels back through 3?"

2. "A CNN filter reuses the same weights across spatial positions. An RNN reuses the same
   weights across time steps. What problem would you run into if an RNN used a completely
   different weight matrix at every time step instead?"

3. "The hidden state h_t has a fixed size no matter how long the input sequence is. What
   trade-off does the network have to make when compressing a 200-word sentence into that
   same fixed-size vector as a 5-word sentence?"

4. "If gradient clipping only rescales gradients that are too large, would it help with a
   vanishing gradient problem where gradients are shrinking toward zero? Why or why not?"

5. "Why can't an RNN process time step 10 before it has finished processing time step 9,
   even if you had unlimited parallel computing hardware available?"

---

## References

- Elman, J. (1990) — *Finding Structure in Time* (Cognitive Science)
- Werbos, P. (1990) — *Backpropagation Through Time: What It Does and How to Do It*
- Pascanu, Mikolov & Bengio (2013) — *On the Difficulty of Training Recurrent Neural Networks*
- Goodfellow, Bengio & Courville — *Deep Learning* (MIT Press), Chapter 10
