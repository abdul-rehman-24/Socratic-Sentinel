---
id: lstm_gru
title: "LSTM and GRU (Gated Recurrent Architectures)"
topic_group: "advanced"
prerequisites: ["rnn_fundamentals", "activation_functions"]
difficulty: advanced
chunk_version: 1
tags: ["lstm", "gru", "gating", "long-term-dependencies", "cell-state", "sequence-models"]
---

# LSTM and GRU (Gated Recurrent Architectures)

## Overview

Vanilla RNNs, as covered in `rnn_fundamentals#vanishing-exploding-in-rnns`, struggle to
learn long-range dependencies because gradients must repeatedly pass through the same
weight matrix and a saturating activation at every time step, causing them to vanish over
long sequences.

**Long Short-Term Memory (LSTM)** networks, and their simplified cousin **Gated Recurrent
Units (GRU)**, solve this with **gating mechanisms** — learned, per-step decisions about
what information to keep, forget, or output. Instead of forcing all information through
the same repeated multiplication every step, gated architectures give the network an
explicit, more direct path for information (and gradients) to flow across many time steps
largely unchanged.

---

## Chunk: why-lstm-was-needed

**ID**: `lstm_gru#why-lstm-was-needed`

The core problem with vanilla RNNs is that the hidden state is **completely overwritten**
at every time step:

```
h_t = tanh(W_hh · h_(t-1) + W_xh · x_t + b_h)
```

There is no mechanism to say "keep most of what's in h_(t-1) unchanged, and only update a
small part of it." Every time step forces a full rewrite, and the repeated tanh
nonlinearity compounds the vanishing gradient problem described in
`rnn_fundamentals#vanishing-exploding-in-rnns`.

**Key insight:** LSTM's central innovation is separating "memory" (the **cell state**) from
"output" (the **hidden state**), and controlling both with learned gates — small neural
network layers that output values between 0 and 1, deciding how much of each piece of
information to let through.

---

## Chunk: lstm-cell-structure

**ID**: `lstm_gru#lstm-cell-structure`

An LSTM cell maintains two states at each time step: the **cell state** C_t (long-term
memory) and the **hidden state** h_t (short-term output). Three gates control how
information flows between them, each implemented as a sigmoid-activated layer (see
`activation_functions#sigmoid`) producing values in (0, 1):

**1. Forget gate** — decides what fraction of the old cell state to keep:
```
f_t = σ(W_f · [h_(t-1), x_t] + b_f)
```

**2. Input gate** — decides what new information to add to the cell state:
```
i_t = σ(W_i · [h_(t-1), x_t] + b_i)
C̃_t = tanh(W_C · [h_(t-1), x_t] + b_C)     (candidate values)
```

**3. Cell state update** — combines the forget and input gates:
```
C_t = f_t ⊙ C_(t-1) + i_t ⊙ C̃_t
```

**4. Output gate** — decides what part of the cell state becomes the visible hidden state:
```
o_t = σ(W_o · [h_(t-1), x_t] + b_o)
h_t = o_t ⊙ tanh(C_t)
```

**Why each gate outputs values in (0, 1):** A gate value near 1 means "let this
information through almost entirely"; a value near 0 means "block it almost entirely."
This is a soft, learned, per-dimension decision — not a hard on/off switch — allowing the
network to partially remember and partially forget different pieces of information
independently.

---

## Chunk: gradient-flow-through-cell-state

**ID**: `lstm_gru#gradient-flow-through-cell-state`

The cell state update equation is the key to why LSTMs mitigate vanishing gradients:

```
C_t = f_t ⊙ C_(t-1) + i_t ⊙ C̃_t
```

Notice this is an **additive** update, not a repeated matrix multiplication followed by a
squashing nonlinearity like in the vanilla RNN hidden state update. When you differentiate
C_t with respect to C_(t-1):

```
∂C_t/∂C_(t-1) = f_t
```

If the forget gate f_t is close to 1 (the network has learned "keep this information"),
the gradient passes through nearly unchanged across that time step — no compounding
shrinkage from repeated matrix multiplication or repeated tanh saturation. This creates
what is often called a **"constant error carousel"** — a near-direct gradient highway
through the cell state across many time steps, conceptually similar in spirit to the
skip/residual connections mentioned as a vanishing-gradient fix in
`backpropagation#vanishing-gradients`.

**Important nuance:** LSTMs reduce, but do not entirely eliminate, vanishing gradients.
If the forget gate learns to output values well below 1 for long stretches, gradients can
still shrink — but the network now has the *capacity* to learn to keep that gate near 1
when long-term memory matters, which a vanilla RNN structurally cannot do.

---

## Chunk: gru-simplified-gating

**ID**: `lstm_gru#gru-simplified-gating`

The **Gated Recurrent Unit (GRU)** achieves a similar effect to LSTM with a simpler
structure — no separate cell state, and only two gates instead of three:

**Update gate** — combines the role of LSTM's forget and input gates:
```
z_t = σ(W_z · [h_(t-1), x_t] + b_z)
```

**Reset gate** — controls how much past information to ignore when computing the new
candidate state:
```
r_t = σ(W_r · [h_(t-1), x_t] + b_r)
h̃_t = tanh(W_h · [r_t ⊙ h_(t-1), x_t] + b_h)
```

**Final update** (a single interpolation between old and new state):
```
h_t = (1 - z_t) ⊙ h_(t-1) + z_t ⊙ h̃_t
```

**Key insight:** Like the LSTM cell state update, this final line is an interpolation
(weighted average) between the previous state and a new candidate — again additive rather
than a full overwrite through repeated multiplication, which is what gives GRU similar
gradient-flow benefits to LSTM.

---

## Chunk: lstm-vs-gru

**ID**: `lstm_gru#lstm-vs-gru`

| Aspect | LSTM | GRU |
|---|---|---|
| Gates | 3 (forget, input, output) | 2 (update, reset) |
| Separate cell state? | Yes (C_t and h_t) | No (only h_t) |
| Parameters | More | Fewer (~25% less) |
| Training speed | Slower per step | Faster per step |
| Typical performance | Often marginally better on very long sequences | Often comparable, sometimes better on smaller datasets |

**Key insight:** There is no universally "better" choice — GRU is a reasonable default
when compute or data is limited (fewer parameters means less risk of overfitting on
smaller datasets, connecting to `overfitting_regularization#what-is-overfitting`), while
LSTM's extra capacity can help on tasks with very long-range dependencies and abundant
training data.

---

## Socratic Question Seeds

1. "The vanilla RNN hidden state gets completely overwritten by a matrix multiply and
   tanh at every time step. The LSTM cell state instead uses C_t = f_t ⊙ C_(t-1) + i_t ⊙ C̃_t.
   What's structurally different about an additive update like this compared to a full
   overwrite, in terms of what a gradient has to pass through?"

2. "If the forget gate f_t were fixed at exactly 1 for every time step, what would
   ∂C_t/∂C_(t-1) equal, and what would that mean for how far back a gradient could travel
   without shrinking?"

3. "A GRU has no separate cell state — it only maintains h_t. Given that GRU still uses an
   interpolation update, (1 - z_t) ⊙ h_(t-1) + z_t ⊙ h̃_t, why might it achieve similar
   gradient-flow benefits to LSTM despite being structurally simpler?"

4. "If a dataset is small and the sequences aren't extremely long, would the extra
   parameters in an LSTM's three gates necessarily help performance, or could they
   introduce a risk you've seen discussed elsewhere in this curriculum?"

5. "Each gate in an LSTM outputs a value between 0 and 1 for every dimension of the state,
   rather than a single global 0-or-1 decision. Why might per-dimension control matter for
   what the network is able to selectively remember or forget?"

---

## References

- Hochreiter, S. & Schmidhuber, J. (1997) — *Long Short-Term Memory* (Neural Computation)
- Cho, K. et al. (2014) — *Learning Phrase Representations using RNN Encoder-Decoder for
  Statistical Machine Translation* (GRU introduction)
- Olah, C. — *Understanding LSTM Networks* (colah.github.io, 2015)
- Chung, J. et al. (2014) — *Empirical Evaluation of Gated Recurrent Neural Networks on
  Sequence Modeling*
- Goodfellow, Bengio & Courville — *Deep Learning* (MIT Press), Chapter 10.10
