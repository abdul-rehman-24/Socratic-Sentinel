---
id: transformers
title: "Transformers"
topic_group: "advanced"
prerequisites: ["attention_mechanism", "backpropagation", "activation_functions"]
difficulty: advanced
chunk_version: 1
tags: ["transformers", "self-attention", "positional-encoding", "encoder-decoder", "sequence-models"]
---

# Transformers

## Overview

The **Transformer** architecture, introduced in "Attention Is All You Need" (2017),
abandons recurrence entirely — no hidden state passed step by step, no RNN cell at all.
Instead, it relies solely on the self-attention mechanism covered in
`attention_mechanism.md`, combined with feedforward layers, to process an entire sequence
in parallel.

This single architectural shift addressed the two core limitations of RNNs discussed in
`rnn_fundamentals#limitations-of-vanilla-rnns`: sequential (non-parallelizable)
computation, and difficulty capturing long-range dependencies. Transformers underlie the
majority of modern large-scale language and multimodal models.

---

## Chunk: why-remove-recurrence

**ID**: `transformers#why-remove-recurrence`

As established in `rnn_fundamentals#limitations-of-vanilla-rnns`, RNNs must process a
sequence one time step at a time — step t cannot begin until step t-1 finishes, which
prevents parallelization across the time dimension no matter how much compute is
available.

Self-attention (`attention_mechanism#self-attention`) computes relationships between all
positions in a sequence simultaneously, using matrix multiplications that are highly
parallelizable on modern hardware (GPUs/TPUs). By building an architecture entirely out of
attention and feedforward layers — with no recurrence — Transformers can process an
entire sequence at once during training, dramatically speeding up training on large
datasets.

**Trade-off:** Removing recurrence also removes any inherent notion of sequence order —
self-attention treats a sequence as an unordered set of positions unless order is
explicitly reintroduced, which motivates the next chunk.

---

## Chunk: positional-encoding

**ID**: `transformers#positional-encoding`

Because self-attention computes relevance between every pair of positions symmetrically,
it has no built-in sense of *where* in the sequence each position is — "the cat sat" and
"sat the cat" would produce identical attention computations unless position information
is added explicitly.

**Positional encoding** injects this order information by adding a position-dependent
vector to each input embedding before it enters the attention layers:

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

x_input = embedding(token) + PE(pos)
```

**Why sine and cosine functions:** These produce a unique, smoothly-varying pattern for
every position, and — because of trigonometric identities — allow the model to learn to
attend to *relative* positions (e.g., "the token 3 positions back") as a linear function of
the absolute positional encodings, rather than only ever learning fixed absolute-position
patterns.

**Key insight:** Positional encoding is what allows a fundamentally order-agnostic
mechanism (self-attention) to still be sensitive to sequence order — it's added once, at
the input, rather than built into the attention computation itself.

---

## Chunk: transformer-block-structure

**ID**: `transformers#transformer-block-structure`

A single Transformer block (repeated many times in a full model) consists of two main
sub-layers, each wrapped with a residual connection and normalization:

```
1. Multi-head self-attention sub-layer:
   x = x + MultiHeadAttention(x)     (residual/skip connection)
   x = LayerNorm(x)

2. Feedforward sub-layer:
   x = x + FeedForward(x)             (residual/skip connection)
   x = LayerNorm(x)
```

where `FeedForward(x)` is typically a small two-layer MLP applied independently to each
position: `FeedForward(x) = ReLU(x·W1 + b1)·W2 + b2` (connecting directly back to
`activation_functions#relu` and `perceptrons_mlp.md`).

**Why residual connections here:** Just as skip connections mitigate vanishing gradients
in very deep CNNs (mentioned in `backpropagation#vanishing-gradients`), they serve the
same purpose in Transformers — which are often stacked dozens of blocks deep. The residual
path `x + SubLayer(x)` gives gradients a direct route back through the network that
doesn't have to pass through every sub-layer's full transformation.

**Why LayerNorm rather than BatchNorm:** Transformers typically use **Layer Normalization**
instead of the Batch Normalization covered in `batch_normalization.md`. LayerNorm
normalizes across the feature dimension for each individual example independently, rather
than across the batch dimension — this matters because sequence lengths vary between
examples, making batch statistics (which assume a shared, well-defined batch structure)
less natural to compute consistently in sequence models.

---

## Chunk: encoder-decoder-architecture

**ID**: `transformers#encoder-decoder-architecture`

The original Transformer, designed for translation tasks, has two stacks:

**Encoder stack:** Processes the input sequence using self-attention (every position can
attend to every other position, including future ones — the full input is available at
once).

**Decoder stack:** Generates the output sequence one token at a time, using two types of
attention:
- **Masked self-attention:** attends only to previous output positions, not future ones —
  since future output tokens don't exist yet when generating token t, the model must not
  be allowed to "see" them during training, which is enforced by masking those positions
  out of the softmax computation.
- **Cross-attention:** the decoder's queries attend to the encoder's keys and values,
  letting each generated output token pull relevant information from anywhere in the input
  sequence.

**Encoder-only vs. decoder-only variants:** Many modern models use only one half of this
architecture — encoder-only models (e.g., for classification/understanding tasks) use
unmasked self-attention throughout, while decoder-only models (e.g., for text generation)
use masked self-attention throughout and drop the cross-attention step entirely.

---

## Chunk: connecting-back-to-fundamentals

**ID**: `transformers#connecting-back-to-fundamentals`

Despite its complexity, a Transformer is trained with the exact same core machinery
covered earlier in this curriculum:

- A **loss function** (typically cross-entropy for next-token prediction, see
  `loss_functions#cross-entropy`) is computed at the output.
- **Backpropagation** (`backpropagation.md`) computes gradients through the entire stack
  of attention and feedforward layers via the chain rule, exactly as in any other network
  — self-attention is simply a more complex, differentiable function whose local gradients
  the framework's autodiff system (`backpropagation#computational-graph`) already knows
  how to compute.
- An **optimizer** (typically Adam, see `optimizers.md`) applies the computed gradients to
  update every weight, including the attention projection matrices W_Q, W_K, W_V.
- **Regularization** techniques like dropout (`overfitting_regularization#dropout`) are
  commonly applied within the feedforward and attention sub-layers to reduce overfitting.

**Key insight:** Transformers are not a departure from the fundamentals covered
throughout this curriculum — they are a specific, highly effective architectural
arrangement of feedforward layers, non-linear activations, normalization, and gradient-
based training, built around the self-attention operation instead of recurrence or
convolution.

---

## Socratic Question Seeds

1. "Self-attention computes relevance between every pair of positions symmetrically, with
   no inherent sense of order. If you shuffled the words in a sentence before feeding them
   into a self-attention layer with no positional encoding, what would change about the
   attention computation itself?"

2. "Positional encoding uses sine and cosine functions rather than, say, just numbering
   positions 1, 2, 3, ... directly. What might go wrong with directly using raw position
   numbers as the sequence gets very long, especially for a network trained mostly on
   shorter sequences?"

3. "In the decoder, masked self-attention prevents a position from attending to future
   tokens. What would go wrong during training if this masking were removed, given that
   the model is being trained to predict the next token?"

4. "A Transformer block wraps each sub-layer in a residual connection: x = x + SubLayer(x).
   If a Transformer has 96 stacked blocks, how does this residual structure relate to the
   vanishing gradient concerns raised earlier for very deep networks?"

5. "Given that self-attention lets every position directly attend to every other position
   in a single step, why might this be particularly advantageous for a task like
   connecting a pronoun to a noun that appeared 40 words earlier in a paragraph, compared
   to an RNN handling the same task?"

---

## References

- Vaswani, A. et al. (2017) — *Attention Is All You Need*
- Devlin, J. et al. (2018) — *BERT: Pre-training of Deep Bidirectional Transformers for
  Language Understanding* (encoder-only variant)
- Radford, A. et al. (2018) — *Improving Language Understanding by Generative
  Pre-Training* (decoder-only / GPT variant)
- Alammar, J. — *The Illustrated Transformer* (jalammar.github.io)
- Ba, Kiros & Hinton (2016) — *Layer Normalization*
