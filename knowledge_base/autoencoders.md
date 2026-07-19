---
id: autoencoders
title: "Autoencoders"
topic_group: "advanced"
prerequisites: ["perceptrons_mlp", "loss_functions", "backpropagation"]
difficulty: intermediate
chunk_version: 1
tags: ["autoencoders", "encoder-decoder", "latent-space", "dimensionality-reduction", "unsupervised-learning"]
---

# Autoencoders

## Overview

An **autoencoder** is a neural network trained to reconstruct its own input — the target
output y is simply the input x itself. This might seem pointless at first (why train a
network to output what you already have?), but the trick is architectural: the network is
forced to compress the input through a narrow **bottleneck** before reconstructing it,
forcing it to learn a compact, meaningful representation of the data rather than simply
memorizing or copying it.

Autoencoders are trained **unsupervised** (or more precisely, "self-supervised") — no
labels are needed, since the input itself serves as the target.

---

## Chunk: encoder-decoder-structure

**ID**: `autoencoders#encoder-decoder-structure`

An autoencoder consists of two parts trained jointly:

```
x → [Encoder] → z (latent representation) → [Decoder] → x̂ (reconstruction)
```

- The **encoder** compresses the input x into a lower-dimensional **latent
  representation** z (also called a "code" or "embedding").
- The **decoder** takes z and attempts to reconstruct the original input as x̂.

**The bottleneck is the entire point:** If z had the same (or higher) dimensionality as x,
the network could trivially learn to just copy the input straight through without learning
anything useful. By forcing z to be lower-dimensional than x, the network *must* discard
some information — and training pushes it to discard the least useful information (noise,
redundancy) while keeping the most useful information (the underlying structure needed to
reconstruct the input well).

**Training objective:** A reconstruction loss, typically Mean Squared Error (see
`loss_functions#mean-squared-error`) between the original input and its reconstruction:
```
L = ||x - x̂||²
```
This loss is backpropagated (`backpropagation.md`) through the entire encoder-decoder
network simultaneously — both halves are trained jointly, not separately.

---

## Chunk: the-latent-space

**ID**: `autoencoders#the-latent-space`

The compressed representation z lives in what's called **latent space** — a learned,
lower-dimensional coordinate system where each dimension (ideally) captures some
meaningful underlying factor of variation in the data, rather than corresponding directly
to any single raw input feature (like one specific pixel).

**Why this is useful beyond reconstruction:**
- **Dimensionality reduction:** z can be used as a compact feature representation for
  other downstream tasks (e.g., feeding z into a separate classifier instead of the raw,
  high-dimensional input).
- **Denoising:** if trained appropriately, the decoder learns to reconstruct clean,
  plausible outputs even from a noisy or corrupted input, since the bottleneck already
  discards information that doesn't fit the learned structure of the data.
- **Anomaly detection:** since the autoencoder is trained to reconstruct only the patterns
  present in its training data, an input that's very different from anything it saw during
  training will typically produce a poor reconstruction — the reconstruction error itself
  becomes a usable anomaly signal.

**Key insight:** The quality of the latent space (how meaningful and well-organized it is)
depends heavily on the bottleneck size and architecture — too large a bottleneck risks
simply memorizing inputs (weak compression, poor generalization of the encoding); too
small a bottleneck may lose important information needed for good reconstruction.

---

## Chunk: undercomplete-vs-overcomplete

**ID**: `autoencoders#undercomplete-vs-overcomplete`

An **undercomplete** autoencoder has a latent dimension smaller than the input dimension —
this is the standard setup described above, where the bottleneck itself forces useful
compression.

An **overcomplete** autoencoder has a latent dimension equal to or larger than the input
dimension. Without additional constraints, this defeats the purpose entirely: the network
can simply learn the identity function, copying inputs through without extracting any
useful structure — since there's no bottleneck forcing it to discard anything.

**Making overcomplete autoencoders useful — regularization:** To prevent a large-latent-
dimension autoencoder from trivially copying its input, additional constraints must be
added, echoing the regularization strategies covered in `overfitting_regularization.md`:
- **Sparse autoencoders** add a penalty encouraging most latent dimensions to be near zero
  for any given input, forcing the network to represent each input using only a small
  active subset of the latent dimensions — conceptually similar in spirit to how L2
  regularization (`overfitting_regularization#l2-weight-decay`) constrains weight
  magnitudes to prevent trivial overfitting solutions.
- **Denoising autoencoders** deliberately corrupt the input with noise before feeding it
  to the encoder, but still require reconstruction of the *original, clean* input — this
  prevents copying, since the corrupted input and the target output are no longer
  identical.

---

## Chunk: autoencoders-vs-pca

**ID**: `autoencoders#autoencoders-vs-pca`

Principal Component Analysis (PCA) is a classical, non-neural-network technique for
dimensionality reduction that shares the same conceptual goal as an autoencoder:
compressing data into fewer dimensions while preserving as much useful information as
possible.

**Key difference:** PCA is restricted to finding a *linear* lower-dimensional subspace
that best preserves variance in the data. A linear autoencoder (one with no activation
functions, or only linear ones) is mathematically closely related to PCA. But because
autoencoders use non-linear activation functions (`activation_functions.md`) between
layers, they can learn much more expressive, *non-linear* compressions — capturing curved
or complex manifold structure in the data that a purely linear method like PCA cannot
represent.

---

## Socratic Question Seeds

1. "If an autoencoder's latent dimension z were the exact same size as the input x, what
   would stop the network from just learning to copy the input straight through without
   learning anything meaningful about its structure?"

2. "A denoising autoencoder is given a corrupted input but must reconstruct the original,
   clean version. Why does this setup prevent the network from simply learning an identity
   mapping, even if the latent dimension is large?"

3. "An autoencoder trained only on images of cats is later given an image of a car as
   input. Based on what the bottleneck was forced to learn during training, what would you
   expect to happen to the reconstruction quality, and how could that be used to detect
   that this input doesn't belong?"

4. "PCA can only capture linear relationships in data, while a non-linear autoencoder can
   capture curved or complex structure. What specific component of the autoencoder's
   architecture is responsible for this extra expressive power, based on what you know
   about activation functions?"

5. "If you made an autoencoder's bottleneck extremely small — say, compressing a
   high-resolution image down to just 2 latent numbers — what trade-off would you expect
   between reconstruction quality and how compressed the representation is?"

---

## References

- Hinton, G. & Salakhutdinov, R. (2006) — *Reducing the Dimensionality of Data with Neural
  Networks*
- Vincent, P. et al. (2008) — *Extracting and Composing Robust Features with Denoising
  Autoencoders*
- Goodfellow, Bengio & Courville — *Deep Learning* (MIT Press), Chapter 14
