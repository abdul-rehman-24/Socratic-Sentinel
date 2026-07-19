---
id: cnn_advanced_architectures
title: "Advanced CNN Architectures: ResNet and Inception"
topic_group: "advanced"
prerequisites: ["cnn_fundamentals", "cnn_pooling_padding", "backpropagation"]
difficulty: advanced
chunk_version: 1
tags: ["resnet", "inception", "skip-connections", "residual-learning", "cnn-architecture"]
---

# Advanced CNN Architectures: ResNet and Inception

## Overview

Early CNN research generally found that deeper networks performed better — until they
didn't. Simply stacking more convolutional layers eventually caused training to get
*worse*, not better, even on the training set itself. **ResNet** and **Inception** are two
influential architectural innovations that addressed this and related problems, each from
a different angle: ResNet by changing what each layer is allowed to learn, and Inception
by changing what each layer is allowed to compute.

---

## Chunk: the-degradation-problem

**ID**: `cnn_advanced_architectures#the-degradation-problem`

Beyond the vanishing gradient problem covered in `backpropagation#vanishing-gradients`,
researchers observed a further puzzle: even very deep networks that *could* still train
(gradients weren't fully vanishing) sometimes performed *worse* on training data than
shallower versions of the same architecture.

**Why this is surprising:** In principle, a deeper network should be able to at least
match a shallower one — the extra layers could simply learn to compute the identity
function (pass their input through unchanged), making the deep network behave exactly like
the shallow one. But in practice, plain stacked convolutional layers struggled to learn
this identity mapping through ordinary gradient descent — leaving deeper plain networks
performing worse rather than merely no better.

**Key insight:** This is a distinct issue from vanishing gradients — it's an
*optimization* difficulty (the network can't easily learn to be at least as good as a
shallower version), not purely a *gradient magnitude* problem.

---

## Chunk: residual-connections

**ID**: `cnn_advanced_architectures#residual-connections`

**ResNet (Residual Network)** addresses the degradation problem by changing what each
block of layers is asked to learn. Instead of a block learning a direct mapping H(x), it
learns a **residual** F(x) = H(x) - x, and the block's actual output is:

```
output = F(x) + x
```

where x is passed forward unchanged via a **skip connection** (also called a shortcut
connection) that bypasses the block's learned transformation entirely.

**Why this solves the degradation problem:** If the optimal thing for a block to do is
nothing at all (the identity function), the block only needs to learn F(x) = 0 — pushing
weights toward zero is a far easier optimization target for gradient descent than learning
to exactly reconstruct an identity mapping through a stack of nonlinear transformations.
The skip connection guarantees the identity path is always available "for free," so adding
more layers can never make the network strictly worse at replicating a shallower one.

**Effect on gradient flow:** As referenced in `backpropagation#vanishing-gradients`, skip
connections also provide a direct path for gradients to flow backward, largely bypassing
the repeated multiplications through many layers' worth of weights and activation
derivatives. This is structurally similar in spirit to the additive cell-state update in
LSTMs (`lstm_gru#gradient-flow-through-cell-state`) — both create an additive shortcut that
avoids compounding attenuation.

```
∂L/∂x = ∂L/∂output · (∂F(x)/∂x + 1)
```

The "+1" term means even if ∂F(x)/∂x becomes very small (a near-vanishing gradient through
the learned path), the gradient can still flow through the "+1" identity term largely
unattenuated.

---

## Chunk: residual-blocks-in-practice

**ID**: `cnn_advanced_architectures#residual-blocks-in-practice`

A typical residual block stacks a small number of convolutional layers (commonly two or
three) inside the F(x) path, with batch normalization (`batch_normalization.md`) and ReLU
activations (`activation_functions#relu`) between them:

```
x → [Conv → BatchNorm → ReLU → Conv → BatchNorm] → F(x)
x ────────────────────────────────────────────────→ (skip connection)
                        F(x) + x → ReLU → output
```

**Dimension mismatch handling:** When F(x) changes the number of channels or spatial size
(e.g., via a strided convolution for downsampling), the skip connection path also needs a
matching transformation — typically a 1×1 convolution — so that F(x) and x can be added
element-wise. This is sometimes called a **projection shortcut**, as opposed to the
**identity shortcut** used when dimensions already match.

**Practical impact:** ResNet architectures with over 100 layers (and later, over 1000 in
research settings) were successfully trained using this approach, at depths where plain
stacked-convolution networks had previously failed to train effectively at all.

---

## Chunk: inception-motivation

**ID**: `cnn_advanced_architectures#inception-motivation`

A separate design question in CNN architecture is: **what filter size should each
convolutional layer use?** A 1×1 filter captures very local, per-pixel patterns; a 5×5
filter captures broader spatial patterns; a 3×3 filter sits in between. Different features
in an image may naturally be captured at different scales — a network designer choosing a
single fixed filter size for every layer is making an arbitrary commitment that may not
suit every feature the network needs to learn.

**Key insight — the Inception module's answer:** Rather than choosing one filter size,
apply *several* filter sizes in parallel within the same layer, and let the network learn
which ones are useful for which purpose.

---

## Chunk: inception-module-structure

**ID**: `cnn_advanced_architectures#inception-module-structure`

A basic Inception module applies multiple convolution filter sizes (and a pooling
operation) to the same input in parallel, then concatenates all their outputs along the
channel dimension:

```
Input
  ├── 1×1 conv ─────────────────┐
  ├── 1×1 conv → 3×3 conv ──────┤
  ├── 1×1 conv → 5×5 conv ──────┼── Concatenate (channel-wise) → Output
  └── 3×3 max pool → 1×1 conv ──┘
```

**Why the extra 1×1 convolutions before the larger filters:** Applying a 5×5 convolution
directly across many channels is computationally expensive — the cost scales with the
number of input channels times the number of output channels times the filter area. A 1×1
convolution first reduces the number of channels (a "bottleneck"), so the subsequent
larger, more expensive convolution operates on far fewer channels — dramatically reducing
computation while a network can still learn to preserve the information that matters most
through that bottleneck.

**Connection to parameter sharing:** Every filter within an Inception module still uses
the same parameter-sharing principle covered in `cnn_fundamentals#parameter-sharing` — the
innovation here is about running *multiple* differently-sized shared filters in parallel
within one layer, not a change to how any individual filter operates.

---

## Chunk: resnet-vs-inception-design-philosophy

**ID**: `cnn_advanced_architectures#resnet-vs-inception-design-philosophy`

| Aspect | ResNet | Inception |
|---|---|---|
| Core problem addressed | Degradation with depth (optimization difficulty) | Choosing the right filter size at each layer |
| Mechanism | Additive skip connections | Parallel multi-scale filters, concatenated |
| Primary benefit | Enables training networks hundreds of layers deep | Captures multi-scale features efficiently per layer |
| Computational strategy | Same computation per layer, more layers | 1×1 bottlenecks reduce cost of wider layers |

**Key insight:** These are not competing solutions to the same problem — they address
different architectural questions (how deep can a network go, versus how wide/multi-scale
should each individual layer be), and many later architectures (e.g., Inception-ResNet)
combine both ideas: Inception-style parallel filters *within* residual blocks.

---

## Socratic Question Seeds

1. "If adding more layers to a plain CNN can only help or stay neutral in theory (since
   extra layers could always learn the identity function), but in practice deeper plain
   networks performed worse, what does that suggest about how hard it is for gradient
   descent to learn an identity mapping through a stack of nonlinear layers?"

2. "In the residual gradient equation ∂L/∂x = ∂L/∂output · (∂F(x)/∂x + 1), what happens to
   the overall gradient if ∂F(x)/∂x shrinks all the way to zero? Does the network still
   receive any gradient signal at that block?"

3. "An Inception module runs 1×1, 3×3, and 5×5 convolutions on the exact same input in
   parallel. Why might a network benefit from having all three available simultaneously,
   rather than a designer having to pick just one filter size for the whole layer?"

4. "A 5×5 convolution applied directly to 256 input channels is far more expensive than
   the same 5×5 convolution applied to 64 channels. How does a 1×1 convolution placed
   before it change this computational cost, and what's it actually doing to the data as
   it passes through?"

5. "ResNet's skip connection adds x directly to F(x)'s output. What would need to change
   about this addition if F(x) produced a feature map with a different number of channels
   than x?"

---

## References

- He, K., Zhang, X., Ren, S. & Sun, J. (2015) — *Deep Residual Learning for Image
  Recognition* (ResNet)
- Szegedy, C. et al. (2014) — *Going Deeper with Convolutions* (Inception / GoogLeNet)
- Szegedy, C. et al. (2016) — *Rethinking the Inception Architecture for Computer Vision*
- Szegedy, C. et al. (2016) — *Inception-v4, Inception-ResNet and the Impact of Residual
  Connections on Learning*
