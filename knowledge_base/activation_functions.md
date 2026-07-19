---
id: activation_functions
title: "Activation Functions"
topic_group: "core"
prerequisites: ["perceptrons_mlp"]
difficulty: beginner
chunk_version: 1
tags: ["activation-functions", "relu", "sigmoid", "tanh", "softmax", "non-linearity"]
---

# Activation Functions

## Overview

An **activation function** is applied to the pre-activation output of a layer (the weighted
sum z = Wx + b) to produce the layer's actual output. Activation functions are what give
neural networks their expressive power — without them, stacking layers would be
mathematically pointless, no matter how many layers you add.

Activation functions serve two distinct purposes depending on where they sit in the
network: **hidden layers** use them to introduce non-linearity so the network can model
complex functions, while the **output layer** uses them to shape raw scores into a form
appropriate for the task (probabilities for classification, unbounded values for
regression).

---

## Chunk: why-nonlinearity

**ID**: `activation_functions#why-nonlinearity`

Consider a network with L layers but **no activation function** — just raw linear
transformations:

```
a[1] = W[1]x + b[1]
a[2] = W[2]a[1] + b[2] = W[2]W[1]x + (W[2]b[1] + b[2])
```

Every composition of linear functions is itself a single linear function. No matter how
many layers you stack, the entire network collapses to:

```
ŷ = W_combined · x + b_combined
```

This is mathematically identical to a single-layer perceptron — depth adds zero
representational power without non-linearity.

**Key insight:** The activation function is what breaks this collapse. By applying a
non-linear function σ between layers, each additional layer can carve out genuinely new
decision boundaries — curves, not just hyperplanes.

**MISCONCEPTION — ACTIVATION_ROLE:**
A common belief is that activation functions "just scale" or "normalize" the values
passing through a layer, similar to a units conversion. This is incorrect. Their essential
job is introducing **non-linearity** — without which depth itself provides no benefit.
A network of any depth with only linear activations is exactly as expressive as a network
with one layer.

---

## Chunk: sigmoid

**ID**: `activation_functions#sigmoid`

```
σ(z) = 1 / (1 + e^(-z))
```

Sigmoid squashes any real-valued input into the range (0, 1). Historically popular for
hidden layers, it's now mostly reserved for specific use cases: binary classification
output layers, and gates in LSTM/GRU cells.

**Derivative:**
```
σ'(z) = σ(z)(1 - σ(z))
```

The derivative peaks at 0.25 when z = 0 and approaches 0 as |z| grows large in either
direction — this is the **saturation** behavior directly responsible for the vanishing
gradient problem covered in `backpropagation#vanishing-gradients`.

**Why it's rarely used in hidden layers today:** For large positive or negative inputs,
the curve flattens almost completely. A neuron operating in this flat region produces
a near-zero gradient, so backpropagation barely updates its incoming weights — the neuron
effectively stops learning.

---

## Chunk: tanh

**ID**: `activation_functions#tanh`

```
tanh(z) = (e^z - e^(-z)) / (e^z + e^(-z))
```

Tanh squashes inputs into the range (-1, 1) — a zero-centered version of sigmoid. Because
its outputs are centered around zero rather than 0.5, gradients flowing to the next layer
tend to be better-behaved than with sigmoid, which is why tanh generally outperforms
sigmoid in hidden layers when saturating activations must be used.

**Derivative:**
```
tanh'(z) = 1 - tanh²(z)
```

Maximum value is 1.0 at z = 0, but like sigmoid, it still saturates toward 0 for large
|z| — tanh reduces but does not eliminate the vanishing gradient problem.

---

## Chunk: relu

**ID**: `activation_functions#relu`

```
ReLU(z) = max(0, z)
```

Rectified Linear Unit is the default choice for hidden layers in most modern
architectures. Its derivative is trivially simple:

```
ReLU'(z) = 1 if z > 0
         = 0 if z ≤ 0
```

**Why ReLU largely solved the vanishing gradient problem for hidden layers:** For any
active neuron (z > 0), the local gradient is exactly 1 — no shrinking, no matter how many
layers deep. This is the key detail referenced in
`backpropagation#vanishing-gradients`: a 100-layer ReLU network can still propagate
strong gradients to early layers, something a 100-layer sigmoid network cannot do.

**Trade-off — the "dying ReLU" problem:** A neuron whose z is always ≤ 0 (for every
input in the dataset) has zero gradient forever — it "dies" and stops contributing.
Leaky ReLU (`LeakyReLU(z) = max(αz, z)` for small α, e.g. 0.01) addresses this by allowing
a small non-zero gradient for z < 0.

---

## Chunk: softmax

**ID**: `activation_functions#softmax`

```
softmax(z)_i = e^(z_i) / Σ_j e^(z_j)
```

Softmax converts a vector of raw scores (logits) into a probability distribution: every
output is in (0, 1), and all outputs sum to exactly 1.

**Softmax is applied at the output layer only**, in multi-class classification networks.
It is not a hidden-layer activation function. Hidden layers need per-neuron non-linearity
(ReLU, tanh, etc.); softmax instead performs a joint, cross-neuron normalization across
the entire output vector — every output value depends on every other output value, which
is exactly the wrong property for a hidden layer, where neurons should be able to activate
independently based on the features they detect.

**MISCONCEPTION — SOFTMAX_SCOPE:**
Some students assume softmax is applied after every layer, treating it as a general-purpose
"squash the outputs" step similar to ReLU or sigmoid. In a standard classification network,
softmax appears exactly once — at the final layer — to turn logits into class
probabilities. Applying it between hidden layers would destroy the independent,
feature-detecting role each hidden neuron is meant to play.

**Connection to loss:** Softmax is almost always paired with cross-entropy loss (see
`loss_functions#cross-entropy`). As shown in `backpropagation#backward-pass`, this pairing
produces an unusually clean gradient at the output layer: δ[L] = ŷ - y.

---

## Chunk: choosing-activations

**ID**: `activation_functions#choosing-activations`

A practical rule of thumb for where each function belongs:

| Location | Typical choice | Why |
|---|---|---|
| Hidden layers (default) | ReLU / Leaky ReLU | Non-saturating gradient, cheap to compute |
| Hidden layers (RNN gates) | Sigmoid, tanh | Bounded outputs needed for gating behavior |
| Output layer — binary classification | Sigmoid | Single probability in (0, 1) |
| Output layer — multi-class classification | Softmax | Probability distribution over classes |
| Output layer — regression | None (linear) | Unbounded real-valued target |

**Key insight:** There is no single "best" activation function — the right choice depends
on where in the network you are and what range of output values that layer needs to
produce.

---

## Socratic Question Seeds

1. "If you stacked ten layers together but removed every activation function, what would
   the combined transformation from input to output actually look like algebraically?"

2. "Sigmoid squashes values into (0, 1) — but so does dividing every input by a very large
   constant. What's functionally different between sigmoid and a simple rescaling
   operation, in terms of what each one lets the network learn?"

3. "Looking at the derivative of ReLU for z > 0, how does it compare to the derivative of
   sigmoid at its steepest point? What does that difference mean for a gradient traveling
   through fifty layers?"

4. "If softmax makes every output depend on every other output in the vector, what would
   happen to a hidden layer's ability to detect independent features if you inserted
   softmax between two hidden layers?"

5. "A binary classifier and a 10-class classifier both need a probability distribution at
   the output. Why might one use sigmoid and the other use softmax instead of both just
   using softmax?"

---

## References

- Nair, V. & Hinton, G. (2010) — *Rectified Linear Units Improve Restricted Boltzmann Machines*
- Goodfellow, Bengio & Courville — *Deep Learning* (MIT Press), Chapter 6.3
- Nielsen, M. — *Neural Networks and Deep Learning*, Chapter 4
- Maas, Hannun & Ng (2013) — *Rectifier Nonlinearities Improve Neural Network Acoustic Models* (Leaky ReLU)
