---
id: backpropagation
title: "Backpropagation"
topic_group: "core"
prerequisites: ["chain_rule", "mlp", "activation_functions", "loss_functions"]
difficulty: intermediate
chunk_version: 2
tags: ["backprop", "gradients", "chain-rule", "weight-update", "training", "gradient-descent"]
---

# Backpropagation

## Overview

Backpropagation (backward propagation of errors) is the algorithm that makes neural network
training possible. It efficiently computes the gradient of the loss function with respect
to every weight in the network by applying the **chain rule of calculus** recursively,
moving from the output layer back to the input layer.

Without backpropagation, computing gradients for a deep network would require a separate
forward pass for every single weight — computationally infeasible. Backprop reduces this
to approximately two forward passes worth of computation regardless of network depth.

---

## Chunk: chain-rule

**ID**: `backpropagation#chain-rule`

The **chain rule** is the mathematical backbone of backpropagation.

If a variable z depends on y, which depends on x, the chain rule states:

```
dz/dx = (dz/dy) · (dy/dx)
```

In a neural network, the scalar loss L depends on the network output ŷ, which depends
on the activation a[L], which depends on the pre-activation z[L], which depends on the
weights W[L] and the previous activation a[L-1] — and so on back to the input.

Applying the chain rule across this entire computational graph gives us:

```
∂L/∂W[l] = ∂L/∂a[l] · ∂a[l]/∂z[l] · ∂z[l]/∂W[l]
          = δ[l] · a[l-1]ᵀ
```

where δ[l] is the "error signal" at layer l (also called the local gradient).

**Key insight:** The chain rule allows gradient computation to be decomposed into local
operations at each layer — each layer only needs to know the gradient flowing into it
from the layer above, and its own local derivative. This modularity is what makes deep
network training tractable.

**MISCONCEPTION — GRADIENT_DIRECTION_CONFUSION:**
The gradient ∂L/∂W points in the direction of STEEPEST ASCENT of the loss.
Gradient DESCENT subtracts the gradient to move toward the minimum:
`W ← W - η · ∂L/∂W`
Many students believe gradients point toward the minimum — they point away from it.

---

## Chunk: forward-pass

**ID**: `backpropagation#forward-pass`

During the **forward pass**, activations flow from input to output. For a network with L layers:

```
a[0] = x                              (input layer)
z[l] = W[l] · a[l-1] + b[l]          (pre-activation: weighted sum + bias)
a[l] = σ(z[l])                        (activation: apply non-linearity)
L    = loss(a[L], y)                  (scalar loss at output)
```

**Critical implementation detail:** All intermediate values z[l] and a[l] MUST be cached
during the forward pass. The backward pass cannot be computed without them because:
- ∂a[l]/∂z[l] = σ'(z[l]) requires z[l]
- ∂z[l]/∂a[l-1] = W[l]ᵀ requires W[l]
- ∂z[l]/∂W[l] = a[l-1] requires the previous activation a[l-1]

**MISCONCEPTION — BACKPROP_VS_FORWARD_PASS:**
Weights are NOT updated during the forward pass. They are computed-on-only (read-only)
during forward. Weight updates happen ONLY after the complete backward pass, once all
gradients ∂L/∂W[l] have been computed for every layer.

The sequence is strictly:
1. Forward pass (compute and cache activations)
2. Backward pass (compute all gradients using cached values)
3. Weight update (apply gradient descent to all weights simultaneously)

---

## Chunk: backward-pass

**ID**: `backpropagation#backward-pass`

The **backward pass** computes gradients layer-by-layer from output to input.

**Step 1: Output layer error signal**
```
δ[L] = ∂L/∂z[L] = ∂L/∂a[L] · σ'(z[L])
```
For cross-entropy loss + softmax output, this simplifies beautifully to:
```
δ[L] = ŷ - y    (predicted probabilities minus one-hot true labels)
```

**Step 2: Backpropagate through hidden layers** (for l = L-1, L-2, ..., 1)
```
δ[l] = (W[l+1]ᵀ · δ[l+1]) ⊙ σ'(z[l])
```
Where:
- W[l+1]ᵀ · δ[l+1] routes the gradient from the layer above back through the weights
- ⊙ is element-wise multiplication (Hadamard product)
- σ'(z[l]) is the derivative of the activation function evaluated at the cached z[l]

**Step 3: Compute parameter gradients**
```
∂L/∂W[l] = δ[l] · a[l-1]ᵀ     (outer product)
∂L/∂b[l] = δ[l]
```

**Why does the error signal use W[l+1]ᵀ (transpose)?**
Because we need to route gradients "backwards" through the same weights that routed
activations forwards. The transpose reverses the linear map direction.

---

## Chunk: gradient-accumulation

**ID**: `backpropagation#gradient-accumulation`

**Gradient accumulation** refers to summing gradients across multiple examples before
performing a weight update. This is used in two important contexts:

**1. Mini-batch gradient descent:**
For a mini-batch of N examples, the loss is the mean loss:
```
L_batch = (1/N) · Σ L_i
```
By linearity of differentiation:
```
∂L_batch/∂W = (1/N) · Σ ∂L_i/∂W
```
Each example contributes an additive gradient — these are summed (accumulated)
and then divided by N before the weight update.

**2. Memory-constrained training:**
When a desired batch size exceeds GPU memory, you can process k smaller sub-batches,
accumulate their gradients (without updating weights), then perform one update.
This is mathematically equivalent to processing the full batch at once.

**Important:** Gradients accumulate additively. If you forget to zero the gradient buffer
between update steps, gradients from previous batches will contaminate current ones.
In PyTorch: `optimizer.zero_grad()` must be called before each backward pass.

**MISCONCEPTION — CHAIN_RULE_APPLICATION:**
A common error is computing ∂L/∂W[l] using ONLY the gradient from the directly adjacent
layer, forgetting that δ[l] already encodes the accumulated gradient from ALL layers above
via the repeated chain rule applications in Step 2 above.

---

## Chunk: vanishing-gradients

**ID**: `backpropagation#vanishing-gradients`

The **vanishing gradient problem** occurs when gradients become exponentially small as
they propagate backward through many layers, causing early layers to learn very slowly.

**Root cause (chain rule compounding):**
At each layer, the error signal is multiplied by σ'(z[l]) — the derivative of the
activation function. For sigmoid and tanh:

```
sigmoid'(z) = σ(z)(1 - σ(z))  →  max value: 0.25 (at z=0)
tanh'(z) = 1 - tanh²(z)        →  max value: 1.0  (at z=0), rapidly → 0
```

Over L layers, the gradient contains a product of L such terms:
```
δ[1] ≈ δ[L] · ∏ σ'(z[l])
```

If each σ'(z[l]) ≈ 0.25, after 10 layers: 0.25^10 ≈ 0.000001.
The gradient at layer 1 is one millionth of the gradient at layer 10.

**Effect:** Early layers receive near-zero gradient signals. Parameters barely update.
The network effectively "freezes" in early layers while only later layers learn.

**MISCONCEPTION — VANISHING_GRADIENT_CAUSE:**
The problem is NOT simply "too many layers" — it's specifically the combination of:
(a) saturating activations (sigmoid, tanh) whose derivatives approach zero, AND
(b) chain rule multiplication compounding those small values.

A 100-layer network using ReLU can train successfully because ReLU's derivative is either
0 or 1 — no compounding attenuation occurs for positive activations.

**Solutions:**
- Use ReLU or Leaky ReLU (derivative is 0 or 1, no saturation in positive region)
- Residual connections / skip connections (gradient highway bypasses deep chains)
- Careful weight initialization (Xavier/Glorot for tanh, He/Kaiming for ReLU)
- Batch normalization (normalizes pre-activations, keeps them in non-saturating range)
- Gradient clipping (addresses the exploding variant — gradient norm exceeds threshold)

---

## Chunk: computational-graph

**ID**: `backpropagation#computational-graph`

Modern deep learning frameworks (PyTorch, JAX, TensorFlow) implement backpropagation
through **automatic differentiation** on a **dynamic computational graph**.

Every operation in the forward pass is recorded as a node in a directed acyclic graph (DAG).
Each node stores:
- Its forward computation (e.g., matrix multiply)
- Its local gradient function (how to backpropagate through it)

During the backward pass, the framework traverses this graph in reverse topological order,
calling each node's gradient function and propagating the gradient backward.

This is why in PyTorch you call `loss.backward()` — it triggers reverse traversal of the
entire computation graph from the loss node back to every leaf (parameter) tensor.

**Advantages of this approach:**
- Handles arbitrary computation graphs (branches, conditionals, loops)
- Gradients are computed automatically — you only write the forward pass
- Dynamic graphs (PyTorch eager mode) allow the graph to change every iteration

**Connection to backpropagation:**
The chain rule equations derived earlier are exactly what each node's gradient function
implements. The framework automates their composition across the full graph.

---

## Socratic Question Seeds

1. "Before we talk about gradients flowing backward — what does the forward pass actually
   compute, and what information does it need to save along the way? Why can't we just
   recompute activations during the backward pass?"

2. "The gradient of the loss with respect to a weight tells us the slope of the loss
   surface in that weight's direction. If you wanted to reduce the loss, which direction
   along that slope would you move in — and what does that mean for the weight update rule?"

3. "What happens mathematically when you multiply five numbers together, each of which is
   between 0 and 0.25? Now imagine doing that across 50 layers. What does this tell us
   about why sigmoid networks struggle to be deep?"

4. "If weights were updated during the forward pass — before the backward pass computed
   gradients — what would go wrong with the gradient calculations that follow?"

5. "In the chain rule formula δ[l] = (W[l+1]ᵀ · δ[l+1]) ⊙ σ'(z[l]), why is W[l+1]
   transposed? What does the transpose do geometrically?"

6. "If you forgot to call optimizer.zero_grad() before each training step in PyTorch,
   what would happen to the gradients over time? How would this affect learning?"

---

## References

- Rumelhart, Hinton & Williams (1986) — *Learning representations by back-propagating errors* (Nature)
- Nielsen, M. — *Neural Networks and Deep Learning*, Chapter 2 (free at neuralnetworksanddeeplearning.com)
- Goodfellow, Bengio & Courville — *Deep Learning* (MIT Press), Chapter 6
- Karpathy, A. — *Yes you should understand backprop* (Medium, 2016)
- PyTorch Autograd documentation — https://pytorch.org/docs/stable/autograd.html
