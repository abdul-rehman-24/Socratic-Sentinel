````markdown
---
id: gradient_descent
title: "Gradient Descent"
topic_group: "core"
prerequisites: ["backpropagation", "chain_rule", "loss_functions"]
difficulty: intermediate
chunk_version: 1
tags: ["gradient-descent", "learning-rate", "optimization"]
---

# Gradient Descent

## Overview

Gradient descent is the optimization algorithm that uses gradients (computed by
backpropagation) to iteratively update model parameters and minimize a loss function.
At each step, parameters are moved in the direction opposite to the gradient of the
loss with respect to those parameters.

The basic update rule for parameter vector \(\theta\) is:

$$
\theta \leftarrow \theta - \eta \nabla_{\theta} L(\theta)
$$

where \(\eta\) is the learning rate and \(\nabla_{\theta} L(\theta)\) is the gradient.

---

## Chunk: discrete-update

**ID**: `gradient_descent#discrete-update`

Gradient descent performs discrete updates to parameters. For a single weight
\(w\) the update is:

```
w <- w - eta * dL/dw
```

Choosing the learning rate \(\eta\) is critical: too large and the updates may
diverge; too small and convergence will be extremely slow.

**MISCONCEPTION — LEARNING_RATE_ZERO:**
Some students think the learning rate can be made arbitrarily small to guarantee
stability. While smaller rates can stabilize learning, they also increase the number
of steps needed and may get trapped in shallow regions without efficient progress.

---

## Chunk: batch-stochastic-mini-batch

**ID**: `gradient_descent#batch-stochastic-mini-batch`

Gradient descent comes in several flavors:

- **Batch Gradient Descent**: Compute gradient over the full dataset each step.
- **Stochastic Gradient Descent (SGD)**: Update using a single example at a time.
- **Mini-batch Gradient Descent**: Use small batches (e.g., 32–256 examples) per step.

Mini-batches strike a good balance between noisy updates (SGD) and expensive
full-batch computation. They also allow efficient use of GPUs.

**MISCONCEPTION — SGD_ALWAYS_BETTER:**
SGD is not always superior — its noisy updates can help escape shallow minima, but
proper batch sizing and learning rate schedules are crucial.

---

## Chunk: learning-rate-schedules

**ID**: `gradient_descent#learning-rate-schedules`

Common learning rate strategies:

- Fixed learning rate
- Step decay (drop by factor at fixed epochs)
- Exponential decay
- Cosine annealing
- Adaptive optimizers (see optimizers.md)

Schedules can dramatically affect convergence speed and final quality.

---

## Socratic Question Seeds

1. "If the gradient points in the direction of steepest increase, which way do we move the weights to reduce loss?"
2. "Why might using the entire dataset to compute each gradient step be impractical?"
3. "What could happen if the learning rate is set too high?"

## References

- Bottou, L. — *Large-scale machine learning with stochastic gradient descent*
- Goodfellow et al. — *Deep Learning*, Chapter on Optimization

````
