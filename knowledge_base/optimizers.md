````markdown
---
id: optimizers
title: "Optimizers (SGD, Momentum, Adam)"
topic_group: "core"
prerequisites: ["gradient_descent"]
difficulty: intermediate
chunk_version: 1
tags: ["optimizers", "sgd", "adam", "momentum"]
---

# Optimizers

## Overview

Optimizers modify the basic gradient descent update to improve convergence speed
and stability. They can add momentum (memory of prior updates), adapt per-parameter
learning rates, or both.

Common optimizers include:
- **SGD** (with optional momentum)
- **RMSProp**
- **Adam** (adaptive moment estimation)

---

## Chunk: momentum

**ID**: `optimizers#momentum`

Momentum keeps an exponentially weighted moving average of past gradients.
Update rules (simplified):

```
v <- beta * v + (1 - beta) * grad
theta <- theta - eta * v
```

Momentum helps accelerate learning in directions with consistent gradients and
reduces oscillation in noisy directions.

**MISCONCEPTION — MOMENTUM_IS_COMPLEX:**
Momentum is conceptually similar to adding inertia; tuning beta (momentum term)
is important but often less sensitive than learning rate.

---

## Chunk: adam

**ID**: `optimizers#adam`

Adam keeps running averages of both gradients and squared gradients (first and
second moments). It adapts the step size per-parameter and includes bias correction.

High-level behavior:

- Fast convergence for many problems
- Works well with default parameters in many cases (beta1=0.9, beta2=0.999)
- Can sometimes generalize slightly worse than well-tuned SGD with momentum

**MISCONCEPTION — ADAM_ALWAYS_BETTER:**
Adam is not a universal replacement; for some large-scale problems, SGD with
momentum generalizes better when carefully tuned.

---

## Socratic Question Seeds

1. "Why might remembering past gradient directions help learning in a valley-shaped loss surface?"
2. "How does adapting the learning rate per-parameter help when some parameters have much larger gradients than others?"

## References

- Kingma & Ba — *Adam: A Method for Stochastic Optimization*
- Duchi et al. — *Adagrad*

````
---
id: optimizers
title: "Optimizers: SGD, Momentum, Adam, RMSProp"
topic_group: "core"
prerequisites: ["gradient_descent", "backpropagation"]
difficulty: intermediate
chunk_version: 1
tags: ["sgd", "adam", "momentum", "rmsprop", "learning-rate", "training"]
---

# Optimizers

> **Status**: Placeholder — detailed curriculum content coming in Day 2.

## Topics to cover:
- Vanilla SGD and its limitations
- Momentum: exponential moving average of gradients
- RMSProp: adaptive learning rates per parameter
- Adam: combines Momentum + RMSProp
- Learning rate schedules and warmup
- Hyperparameter sensitivity

## Known misconceptions to address:
- "Higher learning rate always means faster training"
- "Adam always outperforms SGD"
- "Momentum makes training unstable"
