---
id: loss_functions
title: "Loss Functions"
topic_group: "core"
prerequisites: ["perceptrons_mlp", "activation_functions"]
difficulty: beginner
chunk_version: 1
tags: ["loss-functions", "cross-entropy", "mean-squared-error", "cost-function", "objective"]
---

# Loss Functions

## Overview

A **loss function** quantifies how wrong a single prediction is compared to the true
target — a single scalar number that training tries to minimize. It is the quantity that
gradient descent and backpropagation are built around: every gradient computed in
`backpropagation#chain-rule` starts as ∂L/∂(something), where L is the loss.

Choosing the right loss function is not cosmetic — it directly shapes what "good" means
for the network, what gradients look like during training, and how the network's errors
get penalized.

**Terminology note:** "Loss" typically refers to the error on a single example, while
"cost function" refers to the loss averaged (or summed) across a batch or the full
dataset. In practice the two terms are often used interchangeably.

---

## Chunk: role-of-loss-function

**ID**: `loss_functions#role-of-loss-function`

The loss function sits at the very end of the forward pass, described in
`backpropagation#forward-pass`:

```
a[0] = x
...
a[L] = ŷ                    (network's prediction)
L    = loss(ŷ, y)            (scalar loss, comparing prediction to true label y)
```

Everything backpropagation computes flows from this single scalar. Changing the loss
function changes the error signal at the output layer (δ[L] in
`backpropagation#backward-pass`), which changes every gradient computed downstream of it.

**Key insight:** The loss function is a design choice, not a fixed property of the
network. The same network architecture can be trained toward very different behaviors
depending on which loss function drives its gradients.

---

## Chunk: mean-squared-error

**ID**: `loss_functions#mean-squared-error`

**Mean Squared Error (MSE)** is the standard loss for regression tasks, where the target
y is a continuous real value:

```
L_MSE = (1/N) · Σ (ŷ_i - y_i)²
```

**Properties:**
- Penalizes larger errors disproportionately more than smaller ones (a squared term), so
  a single large mistake contributes much more to the loss than several small ones.
- Its gradient with respect to the prediction is simple and smooth:
  ```
  ∂L_MSE/∂ŷ = 2(ŷ - y) / N
  ```
- Sensitive to outliers, since squaring amplifies large deviations.

**When it's used:** Regression problems — predicting house prices, temperatures, or any
continuous quantity — where the output layer has no activation function (or a linear one),
producing an unbounded real-valued prediction.

---

## Chunk: cross-entropy

**ID**: `loss_functions#cross-entropy`

**Cross-entropy loss** is the standard loss for classification tasks, measuring the
distance between two probability distributions: the true label distribution y (typically
one-hot) and the predicted distribution ŷ (typically the output of softmax, see
`activation_functions#softmax`).

**Binary cross-entropy** (for two-class problems, output via sigmoid):
```
L_BCE = -[y·log(ŷ) + (1-y)·log(1-ŷ)]
```

**Categorical cross-entropy** (for multi-class problems, output via softmax):
```
L_CE = -Σ_i y_i · log(ŷ_i)
```

Since y is one-hot (a 1 at the correct class index, 0 everywhere else), this simplifies to
just the negative log-probability the model assigned to the correct class:
```
L_CE = -log(ŷ_correct_class)
```

**Why it's shaped this way:** If the model assigns high probability to the correct class,
-log(ŷ_correct_class) is close to 0 (low loss). If the model assigns low probability to
the correct class, -log(ŷ_correct_class) grows very large — the loss punishes confident
wrong answers far more harshly than uncertain ones.

**Connection to backpropagation:** As noted in `backpropagation#backward-pass`, pairing
softmax with cross-entropy produces a remarkably clean output-layer gradient:
```
δ[L] = ŷ - y
```
This clean form is one of the main reasons the softmax + cross-entropy pairing is the
default for classification — the calculus works out to something simple and numerically
stable, rather than a coincidence of convenience.

---

## Chunk: matching-loss-to-task

**ID**: `loss_functions#matching-loss-to-task`

Using the wrong loss function for a task doesn't just underperform — it can actively work
against what the network needs to learn.

**Using MSE for classification (a common mistake):** MSE treats the output as a continuous
quantity and produces very small gradients when predictions are confidently wrong on a
bounded output like a sigmoid or softmax probability, because the derivative of MSE
combined with a saturating activation function's own derivative compounds toward zero.
This can cause training to stall exactly when the network most needs to correct itself.
Cross-entropy avoids this because its gradient does not depend on the activation's
derivative in the same compounding way when paired with softmax/sigmoid.

**Using cross-entropy for regression:** Cross-entropy assumes both ŷ and y are
probabilities in [0, 1] — it is undefined or meaningless for unbounded continuous targets
like price or temperature.

| Task | Output activation | Loss function |
|---|---|---|
| Regression | None (linear) | Mean Squared Error |
| Binary classification | Sigmoid | Binary cross-entropy |
| Multi-class classification | Softmax | Categorical cross-entropy |

---

## Chunk: loss-vs-metric

**ID**: `loss_functions#loss-vs-metric`

The **loss** is what the optimizer directly minimizes via gradient descent. A **metric**
(like accuracy, precision, or F1 score) is what humans typically use to judge model
quality — but metrics are usually not used as the loss itself.

**Why not just train directly on accuracy?** Accuracy is a step function — a prediction is
either exactly right or exactly wrong, with no notion of "how close." This means its
derivative is zero almost everywhere and undefined at the decision boundary, giving
gradient descent nothing to work with. Cross-entropy and MSE are both smooth and
differentiable everywhere, giving useful gradient signal even when a prediction is only
slightly better or worse than before.

**Key insight:** Loss and evaluation metric can diverge during training — it's possible
for loss to decrease while a metric like accuracy temporarily plateaus or even worsens,
particularly early in training or with imbalanced datasets. They are related but not
interchangeable.

---

## Socratic Question Seeds

1. "If a model predicts a house price of $310,000 when the true price is $300,000, versus
   predicting $500,000 when the true price is $300,000, how does squaring the error in MSE
   treat these two mistakes differently?"

2. "Cross-entropy loss involves a negative log of the predicted probability for the correct
   class. What happens to that value as the predicted probability approaches 0? What does
   that tell you about how cross-entropy treats a confident wrong answer?"

3. "If you tried to train a classifier directly on accuracy instead of cross-entropy, what
   would the derivative of accuracy look like with respect to a small change in the
   model's weights? What problem does that create for gradient descent?"

4. "Softmax outputs are always between 0 and 1. If you paired softmax with mean squared
   error instead of cross-entropy, what would happen to the gradient when the network is
   very confidently wrong?"

5. "Why does a regression task use an unbounded, activation-free output layer while a
   classification task deliberately bounds its output with sigmoid or softmax? What would
   go wrong if you swapped the two setups?"

---

## References

- Goodfellow, Bengio & Courville — *Deep Learning* (MIT Press), Chapter 6.2 (Cost Functions)
- Nielsen, M. — *Neural Networks and Deep Learning*, Chapter 3 (Cross-Entropy Cost Function)
- Murphy, K. — *Probabilistic Machine Learning: An Introduction*, Chapter 4 (loss functions and MLE)
- Bishop, C. — *Pattern Recognition and Machine Learning*, Chapter 5.2
