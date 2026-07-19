---
id: hyperparameter_tuning
title: "Hyperparameter Tuning"
topic_group: "advanced"
prerequisites: ["gradient_descent", "optimizers", "overfitting_regularization"]
difficulty: intermediate
chunk_version: 1
tags: ["hyperparameters", "tuning", "grid-search", "random-search", "validation-set", "learning-rate"]
---

# Hyperparameter Tuning

## Overview

A **hyperparameter** is any setting chosen *before* training begins and held fixed
throughout it — as opposed to a **parameter** (weight or bias), which is learned *during*
training via gradient descent. Learning rate, batch size, number of layers, dropout rate,
and choice of optimizer are all hyperparameters. Since gradient descent cannot optimize
these values itself, finding good hyperparameters requires a separate search process.

---

## Chunk: parameters-vs-hyperparameters

**ID**: `hyperparameter_tuning#parameters-vs-hyperparameters`

**Parameters** (weights W and biases b) are updated automatically by gradient descent
using gradients computed via backpropagation (see `backpropagation.md`,
`gradient_descent.md`). The training process itself discovers their values.

**Hyperparameters** are chosen by the practitioner before training and are not touched by
the gradient descent update rule at all. Examples span every part of this curriculum:

| Hyperparameter | Where it's covered |
|---|---|
| Learning rate η | `optimizers.md` |
| Batch size | `gradient_descent.md` |
| Number of layers / neurons per layer | `perceptrons_mlp.md` |
| Dropout rate p | `overfitting_regularization#dropout` |
| L2 regularization strength λ | `overfitting_regularization#l2-weight-decay` |
| Choice of optimizer (SGD, Adam, etc.) | `optimizers.md` |
| Number of training epochs | `gradient_descent.md` |

**Key insight:** No amount of additional gradient descent training will fix a poorly chosen
hyperparameter — a learning rate that's too high will keep causing unstable updates no
matter how long training runs, since η itself is never adjusted by the gradient descent
rule it controls.

---

## Chunk: the-validation-set

**ID**: `hyperparameter_tuning#the-validation-set`

Hyperparameter tuning requires data the model was not trained on, to give an honest signal
of how a given hyperparameter setting generalizes — this is the role of the **validation
set**, distinct from both the training set and the test set:

```
Training set   →  used to compute gradients and update parameters (W, b)
Validation set →  used to compare different hyperparameter settings against each other
Test set       →  used only once, at the very end, to report final performance
```

**Why not just use the test set for tuning?** If hyperparameters are repeatedly chosen
based on test set performance, the test set effectively becomes part of the tuning
process — you're indirectly fitting your hyperparameter choices to that specific data,
producing an overly optimistic estimate of how the model will perform on truly unseen
data. This is a subtler version of the same overfitting concept covered in
`overfitting_regularization#what-is-overfitting` — except here it's the *hyperparameters*
being overfit to a specific dataset split, not the model's weights.

**Cross-validation:** A more data-efficient technique where the training data is split
into k folds; the model is trained k times, each time holding out a different fold as a
temporary validation set, and performance is averaged across all k runs — reducing the
risk that a single lucky (or unlucky) validation split skews the hyperparameter choice.

---

## Chunk: search-strategies

**ID**: `hyperparameter_tuning#search-strategies`

**Grid search:** Define a fixed set of candidate values for each hyperparameter, and
exhaustively try every combination. For example, learning rate ∈ {0.1, 0.01, 0.001} and
batch size ∈ {32, 64, 128} produces 9 total combinations to evaluate.

**Limitation:** The number of combinations grows exponentially with the number of
hyperparameters being tuned — grid search becomes computationally infeasible quickly as
more hyperparameters are added.

**Random search:** Instead of trying every combination, randomly sample hyperparameter
combinations from specified ranges (or distributions) a fixed number of times.

**Why random search often outperforms grid search for the same computational budget:**
Not all hyperparameters affect performance equally. If, say, learning rate matters far
more than batch size for a given problem, grid search wastes many evaluations trying
multiple batch sizes at each of only a few learning rate values. Random search, by
contrast, tends to explore a wider range of values for *every* hyperparameter across the
same total number of trials, increasing the chance of finding a good value for whichever
hyperparameter actually matters most — without needing to know in advance which one that
is.

**Bayesian optimization:** A more sophisticated approach that uses the results of
previous trials to intelligently choose which hyperparameter combination to try next,
rather than choosing points on a fixed grid or purely at random — generally more sample-
efficient than either grid or random search, at the cost of additional algorithmic
complexity.

---

## Chunk: learning-rate-schedules

**ID**: `hyperparameter_tuning#learning-rate-schedules`

Rather than treating the learning rate as a single fixed hyperparameter for the entire
training run, a **learning rate schedule** changes it systematically over the course of
training:

- **Step decay:** Reduce the learning rate by a fixed factor every N epochs.
- **Exponential decay:** Continuously shrink the learning rate by a fixed multiplicative
  factor after every step or epoch.
- **Warmup:** Start with a very small learning rate and gradually increase it over the
  first few epochs before switching to a decay schedule — this can help stabilize early
  training, particularly in architectures like Transformers (`transformers.md`), where
  large early updates to randomly initialized weights can destabilize the whole network.

**Why schedule the learning rate at all:** As discussed in relation to
`optimizers.md`, a learning rate large enough to make fast progress early in training
(when far from a good solution) is often too large for fine, precise convergence later in
training (when close to a good solution) — a fixed learning rate must compromise between
these two competing needs, while a schedule can be large early and small later,
addressing both phases of training more effectively.

---

## Chunk: automated-hyperparameter-tuning-caveats

**ID**: `hyperparameter_tuning#automated-hyperparameter-tuning-caveats`

Automated search methods reduce manual effort but do not remove the need for careful
experimental design:

- **Computational cost:** Each trial typically requires training a full model, which can
  be extremely expensive — automated search is often bounded by a fixed compute or time
  budget rather than run until "convergence" of the search itself.
- **Search space design still requires human judgment:** Choosing sensible ranges for each
  hyperparameter (e.g., searching learning rate on a logarithmic scale like {1e-1, 1e-2,
  1e-3, 1e-4} rather than a linear scale) has a large practical impact on how effective
  and efficient the search is.
- **Interaction effects:** Hyperparameters can interact — the best batch size may depend
  on the chosen learning rate, and vice versa — which is exactly why grid and random
  search explore combinations jointly, rather than tuning each hyperparameter
  independently in isolation.

---

## Socratic Question Seeds

1. "If a learning rate is set too high, no amount of additional training epochs will fix
   the instability it causes. Why can't gradient descent simply learn to correct its own
   learning rate the way it learns to correct the weights?"

2. "If you repeatedly checked your model's performance on the test set while trying
   different hyperparameter values, and picked whichever setting scored best on that test
   set, what would that test set score actually be measuring by the time you're done?"

3. "Grid search with 5 hyperparameters, each with 4 candidate values, requires 4^5 = 1024
   total combinations. What happens to this number as you add a 6th hyperparameter, and
   what does that tell you about grid search's scalability?"

4. "If learning rate matters far more to your model's performance than batch size does,
   why might random search find a good learning rate faster than grid search, given the
   same total number of trials?"

5. "A learning rate warmup starts small and increases before decaying. Why might starting
   with a very small learning rate be particularly important for a randomly initialized
   network in its very first few training steps?"

---

## References

- Bergstra, J. & Bengio, Y. (2012) — *Random Search for Hyper-Parameter Optimization*
- Snoek, J., Larochelle, H. & Adams, R. (2012) — *Practical Bayesian Optimization of
  Machine Learning Algorithms*
- Goodfellow, Bengio & Courville — *Deep Learning* (MIT Press), Chapter 11.4
- Smith, L. (2017) — *Cyclical Learning Rates for Training Neural Networks*