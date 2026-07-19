
---
id: transfer_learning
title: "Transfer Learning and Fine-Tuning"
topic_group: "advanced"
prerequisites: ["cnn_fundamentals", "overfitting_regularization", "backpropagation"]
difficulty: intermediate
chunk_version: 1
tags: ["transfer-learning", "fine-tuning", "pretrained-models", "feature-extraction", "frozen-layers"]
---
# Transfer Learning and Fine-Tuning

## Overview

**Transfer learning** is the practice of taking a model already trained on one task
(often on a very large dataset) and reusing some or all of its learned parameters as the
starting point for a new, related task — rather than training a new model entirely from
randomly initialized weights.

The key assumption behind transfer learning: features learned for one task (e.g., edges,
textures, and shapes learned by a CNN trained on general image classification) are often
useful for other, related tasks (e.g., classifying a specific type of medical image), even
though the two tasks have different final objectives.

---

## Chunk: why-transfer-learning-works

**ID**: `transfer_learning#why-transfer-learning-works`

In a deep CNN, different layers tend to learn increasingly abstract, task-specific
features as depth increases:

```
Early layers:  edges, colors, simple textures        (very general, task-agnostic)
Middle layers: shapes, parts, textures combinations   (moderately general)
Late layers:   task-specific object/class features    (highly task-specific)
```

**Key insight:** The early layers of a network trained on a large, diverse dataset (like
ImageNet, with millions of images across a thousand categories) tend to learn very general-
purpose visual features — the kind of low-level pattern detection that's useful for
*almost any* image-related task, not just the one the network was originally trained on.
This is what makes reusing those early layers for a new, different task effective, even
when the new task's data looks quite different from the original training data.

**Practical motivation:** Training a large network from scratch requires substantial
labeled data and compute. Many real-world tasks have far less labeled data available.
Transfer learning lets a new task benefit from features learned on a much larger dataset
than the new task alone could provide.

---

## Chunk: feature-extraction

**ID**: `transfer_learning#feature-extraction`

In the **feature extraction** approach to transfer learning, the pretrained network's
weights are kept entirely **frozen** (not updated during training on the new task) — only
a new final layer (or a small number of new layers) is added on top and trained from
scratch:

```
[Pretrained layers — FROZEN] → [New layer(s) — TRAINABLE] → new task output
```

**What "frozen" means in terms of the training loop:** During backpropagation
(`backpropagation.md`), gradients are still computed and flow backward through the frozen
layers as needed to reach the trainable new layers — but the weight update step
(`W ← W - η · ∂L/∂W`) is simply skipped for the frozen layers' parameters. They remain
exactly as they were in the original pretrained model.

**When this approach makes sense:** When the new task's dataset is small, and/or when the
new task is closely related to the original task the network was pretrained on — the
existing frozen features are likely to already be well-suited, and training additional
parameters on top risks less overfitting than trying to update the entire large network on
a small new dataset (connecting to `overfitting_regularization#what-is-overfitting` — a
huge model relative to a small new dataset is exactly the kind of capacity/data mismatch
that causes overfitting).

---

## Chunk: fine-tuning

**ID**: `transfer_learning#fine-tuning`

**Fine-tuning** goes a step further than feature extraction: rather than keeping the
pretrained layers completely frozen, some or all of them are unfrozen and continue to be
updated by gradient descent on the new task's data — typically starting from the
pretrained weights rather than random initialization.

**Common practice — partial fine-tuning:** Freeze the early layers (which learned general-
purpose features per `transfer_learning#why-transfer-learning-works`) and only unfreeze
and fine-tune the later, more task-specific layers, plus the new final layer(s):

```
[Early layers — FROZEN] → [Late layers — FINE-TUNED] → [New layer(s) — TRAINABLE]
```

**Why use a smaller learning rate during fine-tuning:** The pretrained weights already
encode useful, well-trained information. A learning rate as large as you might use for
training from scratch (see `optimizers.md`, `hyperparameter_tuning#learning-rate-schedules`)
risks large updates that destroy this useful pretrained structure before the network has a
chance to adapt it gradually to the new task — a problem sometimes called "catastrophic
forgetting" of the originally learned features.

**Feature extraction vs. fine-tuning — deciding factor:** More new-task data available, and
a new task less similar to the original pretraining task, both favor fine-tuning more
layers (since there's enough data to safely update more parameters without overfitting,
and the task is different enough that the original late-layer features may not transfer
as well). Less data and higher task similarity favor feature extraction or only fine-tuning
a small number of late layers.

---

## Chunk: domain-and-task-similarity

**ID**: `transfer_learning#domain-and-task-similarity`

How well transfer learning works depends heavily on how similar the original pretraining
task/data is to the new target task/data:

| Similarity to original task | New dataset size | Recommended approach                                                                               |
| --------------------------- | ---------------- | -------------------------------------------------------------------------------------------------- |
| High similarity             | Small            | Feature extraction (freeze most/all pretrained layers)                                             |
| High similarity             | Large            | Fine-tune more layers                                                                              |
| Low similarity              | Small            | Feature extraction, but expect weaker transfer; consider only using earliest (most general) layers |
| Low similarity              | Large            | Fine-tune extensively, or consider training substantial portions from scratch                      |

**Key insight:** Transfer learning is not a guarantee — if the new task is sufficiently
different from the original pretraining task (e.g., transferring from natural photographs
to medical X-rays, which have very different low-level statistics), even the "general"
early-layer features may transfer poorly, and the benefit of starting from pretrained
weights shrinks accordingly.

---

## Socratic Question Seeds

1. "If a CNN's early layers learn general features like edges and textures, while its late
   layers learn highly task-specific features, which layers would you expect to transfer
   well to a completely different image task, and which would you expect to transfer
   poorly?"
2. "During feature extraction, gradients still flow backward through the frozen layers to
   reach the trainable new layers, even though those frozen layers' weights never change.
   Why is this gradient flow still necessary, given that those weights won't be updated?"
3. "If you have only 200 labeled images for a new task, why might fine-tuning the entire
   pretrained network (all layers, not just the last few) risk worse performance than
   fine-tuning just the last layer or two?"
4. "Fine-tuning typically uses a much smaller learning rate than training from scratch.
   What could happen to the pretrained weights' useful structure if you fine-tuned with
   the same large learning rate you'd use for random initialization?"
5. "A network pretrained on natural photographs is being adapted to classify medical
   X-ray images — a domain with very different visual statistics. Based on what
   determines transfer learning's effectiveness, would you expect this transfer to work as
   well as adapting the same network to a new but still photograph-based task?"

---

## References

- Yosinski, J. et al. (2014) — *How Transferable Are Features in Deep Neural Networks?*
- Pan, S. & Yang, Q. (2010) — *A Survey on Transfer Learning*
- Deng, J. et al. (2009) — *ImageNet: A Large-Scale Hierarchical Image Database*
- Goodfellow, Bengio & Courville — *Deep Learning* (MIT Press), Chapter 15.2
