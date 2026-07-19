---
id: cnn_pooling_padding
title: "CNN Pooling, Padding, and Stride"
topic_group: "advanced"
prerequisites: ["cnn_fundamentals"]
difficulty: intermediate
chunk_version: 1
tags: ["pooling", "padding", "stride", "max-pooling", "average-pooling", "cnn", "spatial-dimensions"]
---

# CNN Pooling, Padding, and Stride

## Overview

Beyond the convolution operation and parameter sharing covered in `cnn_fundamentals.md`,
three additional mechanisms shape how information flows through a CNN and how spatial
dimensions change from layer to layer: **padding** (controlling output size and border
information), **stride** (controlling how much the filter moves between applications),
and **pooling** (downsampling feature maps to reduce spatial size while retaining
important information).

---

## Chunk: padding

**ID**: `cnn_pooling_padding#padding`

Without padding, a convolution operation shrinks the spatial dimensions of its output
relative to its input, since a filter can't be centered on border pixels without
extending past the edge of the image.

For an input of size N×N convolved with a filter of size F×F (stride 1, no padding):
```
output size = N - F + 1
```

**Two common padding conventions:**

**"Valid" padding (no padding):** The filter only slides to positions where it fits
entirely within the input. Output shrinks as described above. Border pixels are used in
fewer filter applications than central pixels, so information near the edges of the image
gets relatively less representation in the output.

**"Same" padding:** Zeros are added around the border of the input so that the output
spatial size matches the input spatial size (for stride 1). The amount of padding needed:
```
P = (F - 1) / 2        (for odd filter sizes, stride 1)
```

**Why padding matters beyond just size control:** Without padding, repeated convolutional
layers in a deep CNN would rapidly shrink the spatial dimensions, limiting how deep the
network can be before running out of spatial area entirely, and would also progressively
under-represent border information across many layers.

---

## Chunk: stride

**ID**: `cnn_pooling_padding#stride`

**Stride** controls how many pixels the filter moves between each application. A stride
of 1 moves the filter one pixel at a time (maximum overlap between adjacent filter
positions); a stride of 2 skips every other position.

For an input of size N×N, filter size F×F, padding P, and stride S:
```
output size = ⌊(N + 2P - F) / S⌋ + 1
```

**Effect of stride > 1:** Increases the "step size" of the filter, producing a smaller
output feature map directly from the convolution itself — this can serve a similar
downsampling role to pooling (covered next), and some modern architectures use strided
convolutions instead of separate pooling layers for this reason.

**Trade-off:** Larger stride means fewer filter applications, so less computation, but
also fewer overlapping views of each region of the input — potentially missing
fine-grained local patterns that a stride-1 convolution would capture.

---

## Chunk: max-pooling

**ID**: `cnn_pooling_padding#max-pooling`

**Max pooling** downsamples a feature map by sliding a window (commonly 2×2) across it and
keeping only the maximum value within each window position:

```
Input 4×4 feature map, 2×2 max pooling, stride 2:

[1  3  2  4]        [3  4]
[5  6  1  2]   →     [6  8]
[0  1  8  7]
[2  4  3  6]
```

**Why max, specifically:** Max pooling retains the strongest activation in each local
region — the intuition being that a strong activation indicates the presence of whatever
feature that filter was detecting (an edge, a texture, etc.), and the *exact* pixel
position of that activation within the small pooling window matters less than the fact
that it was detected somewhere in that region.

**Effect on parameter sharing and translation invariance:** Because max pooling reports
only "was this feature present in this region," small translations of the input (shifting
the image by a few pixels) tend to produce the same or very similar pooled output — this
adds a degree of **translation invariance** on top of the translation *equivariance*
already provided by parameter sharing in the convolution itself (see
`cnn_fundamentals#parameter-sharing`).

---

## Chunk: average-pooling

**ID**: `cnn_pooling_padding#average-pooling`

**Average pooling** works identically to max pooling in terms of sliding a window across
the feature map, but takes the mean of all values in the window instead of the maximum:

```
Input 4×4 feature map, 2×2 average pooling, stride 2:

[1  3  2  4]        [3.75  2.25]
[5  6  1  2]   →     [1.75  6.0 ]
[0  1  8  7]
[2  4  3  6]
```

**Max vs. average pooling:** Max pooling tends to preserve the sharpest, most salient
features (good for detecting whether a specific pattern is present anywhere in a region).
Average pooling produces a smoother summary that incorporates every value in the region,
which can retain more overall context but may dilute a single strong, spatially localized
signal by averaging it with weaker neighboring values.

**Global Average Pooling (GAP):** A special case where the pooling window covers the
*entire* remaining spatial extent of a feature map, collapsing each channel to a single
number. This is commonly used just before the final classification layer in modern CNNs,
replacing large fully-connected layers and substantially reducing the number of parameters
(directly relevant to the overfitting discussion in
`overfitting_regularization#what-is-overfitting`, since fewer parameters relative to
training data reduces overfitting risk).

---

## Chunk: why-pooling-is-used

**ID**: `cnn_pooling_padding#why-pooling-is-used`

Pooling layers serve several purposes simultaneously in a CNN:

1. **Dimensionality reduction:** Reduces the spatial size of feature maps, which reduces
   the computation and number of parameters needed in subsequent layers.
2. **Translation invariance:** As discussed above, makes the network's output less
   sensitive to small shifts in where a feature appears within its local region.
3. **Expanding the effective receptive field:** After pooling, each unit in the next
   convolutional layer effectively "sees" a larger region of the original input, since it's
   now operating on a downsampled representation — this lets deeper layers capture larger-
   scale, more abstract patterns without needing proportionally larger filters.

**Modern trend:** Some newer architectures reduce or eliminate explicit pooling layers in
favor of strided convolutions (see `cnn_pooling_padding#stride`), letting the network learn
the downsampling operation itself rather than using a fixed, non-learnable max or average
rule.

---

## Socratic Question Seeds

1. "If a convolution with no padding always shrinks the spatial size of its output, what
   would happen to a very deep CNN's feature maps after 20 or 30 convolutional layers if
   none of them used padding?"

2. "Max pooling keeps only the largest value in each window and discards the rest. What
   information is being thrown away by doing this, and why might that be an acceptable
   trade-off for a feature detector that's just asking 'was this pattern present here?'"

3. "If you shifted an input image two pixels to the right, how would a max-pooled feature
   map's output be affected differently compared to the raw, un-pooled convolutional
   output?"

4. "A stride-2 convolution and a stride-1 convolution followed by 2×2 max pooling both
   reduce spatial dimensions. What's different about *how* each one decides which
   information to keep versus discard?"

5. "Global Average Pooling collapses an entire feature map down to one number per channel
   before the final classification layer, instead of using a large fully-connected layer.
   How might this relate to the overfitting risks discussed earlier in this curriculum?"

---

## References

- LeCun, Y. et al. (1998) — *Gradient-Based Learning Applied to Document Recognition*
- Krizhevsky, Sutskever & Hinton (2012) — *ImageNet Classification with Deep Convolutional
  Neural Networks* (AlexNet)
- Lin, M., Chen, Q. & Yan, S. (2013) — *Network In Network* (Global Average Pooling)
- Goodfellow, Bengio & Courville — *Deep Learning* (MIT Press), Chapter 9.3
