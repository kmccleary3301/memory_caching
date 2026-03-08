# Log-Linear Terminology

Date: 2026-03-08

This repository uses two related but distinct names:

## `LogLinearAttention`

This name refers to the original Guo et al. hierarchical log-linear attention
mechanism built around a Fenwick-style multiscale prefix decomposition.

Use this name only for the original mechanism family from the Log-Linear
Attention paper.

## `LogLinearPP`

This name refers to the Memory Caching paper's `Log-Linear++` comparison
baseline:

- Memory Caching wrapper
- `GRM` aggregation
- logarithmic segmentation
- otherwise standard MC-side recurrent backend path

This is a baseline **inside the Memory Caching framework**. It is not the same
thing as the original Guo et al. `LogLinearAttention` mechanism.

## Repository rule

Never use:

- `LogLinear++`
- `Log-Linear++`
- `LogLinearAttention`

interchangeably.

The repository should preserve this distinction in:

- code names
- config names
- documentation
- claim-boundary wording
- benchmark manifests

## Practical implication

If a component lowers to the current Memory Caching wrapper with:

- `aggregation="grm"`
- `segmentation="logarithmic"`

then it should be labeled `LogLinearPP`, not original `LogLinearAttention`.
