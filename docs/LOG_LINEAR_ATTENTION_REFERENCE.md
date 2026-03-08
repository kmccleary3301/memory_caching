# LogLinearAttention Reference Path

Date: 2026-03-08

This document describes the repository's current original
`LogLinearAttention` implementation scope.

## What exists now

- Fenwick decomposition helpers
- dense oracle
- recurrent correctness-first reference implementation
- correctness-first chunked executor
- tiny-model integration for:
  - `tiny_loglinear_ref_lm`
  - `tiny_loglinear_chunked_lm`

## What this is for

This path exists to support:

- equation-level correctness work
- small-scale model-backed benchmark integration
- future parity-oriented iteration

## What this is not

This path is not yet:

- the paper's optimized chunk-scan training implementation
- a Triton/kernel parity claim
- a full empirical reproduction of the original paper

## Current model-family names

- `tiny_loglinear_ref_lm`
- `tiny_loglinear_chunked_lm`

These names are intended for repository training/eval pipelines and checkpoint
manifests.
