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

## Implementation contract

- level `0` is always the self-token contribution
- older context is represented by an inclusive-prefix logarithmic/Fenwick decomposition
- recurrent and chunked paths are correctness-first references for that contract
- the chunked path is not the paper's optimized chunk-scan training implementation

## Reference profiling

`scripts/reports/loglinear_reference_profile.py` records small recurrent-vs-chunked
reference timings into `outputs/reports/loglinear_reference_profile.json`.

This is only a reference-implementation characterization artifact. It does not
support any optimized-path or paper-performance parity claim.
