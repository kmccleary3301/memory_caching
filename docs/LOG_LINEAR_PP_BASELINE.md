# LogLinearPP Baseline

Date: 2026-03-08

`LogLinearPP` is the repository name for the Memory Caching paper's
`Log-Linear++` comparison baseline.

## Definition

Within this repository, `LogLinearPP` means:

- the existing Memory Caching wrapper
- `GRM` aggregation
- logarithmic segmentation
- a standard recurrent backend selected explicitly by config

## What it is for

This baseline exists to close the Memory Caching paper's missing-baseline gap in
a way that is:

- explicit,
- reproducible,
- package-visible,
- and impossible to confuse with original `LogLinearAttention`.

## What it is not

`LogLinearPP` is **not**:

- the original Fenwick-state log-linear mechanism from Guo et al.
- a replacement for eventual original `LogLinearAttention` implementation work
- evidence of full paper parity by itself

## Current implementation policy

The repository treats `LogLinearPP` as a constrained preset over the existing
`MemoryCachingLayer`, not as a separate low-level recurrent algorithm.

## Config family

- `configs/train/loglinear_pp_pilot.yaml`
- `configs/train/loglinear_pp_mid.yaml`
- `configs/train/loglinear_pp_target.yaml`

Baseline presence is not baseline evidence. Empirical parity remains blocked
until model-backed benchmark artifacts exist and are interpreted against
explicit targets.
