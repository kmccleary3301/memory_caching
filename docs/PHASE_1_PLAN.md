# Phase 1 Plan: MC Core + Linear Memory Backend

## Objective

Deliver a mechanism-faithful implementation of Memory Caching core mechanics with a linear-memory backend, plus guardrail tests for causality and segmentation correctness.

## Deliverables

1. Core data structures
- `SegmentCache` state container
- Memory backend interface (`init_state`, `update`, `apply`)

2. Segmentation
- Constant-size segmentation
- Logarithmic segmentation utility

3. Aggregators
- Residual Memory
- GRM (context-aware gating)
- SSC (top-k router)
- Memory Soup API stub (fully operational after deep-memory backend lands)

4. Baseline linear backend
- Matrix-valued linear memory update and retrieval
- Online + cached retrieval composition through MC wrapper

5. Minimal training/eval scaffolding
- Config objects for model/segmentation/aggregator
- Placeholder CLI entries for smoke workflow orchestration

## Acceptance criteria

1. Causal retrieval path uses only past/current token information.
2. Constant segmentation and log segmentation produce deterministic boundaries.
3. GRM gating uses token-dependent scores against segment context summaries.
4. SSC router selects top-k cached segments per token.
5. Linear-memory equivalence checks are codified for future tests:
- Residual as ungated sum
- Soup equivalence caveat for linear case documented

## Non-goals in Phase 1

- Paper-scale training runs
- Full benchmark parity
- Deep-memory training optimization (DLA/Titans)
