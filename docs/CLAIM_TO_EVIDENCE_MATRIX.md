# Claim to Evidence Matrix

This matrix separates claims that are verifiable from tracked source/tests versus claims that require run-generated artifacts.

## Code-backed claims

| Claim | Evidence type | Location |
|---|---|---|
| Segmentation utilities are deterministic and validated | Unit tests | `tests/test_segmentation.py` |
| MC layer preserves causality and deterministic cache lifecycle | Unit tests | `tests/test_layer.py` |
| RM/GRM/Soup/SSC aggregation paths are implemented and covered | Unit tests | `tests/test_layer.py` |
| Soup fallback is explicit and opt-in for non-mixable backends | Runtime checks + tests | `src/memory_caching/layer.py`, `tests/test_layer.py` |
| Linear backend update/apply/mix mechanics are covered | Unit tests | `tests/test_linear_backend.py` |
| DLA backend update/apply semantics are covered | Unit tests | `tests/test_dla_backend.py` |
| Titans backend update/apply semantics are covered, including paper-vs-gd convention | Unit tests | `tests/test_titans_backend.py` |
| SWLA(c=2) recurrence/update/apply/mix semantics are covered | Unit tests | `tests/test_swla_backend.py`, `tests/test_layer.py` |
| Inner-update scaling invariance across batch/head replication is enforced | Unit tests | `tests/test_dla_backend.py`, `tests/test_titans_backend.py` |
| Differentiable inner-update mode has unrolled temporal gradient-flow checks | Unit tests | `tests/test_dla_backend.py`, `tests/test_titans_backend.py` |
| Backend state/tensor contracts are runtime-enforced | Runtime assertions + tests | `src/memory_caching/contracts.py`, `src/memory_caching/layer.py`, `tests/test_backend_contract_guards.py` |
| Bench CLI emits rule-based adapter warnings and adapter metadata | CLI code + tests | `src/memory_caching/cli.py`, `tests/test_cli.py` |

## Run-generated claims (CI or local scripts)

| Claim | Evidence type | Generation path |
|---|---|---|
| Smoke benchmark trend/parity/stat/checksum reports can be generated reproducibly | Check scripts + report scripts | `scripts/checks/bench_smoke.sh`, `scripts/reports/benchmark_trend.py`, `scripts/reports/parity_dashboard.py`, `scripts/reports/stat_summary.py`, `scripts/reports/artifact_checksums.py` |
| Release gate checks required report bundles and summary artifacts | Gate script | `scripts/reports/release_gate_v1.py` |
| Resume-consistency checks can be generated and validated | Check script + report script | `scripts/checks/resume_consistency.sh`, `scripts/reports/checkpoint_parity.py` |
| Full-dataset execution path is scripted with subset safety guardrails | Execution script | `scripts/checks/paper_scale_execution.sh` |

## Blocked claims

| Blocked claim | Why blocked | Required evidence to unblock |
|---|---|---|
| Full paper-metric parity across all tables/tasks | Log-Linear++ not implemented and paper-scale runs not complete | Implement missing baselines + full paper-scale model-backed eval runs |
| Throughput parity against paper systems | No dedicated normalized perf harness | Hardware-normalized benchmark harness and published runs |
| Exact parity with unpublished author internals | Author reference implementation unavailable | Author release/confirmation or exact official code comparison |
