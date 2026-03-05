# Claim to Evidence Matrix

This matrix separates claims that are verifiable from tracked source/tests versus claims that require run-generated artifacts.

## Code-backed claims

| claim | evidence_type | location |
|---|---|---|
| Segmentation utilities are deterministic and validated | unit_test | `tests/test_segmentation.py` |
| MC layer preserves causality and deterministic cache lifecycle | unit_test | `tests/test_layer.py` |
| RM/GRM/Soup/SSC aggregation paths are implemented and covered | unit_test | `tests/test_layer.py` |
| Soup fallback is explicit and opt-in for non-mixable backends | runtime_check_plus_unit_test | `src/memory_caching/layer.py`, `tests/test_layer.py` |
| Linear backend update/apply/mix mechanics are covered | unit_test | `tests/test_linear_backend.py` |
| DLA backend update/apply semantics are covered | unit_test | `tests/test_dla_backend.py` |
| Titans backend update/apply semantics are covered, including paper-vs-gd convention | unit_test | `tests/test_titans_backend.py`, `tests/test_titans_update_convention.py` |
| SWLA(c=2) recurrence/update/apply/mix semantics are covered | unit_test | `tests/test_swla_backend.py`, `tests/test_layer.py` |
| Inner-update scaling invariance across batch/head replication is enforced | unit_test | `tests/test_dla_backend.py`, `tests/test_titans_backend.py` |
| Differentiable inner-update mode has unrolled temporal gradient-flow checks | unit_test | `tests/test_dla_backend.py`, `tests/test_titans_backend.py` |
| Backend state/tensor contracts are runtime-enforced | runtime_check_plus_unit_test | `src/memory_caching/contracts.py`, `src/memory_caching/layer.py`, `tests/test_backend_contract_guards.py` |
| Bench CLI emits rule-based adapter warnings and adapter metadata | cli_code_plus_unit_test | `src/memory_caching/cli.py`, `tests/test_cli.py` |

## Run-generated claims (CI or local scripts)

| claim | evidence_type | location |
|---|---|---|
| Smoke benchmark trend/parity/stat/checksum reports can be generated reproducibly | generated_evidence_by_scripts | `scripts/checks/bench_smoke.sh`, `scripts/reports/benchmark_trend.py`, `scripts/reports/parity_dashboard.py`, `scripts/reports/stat_summary.py`, `scripts/reports/artifact_checksums.py` |
| Release gate checks required report bundles and summary artifacts | generated_evidence_gate | `scripts/reports/release_gate_v1.py` |
| Resume-consistency checks can be generated and validated | generated_evidence_by_scripts | `scripts/checks/resume_consistency.sh`, `scripts/reports/checkpoint_parity.py` |
| Full-dataset execution path is scripted with subset safety guardrails | generated_evidence_by_script | `scripts/checks/paper_scale_execution.sh` |

## Blocked claims

| blocked_claim | why_blocked | required_evidence_to_unblock |
|---|---|---|
| Full paper-metric parity across all tables/tasks | Log-Linear++ not implemented and paper-scale model-backed runs are not complete | Implement missing baselines + full paper-scale model-backed eval runs |
| Throughput parity against paper systems | No dedicated normalized perf harness | Hardware-normalized benchmark harness and published runs |
| Exact parity with unpublished author internals | Author reference implementation unavailable | Author release/confirmation or exact official code comparison |
