# Claim to Evidence Matrix

This matrix separates mechanism claims from engineering-scaffold claims and blocked scientific claims.

No claim in this document should use `paper-scale`, `parity`, or `full-dataset` as scientific evidence unless the backing artifact records:

- `adapter_type="model_backed"`
- non-smoke targets
- a model family that actually uses Memory Caching

## Code-backed claims

| claim | evidence_type | location |
|---|---|---|
| Segmentation utilities are deterministic and validated | unit_test | `tests/test_segmentation.py` |
| MC layer preserves causality and deterministic cache lifecycle | unit_test | `tests/test_layer.py` |
| RM/GRM/Soup/SSC aggregation paths are implemented and covered | unit_test | `tests/test_layer.py` |
| Soup fallback is explicit and opt-in for non-mixable backends | runtime_check_plus_unit_test | `src/memory_caching/layer.py`, `tests/test_layer.py` |
| Linear backend update/apply/mix mechanics are covered for the repository's unnormalized matrix-memory backend | unit_test | `tests/test_linear_backend.py` |
| DLA backend update/apply semantics are covered | unit_test | `tests/test_dla_backend.py` |
| Titans backend update/apply semantics are covered, including paper-vs-gd convention | unit_test | `tests/test_titans_backend.py`, `tests/test_titans_update_convention.py` |
| SWLA(c=2) Eq. (28)-style recurrence/update/apply/mix semantics are covered | unit_test | `tests/test_swla_backend.py`, `tests/test_layer.py` |
| Inner-update scaling invariance across batch/head replication is enforced | unit_test | `tests/test_dla_backend.py`, `tests/test_titans_backend.py` |
| Differentiable inner-update mode has unrolled temporal gradient-flow checks | unit_test | `tests/test_dla_backend.py`, `tests/test_titans_backend.py` |
| Backend state/tensor contracts are runtime-enforced | runtime_check_plus_unit_test | `src/memory_caching/contracts.py`, `src/memory_caching/layer.py`, `tests/test_backend_contract_guards.py` |
| Bench CLI emits rule-based adapter warnings and adapter metadata | cli_code_plus_unit_test | `src/memory_caching/cli.py`, `tests/test_cli.py` |

## Run-generated claims (CI or local scripts)

| claim | evidence_type | location |
|---|---|---|
| Smoke benchmark trend/parity/stat/checksum reports can be generated reproducibly | generated_evidence_by_scripts | `scripts/checks/bench_smoke.sh`, `scripts/reports/benchmark_trend.py`, `scripts/reports/parity_dashboard.py`, `scripts/reports/stat_summary.py`, `scripts/reports/artifact_checksums.py` |
| Repository release gate checks required report bundles, lint state, and install-smoke contract | generated_evidence_gate | `scripts/reports/release_gate_v1.py` |
| Resume-consistency checks can be generated and validated | generated_evidence_by_scripts | `scripts/checks/resume_consistency.sh`, `scripts/reports/checkpoint_parity.py` |
| Full-dataset scaffold execution path is scripted with subset safety guardrails | generated_evidence_by_script | `scripts/checks/paper_scale_execution.sh` |
| Scientific train manifests are validated for model family, MC truthfulness, tokenizer metadata, config path, and training-data provenance | generated_evidence_by_scripts | `src/memory_caching/scientific_manifest.py`, `scripts/checks/scientific_manifest_lint.py`, `artifacts/train_manifest.json` |
| Scientific benchmark manifests require `adapter_type=model_backed` and explicit `model_info` metadata | generated_evidence_by_scripts | `src/memory_caching/scientific_manifest.py`, `scripts/checks/scientific_manifest_lint.py`, `outputs/benchmarks/full_dataset/*/manifest.json` |
| Scientific parity dashboards require non-smoke targets | generated_evidence_by_scripts | `configs/bench/scientific_targets.yaml`, `scripts/reports/parity_dashboard.py`, `outputs/reports/full_dataset_parity_dashboard.json` |
| Scientific release eligibility requires both manifest lint and scientific gate success | generated_evidence_gate | `scripts/checks/scientific_manifest_lint.py`, `scripts/reports/release_gate_v1.py`, `outputs/reports/release_gate_scientific_v1.json` |

## Blocked scientific claims

| blocked_claim | why_blocked | required_evidence_to_unblock |
|---|---|---|
| Full paper-metric parity across all tables/tasks | Log-Linear++ not implemented and paper-scale model-backed runs are not complete | Implement missing baselines + full paper-scale model-backed eval runs |
| Throughput parity against paper systems | No dedicated normalized perf harness | Hardware-normalized benchmark harness and published runs |
| Exact parity with unpublished author internals | Author reference implementation unavailable | Author release/confirmation or exact official code comparison |
| Scientific release status for current scaffold outputs | Current benchmark evidence is dominated by rule-based adapters and smoke targets | Model-backed metrics, non-smoke targets, and Memory-Caching model manifests |
| Table-level parity claims without missing paper baselines | Missing baselines such as `Log-Linear++` remain open | Implement and evaluate the missing baseline set |
