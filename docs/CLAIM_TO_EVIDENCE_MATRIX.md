# Claim to Evidence Matrix

## Supported claims

| Claim | Evidence type | Location |
|---|---|---|
| Segmentation utilities are deterministic and validated | Unit tests | `tests/test_segmentation.py` |
| Linear backend update/apply/mix math is correct | Unit tests | `tests/test_linear_backend.py` |
| MC layer is causal and cache-length deterministic | Unit tests | `tests/test_layer.py` |
| SSC edge behavior is stable | Unit tests | `tests/test_layer.py` |
| Soup fallback works without mixable backend | Unit tests | `tests/test_layer.py` |
| DLA backend init/apply/update semantics are wired | Unit tests | `tests/test_dla_backend.py` |
| DLA Soup vs GRM can diverge | Unit tests | `tests/test_dla_backend.py` |
| Titans backend init/apply/update semantics are wired | Unit tests | `tests/test_titans_backend.py` |
| Titans Soup vs GRM can diverge | Unit tests | `tests/test_titans_backend.py` |
| Smoke harness writes stable metric schema | Unit tests + CLI | `tests/test_smoke.py`, `src/memory_caching/smoke.py` |
| NIAH/MQAR harnesses are deterministic | Unit tests | `tests/test_bench.py` |
| Benchmark stack exposes a model-backed adapter interface | Adapter API code | `src/memory_caching/bench/adapters.py` |
| LongBench/retrieval scaffolds validate configs and produce rows | Unit tests + runner code | `tests/test_bench.py`, `src/memory_caching/bench/runner.py` |
| LongBench/retrieval support JSONL dataset-file ingestion path | Unit tests + loader code + sample files | `tests/test_bench.py`, `src/memory_caching/bench/longbench.py`, `src/memory_caching/bench/retrieval.py`, `examples/` |
| Benchmark artifacts include manifest, rows, csv, report | Unit tests + code | `tests/test_bench.py`, `src/memory_caching/bench/artifacts.py` |
| Backend I/O/state contract is runtime-enforced in layer execution | Runtime assertions + contract doc | `src/memory_caching/layer.py`, `src/memory_caching/contracts.py`, `docs/BACKEND_API_CONTRACT.md` |
| Backend contract guard failures are covered by negative tests | Unit tests | `tests/test_backend_contract_guards.py` |
| Smoke reproducibility is deterministic per backend seed | Unit tests | `tests/test_smoke_repro.py` |
| Checkpoint vs restart divergence is validated across all aggregators | Unit tests | `tests/test_phase2_parity_modes.py` |
| Soup fallback equivalence for non-mixable backends is validated | Unit tests | `tests/test_phase2_parity_modes.py` |
| DLA/Titans stopgrad vs differentiable update modes diverge in graph properties | Unit tests | `tests/test_phase2_parity_modes.py` |
| Release-gate automation emits phase summaries and trend reports | Check scripts + report scripts | `scripts/checks/phase2.sh`, `scripts/checks/bench_smoke.sh`, `scripts/checks/pipeline_smoke.sh`, `scripts/reports/benchmark_trend.py`, `scripts/reports/write_phase_summary.py` |
| Release-gate checker blocks incomplete evidence bundles | Gate script + CI workflow | `scripts/reports/release_gate_v1.py`, `.github/workflows/repro_checks.yml` |
| Legacy/non-v1 artifact bundles can be identified and quarantined | Quarantine script + CI dry-run scan | `scripts/reports/quarantine_legacy_artifacts.py`, `.github/workflows/repro_checks.yml` |
| Tokenizer pipeline emits deterministic vocab artifacts and corpus fingerprint | Pipeline scripts + manifests | `scripts/data/train_tokenizer.py`, `artifacts/tokenizer_manifest.json` |
| Data processing emits deterministic tokenized shards with source and split distribution evidence | Pipeline scripts + manifests | `scripts/data/process_data.py`, `artifacts/data_manifest.json` |
| Training pipeline writes actual torch checkpoints with scheduler state, RNG state, and telemetry traces | Pipeline scripts + checkpoint contract | `scripts/train/train_loop.py`, `docs/CHECKPOINT_CONTRACT.md`, `artifacts/train_manifest.json` |
| Resume path can be parity-checked against uninterrupted training | Consistency check + parity report script | `scripts/checks/resume_consistency.sh`, `scripts/reports/checkpoint_parity.py` |
| Phase3 parity/statistics/checksum evidence is generated and gate-enforced | Bench script + report scripts + gate | `scripts/checks/bench_smoke.sh`, `scripts/reports/parity_dashboard.py`, `scripts/reports/stat_summary.py`, `scripts/reports/artifact_checksums.py`, `scripts/reports/release_gate_v1.py` |
| Clean-environment reproducibility pass can be archived and gate-checked | Independent repro script + archived manifest + gate check | `scripts/checks/independent_repro_pass.sh`, `outputs/independent_repro/*/manifest.json`, `scripts/reports/release_gate_v1.py` |

## Blocked claims

| Blocked claim | Why blocked | Required evidence to unblock |
|---|---|---|
| Full paper-metric parity | LongBench/retrieval still scaffold-level | full dataset-backed runs + parity tables |
| Throughput parity vs paper systems | no throughput benchmark suite yet | reproducible perf harness + hardware metadata |
| Exact unpublished implementation parity | author internals unavailable | paper/code release or direct author confirmation |
