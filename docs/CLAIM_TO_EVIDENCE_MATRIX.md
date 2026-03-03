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
| LongBench/retrieval scaffolds validate configs and produce rows | Unit tests + runner code | `tests/test_bench.py`, `src/memory_caching/bench/runner.py` |
| Benchmark artifacts include manifest, rows, csv, report | Unit tests + code | `tests/test_bench.py`, `src/memory_caching/bench/artifacts.py` |

## Blocked claims

| Blocked claim | Why blocked | Required evidence to unblock |
|---|---|---|
| Full paper-metric parity | LongBench/retrieval still scaffold-level | full dataset-backed runs + parity tables |
| Throughput parity vs paper systems | no throughput benchmark suite yet | reproducible perf harness + hardware metadata |
| Exact unpublished implementation parity | author internals unavailable | paper/code release or direct author confirmation |
