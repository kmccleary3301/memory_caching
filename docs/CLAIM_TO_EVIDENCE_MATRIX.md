# Claim to Evidence Matrix

| Claim | Evidence type | Location |
|---|---|---|
| Segmentation utilities are deterministic and validated | Unit tests | `tests/test_segmentation.py` |
| Linear backend update/apply/mix math is correct | Unit tests | `tests/test_linear_backend.py` |
| MC layer is causal and cache-length deterministic | Unit tests | `tests/test_layer.py` |
| SSC edge behavior is stable | Unit tests | `tests/test_layer.py` |
| Soup fallback works without mixable backend | Unit tests | `tests/test_layer.py` |
| DLA backend init/apply/update semantics are wired | Unit tests | `tests/test_dla_backend.py` |
| DLA Soup vs GRM can diverge | Unit tests | `tests/test_dla_backend.py` |
| Smoke harness writes stable metric schema | Unit tests + CLI | `tests/test_smoke.py`, `src/memory_caching/smoke.py` |
| NIAH and MQAR harnesses are deterministic | Unit tests | `tests/test_bench.py` |
| Benchmark artifacts include schema-versioned manifest | Unit tests + code | `tests/test_bench.py`, `src/memory_caching/bench/artifacts.py` |
