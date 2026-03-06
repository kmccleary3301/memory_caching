# Consumer Support Matrix

| Surface | Status | Notes |
|---|---|---|
| Python 3.10 | supported | primary tested path |
| Python 3.11 | supported | primary tested path |
| Python 3.12 | supported | primary tested path |
| CPU install | supported | install smoke covers wheel, sdist, and dev paths |
| CUDA install | supported with local torch selection | install the torch build matching your CUDA stack |
| `.[dev]` extras | supported | includes test/build/release tooling |
| `.[research]` extras | supported | includes YAML/report tooling |
| Benchmark/report scripts | repo tooling only | not part of the stable package API |
