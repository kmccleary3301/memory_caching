# Paper to Code Mapping

paper: Memory Caching: RNNs with Growing Memory (arXiv:2602.24281v1)
map_source: docs/paper_to_code_map.yaml
generated_by: scripts/reports/generate_paper_to_code.py
generated_at_utc: 2026-03-06T07:30:56.014993+00:00

| Section | Mechanism | Paper Anchor | Code Paths | Symbols / Coverage | Status |
|---|---|---|---|---|---|
| MC wrapper and segmentation | Constant/logarithmic segmentation | Segmentation discussion + binary decomposition example | src/memory_caching/segmentation.py<br>src/memory_caching/layer.py | memory_caching.segmentation::logarithmic_segments<br>memory_caching.layer::MemoryCachingLayer | implemented |
| MC wrapper and segmentation | Checkpoint vs restart segment initialization | Checkpoint vs independent compressor discussion | src/memory_caching/layer.py<br>src/memory_caching/config.py | memory_caching.layer::MemoryCachingLayer<br>memory_caching.config::MCConfig | implemented |
| Aggregation strategies | Residual Memory (RM) | RM equation | src/memory_caching/layer.py<br>tests/test_layer.py<br>tests/test_paper_equations.py | memory_caching.layer::MemoryCachingLayer | implemented |
| Aggregation strategies | Gated Residual Memory (GRM) | GRM equation and gamma weighting | src/memory_caching/layer.py<br>tests/test_layer.py<br>tests/test_paper_equations.py | memory_caching.layer::MemoryCachingLayer | implemented |
| Aggregation strategies | Memory Soup | Soup definition; true state-space mixing on mixable backends, explicit output-mixture fallback otherwise | src/memory_caching/layer.py<br>tests/test_layer.py<br>tests/test_paper_equations.py | memory_caching.layer::MemoryCachingLayer | implemented_with_caveat |
| Aggregation strategies | Sparse Selective Caching (SSC) | SSC top-k selective routing | src/memory_caching/layer.py<br>tests/test_layer.py<br>tests/test_paper_equations.py | memory_caching.layer::MemoryCachingLayer | implemented |
| Backends | Linear memory backend | Unnormalized matrix-memory reference backend for the wrapper; not a full normalized linear-attention baseline and not a paper-parity claim | src/memory_caching/backends/linear.py<br>tests/test_linear_backend.py | memory_caching.backends.linear::LinearMemoryBackend | implemented |
| Backends | DLA backend | DLA-style deep-memory equations with constant scalar coefficients; implementation-faithful but not paper-metric validated | src/memory_caching/backends/dla.py<br>tests/test_dla_backend.py | memory_caching.backends.dla::DLABackend | implemented_with_caveat |
| Backends | Titans backend | Titans-style deep-memory equations with explicit sign convention and constant scalar coefficients; implementation-faithful but not paper-metric validated | src/memory_caching/backends/titans.py<br>tests/test_titans_backend.py<br>tests/test_titans_update_convention.py | memory_caching.backends.titans::TitansBackend | implemented_with_caveat |
| Backends | SWLA c=2 backend | SWLA c=2 recurrence with previous-outer-product carry and constant scalar coefficients; implementation-faithful but not paper-metric validated | src/memory_caching/backends/swla.py<br>tests/test_swla_backend.py<br>tests/test_layer.py | memory_caching.backends.swla::SWLABackend | implemented_with_caveat |
| Backends | Paper-equation analytic invariants | RM/GRM/Soup/SSC formula-level checks | tests/test_paper_equations.py<br>docs/IMPLEMENTATION_STATUS.md | Test-and-doc mapping item; no single runtime module::symbol anchor. | implemented |
| Claims and release discipline | Claim boundary and blocked-claim policy | Public claim discipline (project policy) | docs/CLAIM_BOUNDARY.md<br>docs/CLAIM_TO_EVIDENCE_MATRIX.md<br>scripts/checks/claim_evidence_lint.py | Policy/docs mapping item; symbol linkage is not applicable. | implemented |

symbol_coverage_total_items: 12
symbol_coverage_items_with_symbols: 10
symbol_coverage_items_with_reason: 2
symbol_coverage_items_missing: 0

