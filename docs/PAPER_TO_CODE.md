# Paper to Code Mapping

paper: Memory Caching: RNNs with Growing Memory (arXiv:2602.24281v1)
map_source: docs/paper_to_code_map.yaml
generated_by: scripts/reports/generate_paper_to_code.py
generated_at_utc: 2026-03-05T01:46:09.439152+00:00

| Section | Mechanism | Paper Anchor | Code Paths | Status |
|---|---|---|---|---|
| MC wrapper and segmentation | Constant/logarithmic segmentation | Segmentation discussion + binary decomposition example | src/memory_caching/segmentation.py<br>src/memory_caching/layer.py | implemented |
| MC wrapper and segmentation | Checkpoint vs restart segment initialization | Checkpoint vs independent compressor discussion | src/memory_caching/layer.py<br>src/memory_caching/config.py | implemented |
| Aggregation strategies | Residual Memory (RM) | RM equation | src/memory_caching/layer.py<br>tests/test_layer.py | implemented |
| Aggregation strategies | Gated Residual Memory (GRM) | GRM equation and gamma weighting | src/memory_caching/layer.py<br>tests/test_layer.py | implemented |
| Aggregation strategies | Memory Soup | Soup definition | src/memory_caching/layer.py<br>tests/test_layer.py | implemented_with_caveat |
| Aggregation strategies | Sparse Selective Caching (SSC) | SSC top-k selective routing | src/memory_caching/layer.py<br>tests/test_layer.py | implemented |
| Backends | Linear memory backend | LA-style memory update | src/memory_caching/backends/linear.py<br>tests/test_linear_backend.py | implemented |
| Backends | DLA backend | DLA equations | src/memory_caching/backends/dla.py<br>tests/test_dla_backend.py | implemented_with_caveat |
| Backends | Titans backend | Titans equations | src/memory_caching/backends/titans.py<br>tests/test_titans_backend.py | implemented_with_caveat |
| Backends | SWLA c=2 backend | SWLA recurrence/retrieval | src/memory_caching/backends/swla.py<br>tests/test_swla_backend.py<br>tests/test_layer.py | implemented_with_caveat |
| Claims and release discipline | Claim boundary and blocked-claim policy | Public claim discipline (project policy) | docs/CLAIM_BOUNDARY.md<br>docs/CLAIM_TO_EVIDENCE_MATRIX.md<br>scripts/checks/claim_evidence_lint.py | implemented |
