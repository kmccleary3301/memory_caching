# Sign-off Checklist

Scientific gate green is not the same as paper reproduction complete.

## Mechanism complete

- [ ] Core modules implemented
- [ ] Unit tests green
- [ ] Smoke commands green
- [ ] Evidence docs updated
- [ ] `outputs/checks/phase2_summary.json` present and `ok=true`
- [ ] `outputs/checks/phase3_summary.json` present and `ok=true`
- [ ] `outputs/checks/phase4_summary.json` present and `ok=true`

## Parity complete

- [ ] Benchmark suites complete and repeatable
- [ ] Data/training recipe parity achieved
- [ ] Engineering release gate v1 green
- [ ] Scientific release gate v1 green
- [ ] Claim boundary allows parity statements
- [ ] `outputs/reports/phase3_benchmark_trend.json` contains niah/mqar/longbench/retrieval
- [ ] `outputs/reports/release_gate_repo_v1.json` has `ok=true`
- [ ] `outputs/reports/release_gate_scientific_v1.json` has `ok=true`
- [ ] Missing baselines such as `Log-Linear++` are implemented or explicitly called out as the remaining parity blocker
