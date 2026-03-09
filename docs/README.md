# Documentation Home

<div align="center">

<a href="../README.md"><img alt="Project" src="https://img.shields.io/badge/project-memory--caching-2088FF"></a>
<a href="https://pypi.org/project/memory-caching/"><img alt="PyPI" src="https://img.shields.io/pypi/v/memory-caching?logo=pypi&color=F4B400"></a>
<a href="https://arxiv.org/abs/2602.24281"><img alt="Paper" src="https://img.shields.io/badge/paper-arXiv%3A2602.24281-B31B1B?logo=arxiv"></a>
<a href="./PAPER_PARITY_BLOCKERS.md"><img alt="Parity Status" src="https://img.shields.io/badge/paper%20parity-blocked-important"></a>

</div>

This directory contains the repository's operating manual: implementation scope,
claim boundaries, public API, release path, and the documents needed to judge
what this reproduction does and does not prove.

---

## If You Read 5 Docs

1. [reproduction_report.md](reproduction_report.md)
2. [PUBLIC_API.md](PUBLIC_API.md)
3. [CLAIM_TO_EVIDENCE_MATRIX.md](CLAIM_TO_EVIDENCE_MATRIX.md)
4. [PAPER_PARITY_BLOCKERS.md](PAPER_PARITY_BLOCKERS.md)
5. [PYPI_RELEASE_RUNBOOK.md](PYPI_RELEASE_RUNBOOK.md)

---

## Documentation Map

| Topic | Link | Why it exists |
|---|---|---|
| Project status | [reproduction_report.md](reproduction_report.md) | Current reproduction status and evidence framing |
| Progress | [PROGRESS_LEDGER.md](PROGRESS_LEDGER.md) | Weighted plan status and latest milestone |
| Public package surface | [PUBLIC_API.md](PUBLIC_API.md) | Stable runtime API and non-public repo tooling |
| Log-linear terminology | [LOG_LINEAR_TERMINOLOGY.md](LOG_LINEAR_TERMINOLOGY.md) | Naming and scope boundary for `LogLinearPP` vs original `LogLinearAttention` |
| LogLinearPP baseline | [LOG_LINEAR_PP_BASELINE.md](LOG_LINEAR_PP_BASELINE.md) | Meaning of the MC-paper baseline preset |
| LogLinearAttention reference | [LOG_LINEAR_ATTENTION_REFERENCE.md](LOG_LINEAR_ATTENTION_REFERENCE.md) | Original mechanism reference-path status |
| Log-linear pilot runbook | [LOG_LINEAR_PILOT_RUNBOOK.md](LOG_LINEAR_PILOT_RUNBOOK.md) | Pilot-scale model-backed execution path for the log-linear families |
| Architecture | [ARCHITECTURE.md](ARCHITECTURE.md) | Layer structure, backend flow, report pipeline |
| Claim boundaries | [CLAIM_BOUNDARY.md](CLAIM_BOUNDARY.md) | What the repo explicitly does not claim |
| Claim-to-evidence map | [CLAIM_TO_EVIDENCE_MATRIX.md](CLAIM_TO_EVIDENCE_MATRIX.md) | Exact support for each claim class |
| Paper mapping | [PAPER_TO_CODE.md](PAPER_TO_CODE.md) | Mechanism mapping from paper to code |
| Implementation scope | [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md) | Implemented vs blocked features |
| Paper parity blockers | [PAPER_PARITY_BLOCKERS.md](PAPER_PARITY_BLOCKERS.md) | Remaining blockers to literal parity |
| Package release path | [PYPI_RELEASE_RUNBOOK.md](PYPI_RELEASE_RUNBOOK.md) | Build, smoke, preflight, and publish steps |
| Consumer support | [CONSUMER_SUPPORT_MATRIX.md](CONSUMER_SUPPORT_MATRIX.md) | Supported environments and expectations |
| Contributor start | [CONTRIBUTOR_ONBOARDING.md](CONTRIBUTOR_ONBOARDING.md) | New contributor path |
| Contributor dry run | [CONTRIBUTOR_DRY_RUN.md](CONTRIBUTOR_DRY_RUN.md) | Time-boxed onboarding evidence |

---

## By Use Case

### I want to use the package

- [PUBLIC_API.md](PUBLIC_API.md)
- [CONSUMER_SUPPORT_MATRIX.md](CONSUMER_SUPPORT_MATRIX.md)
- [PYPI_RELEASE_RUNBOOK.md](PYPI_RELEASE_RUNBOOK.md)

### I want to understand the reproduction

- [reproduction_report.md](reproduction_report.md)
- [CLAIM_TO_EVIDENCE_MATRIX.md](CLAIM_TO_EVIDENCE_MATRIX.md)
- [PAPER_TO_CODE.md](PAPER_TO_CODE.md)
- [PAPER_PARITY_BLOCKERS.md](PAPER_PARITY_BLOCKERS.md)
- [LOG_LINEAR_TERMINOLOGY.md](LOG_LINEAR_TERMINOLOGY.md)
- [LOG_LINEAR_PP_BASELINE.md](LOG_LINEAR_PP_BASELINE.md)
- [LOG_LINEAR_ATTENTION_REFERENCE.md](LOG_LINEAR_ATTENTION_REFERENCE.md)

### I want to contribute safely

- [CONTRIBUTING.md](CONTRIBUTING.md)
- [CONTRIBUTOR_ONBOARDING.md](CONTRIBUTOR_ONBOARDING.md)
- [ARCHITECTURE.md](ARCHITECTURE.md)

### I want to run gates or reports

- [RELEASE_GATE_CHECKLIST_V1.md](RELEASE_GATE_CHECKLIST_V1.md)
- [BENCHMARK_COMMAND_MATRIX.md](BENCHMARK_COMMAND_MATRIX.md)
- [BENCHMARK_SWEEP_RUNBOOK.md](BENCHMARK_SWEEP_RUNBOOK.md)

---

## Guiding Reading Order

The repository has three distinct layers of meaning:

1. `engineering scaffold`
   - packaging, tests, installability, report generation, artifact plumbing
2. `scientific evidence`
   - model-backed runs, truthful manifests, non-smoke targets
3. `paper parity`
   - the much stricter claim that the paper's reported baseline coverage and
     result surface are actually reproduced

Most confusion disappears if these are kept separate while reading the docs.

---

## Notes

- `docs_tmp/` is intentionally excluded from the package/documentation surface.
- `outputs/` contains generated artifacts and is not part of the stable package
  API.
- Historical docs remain in place where they carry audit value, but the files
  linked above are the current entrypoints.
