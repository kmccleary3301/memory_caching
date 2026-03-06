# PyPI Release Runbook

## Preconditions

- engineering release gate green
- scientific release gate reviewed separately
- package metadata reviewed
- license decision reviewed explicitly

## Commands

```bash
python -m pip install -e ".[dev]"
./scripts/checks/install_smoke.sh
python -m twine check dist/*
uv run python scripts/reports/release_gate_v1.py --mode repo --out outputs/reports/release_gate_repo_v1.json
uv run python scripts/reports/release_gate_v1.py --mode scientific --out outputs/reports/release_gate_scientific_v1.json
```

## Publication policy

- Engineering gate green is required for package publication.
- Scientific gate green is required only for scientific evidence claims, not for package publication.
- A green scientific gate is still not the same as full paper parity.
