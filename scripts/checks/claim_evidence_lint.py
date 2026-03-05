from __future__ import annotations

import argparse
from pathlib import Path
import re


def _extract_section(text: str, heading: str) -> str:
    marker = f"## {heading}"
    start = text.find(marker)
    if start < 0:
        raise SystemExit(f"missing section heading: {marker}")
    start = text.find("\n", start)
    if start < 0:
        return ""
    remaining = text[start + 1 :]
    end = remaining.find("\n## ")
    if end < 0:
        return remaining
    return remaining[:end]


def _normalize_header(name: str) -> str:
    return name.strip().lower().replace(" ", "_")


def _is_separator_row(line: str) -> bool:
    compact = line.replace("|", "").replace(":", "").replace("-", "").strip()
    return compact == ""


def _parse_markdown_table(section: str) -> tuple[list[str], list[list[str]]]:
    lines = [line.strip() for line in section.splitlines() if line.strip().startswith("|")]
    if len(lines) < 2:
        raise SystemExit("expected markdown table with header + separator + rows")
    header = [c.strip() for c in lines[0].strip("|").split("|")]
    rows: list[list[str]] = []
    for line in lines[1:]:
        if _is_separator_row(line):
            continue
        rows.append([c.strip() for c in line.strip("|").split("|")])
    return header, rows


def _extract_paths(cell: str) -> list[str]:
    matches = re.findall(r"`([^`]+)`", cell)
    if matches:
        return [m.strip() for m in matches if m.strip()]
    return [p.strip() for p in cell.split(",") if p.strip()]


def _lint_code_backed_claims(root: Path, section_text: str, errors: list[str]) -> None:
    header, rows = _parse_markdown_table(section_text)
    normalized = [_normalize_header(h) for h in header]
    required = {"claim", "evidence_type", "location"}
    if not required.issubset(set(normalized)):
        errors.append(
            "Code-backed claims table header must include: claim, evidence_type, location"
        )
        return

    location_idx = normalized.index("location")
    for row in rows:
        if location_idx >= len(row):
            errors.append("Code-backed claims row missing location column")
            continue
        for part in _extract_paths(row[location_idx]):
            if part.startswith("outputs/") or part.startswith("artifacts/"):
                errors.append(
                    f"code-backed claim references generated path (move to run-generated section): {part}"
                )
                continue
            path = root / part
            if not path.exists():
                errors.append(f"missing referenced path in code-backed claims: {part}")


def _lint_run_generated_claims(root: Path, section_text: str, errors: list[str]) -> None:
    header, rows = _parse_markdown_table(section_text)
    normalized = [_normalize_header(h) for h in header]
    required = {"claim", "evidence_type", "location"}
    if not required.issubset(set(normalized)):
        errors.append(
            "Run-generated claims table header must include: claim, evidence_type, location"
        )
        return

    evidence_idx = normalized.index("evidence_type")
    location_idx = normalized.index("location")
    for row in rows:
        if evidence_idx >= len(row) or location_idx >= len(row):
            errors.append("Run-generated claims row missing required columns")
            continue

        evidence_type = row[evidence_idx].strip().lower()
        if "generated" not in evidence_type:
            errors.append(
                "run-generated claim row evidence_type must include 'generated'"
            )

        for part in _extract_paths(row[location_idx]):
            if part.startswith("outputs/") or part.startswith("artifacts/"):
                continue
            path = root / part
            if not path.exists():
                errors.append(f"missing referenced path in run-generated claims: {part}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    claim_matrix_path = root / "docs/CLAIM_TO_EVIDENCE_MATRIX.md"

    errors: list[str] = []

    smoke_targets = root / "configs/bench/smoke_targets.yaml"
    if not smoke_targets.exists():
        errors.append("missing required config: configs/bench/smoke_targets.yaml")

    legacy_targets = root / "configs/bench/paper_targets.yaml"
    if legacy_targets.exists():
        errors.append("legacy config must not exist: configs/bench/paper_targets.yaml")

    claim_matrix_text = claim_matrix_path.read_text()
    code_backed = _extract_section(claim_matrix_text, "Code-backed claims")
    run_generated = _extract_section(claim_matrix_text, "Run-generated claims (CI or local scripts)")
    _lint_code_backed_claims(root, code_backed, errors)
    _lint_run_generated_claims(root, run_generated, errors)

    if errors:
        raise SystemExit("\n".join(errors))

    print("claim_evidence_lint: PASS")


if __name__ == "__main__":
    main()
