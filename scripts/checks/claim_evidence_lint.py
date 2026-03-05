from __future__ import annotations

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[2]
CLAIM_MATRIX_PATH = ROOT / "docs/CLAIM_TO_EVIDENCE_MATRIX.md"


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


def main() -> None:
    errors: list[str] = []

    smoke_targets = ROOT / "configs/bench/smoke_targets.yaml"
    if not smoke_targets.exists():
        errors.append("missing required config: configs/bench/smoke_targets.yaml")

    legacy_targets = ROOT / "configs/bench/paper_targets.yaml"
    if legacy_targets.exists():
        errors.append("legacy config must not exist: configs/bench/paper_targets.yaml")

    claim_matrix_text = CLAIM_MATRIX_PATH.read_text()
    section_text = _extract_section(claim_matrix_text, "Code-backed claims")
    paths = re.findall(r"`([^`]+)`", section_text)
    for raw in paths:
        for part in [p.strip() for p in raw.split(",") if p.strip()]:
            if part.startswith("outputs/") or part.startswith("artifacts/"):
                errors.append(
                    f"code-backed claim references generated path (move to run-generated section): {part}"
                )
                continue
            path = ROOT / part
            if not path.exists():
                errors.append(f"missing referenced path in code-backed claims: {part}")

    if errors:
        raise SystemExit("\n".join(errors))

    print("claim_evidence_lint: PASS")


if __name__ == "__main__":
    main()

