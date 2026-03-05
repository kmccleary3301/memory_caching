from __future__ import annotations

from pathlib import Path
import subprocess
import sys


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "checks" / "paper_to_code_sync.py"


def _run_sync(project_root: Path, *, schema_only: bool = False) -> subprocess.CompletedProcess[str]:
    cmd = [sys.executable, str(SCRIPT), "--project-root", str(project_root)]
    if schema_only:
        cmd.append("--schema-only")
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )


def _write_generator(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        """from __future__ import annotations
import argparse
from pathlib import Path
import yaml

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--map-yaml", required=True)
    p.add_argument("--out-md", required=True)
    p.add_argument("--generated-at-utc", default="")
    args = p.parse_args()
    data = yaml.safe_load(Path(args.map_yaml).read_text())
    section = data["sections"][0]
    item = section["items"][0]
    code_cell = "<br>".join(item.get("code", []))
    symbols = item.get("symbols", [])
    if symbols:
        coverage = "<br>".join(symbols)
        with_symbols = 1
        with_reason = 0
    else:
        coverage = item.get("symbol_coverage_reason", "")
        with_symbols = 0
        with_reason = 1 if coverage else 0
    missing = 1 - with_symbols - with_reason
    text = "\\n".join([
        "# Paper to Code Mapping",
        "",
        f"paper: {data.get('paper', 'unknown')}",
        f"map_source: {args.map_yaml}",
        f"generated_by: {data.get('generated_by', 'unknown')}",
        f"generated_at_utc: {args.generated_at_utc}",
        "",
        "| Section | Mechanism | Paper Anchor | Code Paths | Symbols / Coverage | Status |",
        "|---|---|---|---|---|---|",
        f"| {section['name']} | {item['mechanism']} | {item['paper_anchor']} | {code_cell} | {coverage} | {item['status']} |",
        "",
        "symbol_coverage_total_items: 1",
        f"symbol_coverage_items_with_symbols: {with_symbols}",
        f"symbol_coverage_items_with_reason: {with_reason}",
        f"symbol_coverage_items_missing: {missing}",
        "",
    ])
    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_md).write_text(text)

if __name__ == "__main__":
    main()
"""
    )


def _write_repo_scaffold(
    root: Path,
    *,
    missing_code_path: bool = False,
    symbols: list[str] | None = None,
    symbol_coverage_reason: str = "",
) -> None:
    (root / "docs").mkdir(parents=True, exist_ok=True)
    (root / "src").mkdir(parents=True, exist_ok=True)
    _write_generator(root / "scripts/reports/generate_paper_to_code.py")

    (root / "src/test_mod.py").write_text("class Symbol: pass\n")
    if not missing_code_path:
        (root / "src/existing.py").write_text("x = 1\n")

    code_path = "src/missing.py" if missing_code_path else "src/existing.py"
    symbols = symbols if symbols is not None else ["test_mod::Symbol"]
    symbol_lines = ""
    if len(symbols) > 0:
        symbol_lines = "\n".join([f'        symbols:\n' + "\n".join([f'          - "{s}"' for s in symbols])])
    reason_line = ""
    if symbol_coverage_reason:
        reason_line = f'\n        symbol_coverage_reason: "{symbol_coverage_reason}"'

    map_yaml = f"""paper: "paper"
generated_by: "scripts/reports/generate_paper_to_code.py"
sections:
  - name: "S"
    items:
      - mechanism: "M"
        paper_anchor: "A"
        code:
          - "{code_path}"{reason_line}
{symbol_lines}
        status: "implemented"
"""
    (root / "docs/paper_to_code_map.yaml").write_text(map_yaml)

    generated_at = "2026-03-05T00:00:00+00:00"
    subprocess.run(
        [
            sys.executable,
            "scripts/reports/generate_paper_to_code.py",
            "--map-yaml",
            "docs/paper_to_code_map.yaml",
            "--out-md",
            "docs/PAPER_TO_CODE.md",
            "--generated-at-utc",
            generated_at,
        ],
        check=True,
        cwd=str(root),
    )


def test_paper_to_code_sync_passes_when_map_and_doc_match(tmp_path: Path) -> None:
    _write_repo_scaffold(tmp_path, missing_code_path=False)
    proc = _run_sync(tmp_path)
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "paper_to_code_sync: PASS" in proc.stdout


def test_paper_to_code_sync_fails_on_missing_mapped_path(tmp_path: Path) -> None:
    _write_repo_scaffold(tmp_path, missing_code_path=True)
    proc = _run_sync(tmp_path)
    assert proc.returncode != 0
    out = (proc.stderr or "") + (proc.stdout or "")
    assert "map path missing for mechanism" in out


def test_paper_to_code_sync_fails_on_malformed_symbol_spec(tmp_path: Path) -> None:
    _write_repo_scaffold(tmp_path, symbols=["badformat"])
    proc = _run_sync(tmp_path, schema_only=True)
    assert proc.returncode != 0
    out = (proc.stderr or "") + (proc.stdout or "")
    assert "invalid symbol spec" in out


def test_paper_to_code_sync_fails_on_missing_module(tmp_path: Path) -> None:
    _write_repo_scaffold(tmp_path, symbols=["no_such_module::Symbol"])
    proc = _run_sync(tmp_path, schema_only=True)
    assert proc.returncode != 0
    out = (proc.stderr or "") + (proc.stdout or "")
    assert "failed to import module" in out


def test_paper_to_code_sync_fails_on_missing_symbol(tmp_path: Path) -> None:
    _write_repo_scaffold(tmp_path, symbols=["test_mod::Missing"])
    proc = _run_sync(tmp_path, schema_only=True)
    assert proc.returncode != 0
    out = (proc.stderr or "") + (proc.stdout or "")
    assert "missing symbol" in out


def test_paper_to_code_sync_generation_is_reproducible_with_fixed_timestamp(tmp_path: Path) -> None:
    _write_repo_scaffold(tmp_path, symbols=["test_mod::Symbol"])
    first = (tmp_path / "docs/PAPER_TO_CODE.md").read_text()

    subprocess.run(
        [
            sys.executable,
            "scripts/reports/generate_paper_to_code.py",
            "--map-yaml",
            "docs/paper_to_code_map.yaml",
            "--out-md",
            "docs/PAPER_TO_CODE.md",
            "--generated-at-utc",
            "2026-03-05T00:00:00+00:00",
        ],
        check=True,
        cwd=str(tmp_path),
    )
    second = (tmp_path / "docs/PAPER_TO_CODE.md").read_text()
    assert first == second
