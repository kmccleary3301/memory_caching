from __future__ import annotations

from pathlib import Path
import subprocess
import sys


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "checks" / "paper_to_code_sync.py"


def _run_sync(project_root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--project-root", str(project_root)],
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
    symbol_cell = "<br>".join(item.get("symbols", []))
    text = "\\n".join([
        "# Paper to Code Mapping",
        "",
        f"paper: {data.get('paper', 'unknown')}",
        f"map_source: {args.map_yaml}",
        f"generated_by: {data.get('generated_by', 'unknown')}",
        f"generated_at_utc: {args.generated_at_utc}",
        "",
        "| Section | Mechanism | Paper Anchor | Code Paths | Symbols | Status |",
        "|---|---|---|---|---|---|",
        f"| {section['name']} | {item['mechanism']} | {item['paper_anchor']} | {code_cell} | {symbol_cell} | {item['status']} |",
        "",
    ])
    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_md).write_text(text)

if __name__ == "__main__":
    main()
"""
    )


def _write_repo_scaffold(root: Path, *, missing_code_path: bool) -> None:
    (root / "docs").mkdir(parents=True, exist_ok=True)
    (root / "src").mkdir(parents=True, exist_ok=True)
    _write_generator(root / "scripts/reports/generate_paper_to_code.py")

    (root / "src/test_mod.py").write_text("class Symbol: pass\n")
    if not missing_code_path:
        (root / "src/existing.py").write_text("x = 1\n")

    code_path = "src/missing.py" if missing_code_path else "src/existing.py"
    map_yaml = f"""paper: "paper"
generated_by: "scripts/reports/generate_paper_to_code.py"
sections:
  - name: "S"
    items:
      - mechanism: "M"
        paper_anchor: "A"
        code:
          - "{code_path}"
        symbols:
          - "test_mod::Symbol"
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
