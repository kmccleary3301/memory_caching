#!/usr/bin/env bash
set -euo pipefail

STAMP=$(date -u +"%Y%m%dT%H%M%SZ")
WORK_ROOT=$(mktemp -d)
CLONE_DIR="${WORK_ROOT}/memory_caching_clean"
ARCHIVE_DIR="outputs/independent_repro/${STAMP}"

git clone --depth 1 . "${CLONE_DIR}"
cd "${CLONE_DIR}"

uv sync --extra dev
./scripts/checks/phase2.sh
./scripts/checks/bench_smoke.sh
./scripts/checks/pipeline_smoke.sh
./scripts/checks/resume_consistency.sh
uv run python scripts/reports/release_gate_v1.py --mode repo --out outputs/reports/release_gate_repo_v1.json

mkdir -p "${OLDPWD}/${ARCHIVE_DIR}"
cp -r outputs/checks "${OLDPWD}/${ARCHIVE_DIR}/checks"
cp -r outputs/reports "${OLDPWD}/${ARCHIVE_DIR}/reports"

cat > "${OLDPWD}/${ARCHIVE_DIR}/manifest.json" <<EOF
{
  "schema_version": "v1",
  "timestamp_utc": "${STAMP}",
  "source_repo": "$(pwd)",
  "archive_dir": "${ARCHIVE_DIR}",
  "status": "pass"
}
EOF

echo "independent repro pass archived to ${ARCHIVE_DIR}"
