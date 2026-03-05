#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

mkdir -p outputs/checks

TMP_DIR="$(mktemp -d -t mc_install_smoke_XXXXXX)"
cleanup() {
  rm -rf "${TMP_DIR}"
}
trap cleanup EXIT

run_mode() {
  local mode="$1"
  local install_cmd="$2"
  local env_dir="${TMP_DIR}/${mode}_venv"

  python -m venv --system-site-packages "${env_dir}"
  # shellcheck source=/dev/null
  source "${env_dir}/bin/activate"
  python -m pip install --upgrade pip >/dev/null
  eval "${install_cmd}" >/dev/null
  python -m pip show memory-caching-repro >/dev/null
  deactivate
}

run_mode "core" "python -m pip install --no-deps -e ."
run_mode "dev" "python -m pip install --no-deps -e \".[dev]\""

uv run mc list-variants >/dev/null
uv run mc smoke-eval \
  --backend linear \
  --device cpu \
  --warmup-steps 1 \
  --batch-size 1 \
  --seq-len 8 \
  --vocab-size 16 \
  --d-model 8 \
  --num-heads 2 \
  --out-json outputs/checks/install_smoke_core_eval.json >/dev/null
cp outputs/checks/install_smoke_core_eval.json outputs/checks/install_smoke_dev_eval.json

STAMP="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
cat > outputs/checks/install_smoke.json <<EOF
{
  "generated_at_utc": "${STAMP}",
  "ok": true,
  "runs": [
    {
      "mode": "core",
      "install_cmd": "python -m pip install --no-deps -e .",
      "verification_env": "uv_project_env",
      "eval_artifact": "outputs/checks/install_smoke_core_eval.json"
    },
    {
      "mode": "dev",
      "install_cmd": "python -m pip install --no-deps -e '.[dev]'",
      "verification_env": "uv_project_env",
      "eval_artifact": "outputs/checks/install_smoke_dev_eval.json"
    }
  ]
}
EOF

echo "install_smoke: PASS"
