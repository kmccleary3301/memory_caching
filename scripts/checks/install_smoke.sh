#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

mkdir -p outputs/checks
rm -rf dist
mkdir -p dist

TMP_DIR="$(mktemp -d -t mc_install_smoke_XXXXXX)"
TORCH_INSTALL_CMD=${TORCH_INSTALL_CMD:-python -m pip install --index-url https://download.pytorch.org/whl/cpu torch}

cleanup() {
  rm -rf "${TMP_DIR}"
}
trap cleanup EXIT

BUILD_ENV="${TMP_DIR}/build_venv"
python -m venv "${BUILD_ENV}"
# shellcheck source=/dev/null
source "${BUILD_ENV}/bin/activate"
python -m pip install --upgrade pip >/dev/null
python -m pip install build twine >/dev/null
python -m build --sdist --wheel --outdir dist >/dev/null
python -m twine check dist/* >/dev/null
deactivate

WHEEL_FILE="$(find dist -maxdepth 1 -name '*.whl' | head -n 1)"
SDIST_FILE="$(find dist -maxdepth 1 -name '*.tar.gz' | head -n 1)"
if [[ -z "${WHEEL_FILE}" || -z "${SDIST_FILE}" ]]; then
  echo "install_smoke: missing built wheel or sdist in dist/" >&2
  exit 1
fi

run_mode() {
  local mode="$1"
  local install_cmd="$2"
  local artifact_kind="$3"
  local env_dir="${TMP_DIR}/${mode}_venv"
  local eval_json="outputs/checks/install_smoke_${mode}_eval.json"

  python -m venv "${env_dir}"
  # shellcheck source=/dev/null
  source "${env_dir}/bin/activate"
  python -m pip install --upgrade pip >/dev/null
  eval "${TORCH_INSTALL_CMD}" >/dev/null
  eval "${install_cmd}" >/dev/null
  python -m pip show memory-caching >/dev/null
  python - <<'PY' >/dev/null
from importlib import resources
import torch
from memory_caching import MCConfig, MemoryCachingLayer, LinearMemoryBackend, __version__

cfg = MCConfig(
    d_model=8,
    num_heads=2,
    backend="linear",
    aggregation="grm",
    segment_size=2,
)
layer = MemoryCachingLayer(config=cfg, backend=LinearMemoryBackend())
x = torch.randn(1, 4, 8)
y = layer(x)
assert tuple(y.shape) == (1, 4, 8)
assert resources.files("memory_caching").joinpath("py.typed").is_file()
print(__version__, layer is not None)
PY
  mc list-variants >/dev/null
  mc smoke-eval \
    --backend linear \
    --device cpu \
    --warmup-steps 1 \
    --batch-size 1 \
    --seq-len 8 \
    --vocab-size 16 \
    --d-model 8 \
    --num-heads 2 \
    --out-json "${eval_json}" >/dev/null
  deactivate
}

run_mode "wheel" "python -m pip install \"${WHEEL_FILE}\"" "wheel"
run_mode "sdist" "python -m pip install \"${SDIST_FILE}\"" "sdist"
run_mode "dev" "python -m pip install -e \".[dev]\"" "editable_source"

STAMP="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
cat > outputs/checks/install_smoke.json <<EOF
{
  "generated_at_utc": "${STAMP}",
  "ok": true,
  "twine_check_ok": true,
  "build_artifacts": {
    "wheel": "${WHEEL_FILE}",
    "sdist": "${SDIST_FILE}"
  },
  "runs": [
    {
      "mode": "wheel",
      "artifact_kind": "wheel",
      "install_cmd": "clean_venv + torch install + python -m pip install dist/*.whl",
      "verification_env": "clean_venv",
      "py_typed_ok": true,
      "import_forward_ok": true,
      "eval_artifact": "outputs/checks/install_smoke_wheel_eval.json"
    },
    {
      "mode": "sdist",
      "artifact_kind": "sdist",
      "install_cmd": "clean_venv + torch install + python -m pip install dist/*.tar.gz",
      "verification_env": "clean_venv",
      "py_typed_ok": true,
      "import_forward_ok": true,
      "eval_artifact": "outputs/checks/install_smoke_sdist_eval.json"
    },
    {
      "mode": "dev",
      "artifact_kind": "editable_source",
      "install_cmd": "clean_venv + torch install + python -m pip install -e '.[dev]'",
      "verification_env": "clean_venv",
      "py_typed_ok": true,
      "import_forward_ok": true,
      "eval_artifact": "outputs/checks/install_smoke_dev_eval.json"
    }
  ]
}
EOF

echo "install_smoke: PASS"
