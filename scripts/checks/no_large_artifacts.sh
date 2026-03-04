#!/usr/bin/env bash
set -euo pipefail

MAX_BYTES=${MAX_BYTES:-5242880}

forbidden_exts=(
  "pt"
  "pth"
  "ckpt"
  "bin"
  "safetensors"
  "onnx"
  "h5"
  "keras"
  "npy"
  "npz"
)

forbidden_prefixes=(
  "outputs/"
  "docs_tmp/"
  "artifacts/checkpoints/"
)

failures=0

while IFS= read -r -d '' path; do
  for prefix in "${forbidden_prefixes[@]}"; do
    if [[ "${path}" == "${prefix}"* ]]; then
      echo "forbidden tracked path: ${path} (prefix ${prefix})" >&2
      failures=1
    fi
  done

  lower_path=$(printf '%s' "${path}" | tr '[:upper:]' '[:lower:]')
  ext="${lower_path##*.}"
  if [[ "${ext}" != "${lower_path}" ]]; then
    for bad_ext in "${forbidden_exts[@]}"; do
      if [[ "${ext}" == "${bad_ext}" ]]; then
        echo "forbidden tracked extension: ${path} (*.${bad_ext})" >&2
        failures=1
      fi
    done
  fi

  size_bytes=$(wc -c < "${path}")
  if (( size_bytes > MAX_BYTES )); then
    echo "tracked file exceeds MAX_BYTES=${MAX_BYTES}: ${path} (${size_bytes} bytes)" >&2
    failures=1
  fi
done < <(git ls-files -z)

if (( failures != 0 )); then
  exit 1
fi

echo "artifact guard: PASS (MAX_BYTES=${MAX_BYTES})"
