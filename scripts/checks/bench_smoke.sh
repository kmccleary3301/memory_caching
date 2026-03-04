#!/usr/bin/env bash
set -euo pipefail

BENCH_ROOT="outputs/benchmarks/phase3_smoke"
mkdir -p "${BENCH_ROOT}" outputs/checks outputs/reports

uv run mc bench niah --adapter all --tasks s_niah_1,s_niah_2,s_niah_3 --context-lengths 4096,8192 --samples-per-length 4 --seed 0 --out-dir "${BENCH_ROOT}/niah"
uv run mc bench mqar --adapter all --samples 16 --num-pairs 8 --num-queries 2 --seed 0 --out-dir "${BENCH_ROOT}/mqar"
uv run mc bench longbench --adapter all --tasks single_doc_qa,multi_doc_qa,code --samples-per-task 2 --seed 0 --out-dir "${BENCH_ROOT}/longbench"
uv run mc bench retrieval --adapter all --datasets swde,squad,fda --truncation-lengths 512,1024 --samples-per-dataset 2 --seed 0 --out-dir "${BENCH_ROOT}/retrieval"
uv run mc bench longbench --adapter all --tasks single_doc_qa,code --samples-per-task 2 --seed 0 --dataset-file examples/longbench_subset.jsonl --out-dir "${BENCH_ROOT}/longbench_dataset_file"
uv run mc bench retrieval --adapter all --datasets swde,squad --truncation-lengths 64 --samples-per-dataset 2 --seed 0 --dataset-file examples/retrieval_subset.jsonl --out-dir "${BENCH_ROOT}/retrieval_dataset_file"

uv run python scripts/reports/validate_evidence_bundle.py --root "${BENCH_ROOT}"
uv run python scripts/reports/benchmark_trend.py --root "${BENCH_ROOT}" --out-json outputs/reports/phase3_benchmark_trend.json --out-md outputs/reports/phase3_benchmark_trend.md
uv run python scripts/reports/write_phase_summary.py --phase phase3 --out outputs/checks/phase3_summary.json --input outputs/reports/phase3_benchmark_trend.json
