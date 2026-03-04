# Benchmark Command Matrix

- NIAH:
  - `uv run mc bench niah --adapter all --tasks s_niah_1,s_niah_2,s_niah_3 --context-lengths 4096,8192,16384 --samples-per-length 16 --position-mode uniform --seed 0`
- MQAR:
  - `uv run mc bench mqar --adapter all --samples 64 --pair-grid 8,16,32 --query-grid 1,4,8 --seed 0`
- LongBench scaffold:
  - `uv run mc bench longbench --adapter all --tasks single_doc_qa,multi_doc_qa,summarization,few_shot,code --samples-per-task 4 --seed 0`
- LongBench dataset-file mode:
  - `uv run mc bench longbench --adapter all --tasks single_doc_qa,code --samples-per-task 2 --seed 0 --dataset-file examples/longbench_subset.jsonl`
- Retrieval scaffold:
  - `uv run mc bench retrieval --adapter all --datasets swde,squad,fda --truncation-lengths 512,1024,2048,16384 --samples-per-dataset 4 --seed 0`
- Retrieval dataset-file mode:
  - `uv run mc bench retrieval --adapter all --datasets swde,squad --truncation-lengths 64 --samples-per-dataset 2 --seed 0 --dataset-file examples/retrieval_subset.jsonl`
- Sweep orchestration (timeout/retry/resume):
  - `uv run python scripts/bench/run_benchmark_sweep.py --root outputs/benchmarks/sweeps/default --retries 1 --timeout-sec 1800`
