# Training Parity Table

targets_yaml: configs/train/paper_targets.yaml

| Profile | Target Seq | Actual Seq | Target Batch | Actual Batch | Target Steps | Actual Steps | Steps Ratio | Eval Proxy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mid | 8192 | 128 | 2 | 2 | 5000 | 4 | 0.0008 | 0.000028 |
| pilot | 4096 | 128 | 4 | 4 | 1000 | 4 | 0.0040 | 0.000000 |
| target | 16384 | 128 | 1 | 1 | 10000 | 4 | 0.0004 | 0.000023 |
