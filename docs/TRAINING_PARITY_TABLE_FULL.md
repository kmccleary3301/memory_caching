# Training Parity Table

targets_yaml: configs/train/paper_targets_full.yaml

| Profile | Target Seq | Actual Seq | Target Batch | Actual Batch | Target Steps | Actual Steps | Steps Ratio | Eval Proxy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mid_full | 8192 | 8192 | 2 | 2 | 5000 | 5000 | 1.0000 | 0.000024 |
| pilot_full | 4096 | 4096 | 4 | 4 | 1000 | 1000 | 1.0000 | 0.000027 |
| target_full | 16384 | 16384 | 1 | 1 | 10000 | 10000 | 1.0000 | 0.000027 |
