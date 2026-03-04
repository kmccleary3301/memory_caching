# Checkpoint Contract

Each checkpoint save event should include metadata fields:

- run_name
- global_step
- config_path
- backend_type
- aggregation_type
- sequence_length
- git_commit
- timestamp_utc
- loss_tail
- batch_size
- seed

Save/resume path conventions:

- `artifacts/checkpoints/<profile>/step_<global_step>.pt`
- companion metadata: `artifacts/checkpoints/<profile>/step_<global_step>.json`

Resume consistency evidence:

- Run `./scripts/checks/resume_consistency.sh`
- Output report: `outputs/checks/resume_consistency.json`
