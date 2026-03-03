# NIAH Normalization Policy

Scoring uses exact-match after normalization:

1. strip leading/trailing whitespace
2. collapse repeated whitespace to single spaces
3. lowercase normalization

This policy is applied consistently in `normalize_answer` for NIAH scoring.
