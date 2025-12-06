# Reproducibility Protocol

Artifacts
- Problems: raw instances + normalized inputs; SHA-256 checksums for both.
- Results: solution JSON per request, JSONL per bench run, CSV/MD summaries.

Environment Capture
- Python version, OS, package versions/hashes, CPU count, env variables (QAP_MODULAR_REPO_PATH, safe-mode, timeouts).

Submission Template (YAML)
```
benchmark: QAPLIB
instance: tai20a
hash_raw: "..."
hash_norm: "..."
mode: hybrid
time_limit: 60
seed: 42
objective: 12345.0
solve_time: 3.14
bound: null
commit: <git-sha>
env:
  python: 3.11.9
  os: linux
  numpy: 1.26.4
```

Validation
- Schema validation for inputs/outputs; drift thresholds for nightly CI.

