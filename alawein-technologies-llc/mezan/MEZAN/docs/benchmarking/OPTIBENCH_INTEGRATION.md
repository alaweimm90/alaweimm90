# OptiBench Integration

Goals
- Standardize instance registries with checksums (SHA-256), result schemas, and replay scripts for reproducibility and drift detection.

Registry
- Proposed folder: `Libria/libria-meta/optibench/`
  - `registry.yaml`: list of named benchmarks with source URLs, expected checksums, and loader hints.
  - `schema/`: JSON schemas for results and summaries.
  - `replay.py`: re-run selected entries and produce comparable outputs.

Checksums
- Record SHA-256 of raw instances (e.g., QAPLIB `.dat`) and of normalized inputs (A/B matrices) to detect parsing differences.

Reproducibility
- Capture: commit, env (Python version, package hashes), seeds, solver parameters, and wall-time caps.
- Store: results as JSONL, summaries as CSV/MD/JSON.

Nightly Replay (CI)
- Sample subset; compare objectives/solve_time within tolerances; alert on drift.

