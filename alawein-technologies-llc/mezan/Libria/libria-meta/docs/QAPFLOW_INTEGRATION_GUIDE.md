# QAPFlow Integration Guide

This guide explains how the modular QAP repository integrates into MEZAN’s QAPFlow and how to run solvers, benchmarks, and the orchestrator service.

Overview
- Base implementation: QAP_modular_repo-main (source of truth for solver methods)
- Adapter: `Libria/libria-meta/Librex.QAP_backend.py` maps MEZAN problem format to QAP A/B and calls modular solvers
- Front-ends: QAPFlow CLI (`python -m libria_meta.cli.qapflow_cli`) and HTTP service (`python -m libria_meta.services.mezan_orchestrator`)
- Schemas: JSON schemas validate inputs and outputs for consistency

Environment Setup
- Set the modular repo path to its `src` directory:
  - WSL/Linux: `export QAP_MODULAR_REPO_PATH="/mnt/c/Users/mesha/Documents/Projects/Active/Librex.QAP/projects/QAP_modular_repo-main/src"`
  - Windows PowerShell: `$env:QAP_MODULAR_REPO_PATH="C:\\Users\\mesha\\Documents\\Projects\\Active\\Librex.QAP\\projects\\QAP_modular_repo-main\\src"`
- To use the external backend, set: `QAPFLOW_BACKEND=external` (or pass `--backend external` to CLI)

Windows vs WSL Notes
- The backend has a WSL default path; on native Windows, always set `QAP_MODULAR_REPO_PATH` explicitly.
- Ensure your Python process can import from that path (e.g., the `src` contains `solvers/`).
- Consider running within WSL for consistent paths across tooling.

Problem Formats
- Preferred (direct QAP form):
  ```json
  { "A": [[...]], "B": [[...]], "constraints": {}}
  ```
- Fallback (assignment-style):
  ```json
  { "fit": [[...]], "interaction": [[...]], "constraints": {}}
  ```
  If `interaction` is omitted, backend uses `B = I` (reduces to linear assignment in QAP form).

CLI Examples
- Solve with hybrid mode (external backend):
  ```bash
  QAPFLOW_BACKEND=external \
    python -m libria_meta.cli.qapflow_cli \
    --problem Libria/libria-meta/examples/problems/qapflow_minimal.json \
    --mode hybrid --time-limit 60
  ```
- Benchmark QAPLIB directory (requires `QAPLIB_DATA_DIR`):
  ```bash
  QAPLIB_DATA_DIR=/path/to/qaplib \
  QAPFLOW_BACKEND=external \
    python -m libria_meta.cli.qapflow_cli \
    --bench qaplib \
    --mode nesterov \
    --time-limit 60 \
    --out results.json
  ```

Multi-Mode Sweeps and CSV Output
- Sweep multiple modes: `--modes hybrid,nesterov,enhanced`
- Write CSV instead of JSON by using `--out file.csv`.
 - CSV run also writes a `_summary.csv` and `_summary.md` with mode-level aggregates.

Robustness Bench (Optional)
- `--robust-eps` adds symmetric noise to A/B in bench mode to test sensitivity.
- This is an evaluation flag; the solver itself remains unchanged.

Homotopy Hint (Optional)
- `--use-homotopy` passes a hint to the backend. If the modular solver supports homotopy, it may use it; otherwise it is ignored.

Supported Modes (via modular repo)
- `hybrid`, `fft`, `enhanced`, `nesterov`, `instance_adaptive`, `aggressive`
- The adapter imports functions from `solvers/` in the modular repo and normalizes outputs to QAPFlow’s solution schema.
- If the import fails or the repo path is missing, QAPFlow falls back to a greedy assignment to keep pipelines operational.

Auto Mode Policy
- You can set `--mode auto` (or `{"mode": "auto"}` via HTTP) to let the backend choose a solver based on problem size `n`:
  - `n <= 20` → `hybrid`
  - `20 < n <= 50` → `nesterov`
  - `n > 50` → `enhanced`
- Environment override: `QAPFLOW_AUTO_POLICY="S,M"` sets thresholds (defaults `20,50`).
  - Example: `QAPFLOW_AUTO_POLICY="30,80"` → `n <= 30` → `hybrid`; `31–80` → `nesterov`; `>80` → `enhanced`
  - Selected policy is echoed in solution metadata as `metadata.auto_policy` when auto mode is used.

HTTP Orchestrator Service
- Start: `python -m libria_meta.services.mezan_orchestrator`
- Endpoints (POST JSON, schemas enforced):
  - `/qapflow/solve` (QAPFlow)
  - `/qapflow/solve_ab` (alias; same payload)
  - `/workflow/solve` (WorkFlow)
  - `/allocflow/solve` (AllocFlow)
- Health: `GET /health` returns available modes (if modular repo is importable) and detected `QAP_MODULAR_REPO_PATH`.

Docker (Optional)
- Build: `docker build -t mezan-orchestrator -f Libria/libria-meta/Dockerfile .`
- Run: `docker run --rm -p 8081:8081 mezan-orchestrator`

Web UI (Optional)
- Dashboard: `GET /ui` (health + latest bench jobs)
- Bench list: `GET /bench/ui`
- Bench job detail: `GET /bench/ui/{job_id}` (charts + tables)

Troubleshooting
- If backend import fails, confirm `QAP_MODULAR_REPO_PATH` is set to the modular repo `src` and that the path is readable.
- On Windows, prefer running via WSL for path consistency, or set the env var to the Windows path and ensure Python can import it.
- For QAPLIB, ensure `.dat` files are present and `QAPLIB_DATA_DIR` points to that directory.

Files
- Adapter: `Libria/libria-meta/Librex.QAP_backend.py`
- CLI: `libria_meta/cli/qapflow_cli.py`
- Schemas: `libria_meta/schemas/qapflow_problem.json`, `libria_meta/schemas/qapflow_solution.json`
- Orchestrator: `libria_meta/services/mezan_orchestrator.py`
