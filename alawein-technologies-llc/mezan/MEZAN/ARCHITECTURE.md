# MEZAN System Architecture

1. System Overview
- MEZAN receives composite problem requests (JSON), compiles them into subproblems (assignment, workflow, allocation, optional graph/meta/dual/evo), calls solver frontends/backends, and aggregates results into an actionable plan.
- The system enforces schemas at boundaries and supports safe-mode and request timeouts at the API layer.

2. Orchestrator Layer
- HTTP server: `libria_meta.services.mezan_orchestrator`
  - Endpoints: `/qapflow/solve`, `/workflow/solve`, `/allocflow/solve`, `/mezan/solve`, plus health.
  - Validates payloads against schemas; caps request size via `ORCH_MAX_BODY` and optional wall-time via `ORCH_REQUEST_TIMEOUT`.
- Composite CLI: `libria_meta.mezan_engine.cli` for local invocation of C2SC + sub-solvers without HTTP.

3. Constraint-to-Solver Compiler (C2SC)
- Maps composite requests into subproblems with typed schemas.
- Establishes execution order (e.g., assignment → workflow → allocation) and propagates constraints and intermediate products.
- Encapsulated in `libria_meta.mezan_engine.orchestrator`.

4. Solver Frontends (*Flow)
- QAPFlow: assignment/QAP subproblems; accepts QAP (A,B) or assignment-style (fit/interaction) inputs.
- WorkFlow: pipeline sequencing/routing; DAG-oriented.
- AllocFlow: budget/compute allocation; knapsack/bandit-oriented.
- GraphFlow, MetaFlow, DualFlow, EvoFlow: stubs/specs are present; adapters are wired for future integration.

5. Backend Adapters
- QAPFlow uses `Librex.QAP_backend` as a shim to an external modular solver repository.
  - Modes: `hybrid`, `fft`, `enhanced`, `nesterov`, `instance_adaptive`, `aggressive`, and `auto` (size-based policy).
  - Env `QAP_MODULAR_REPO_PATH` points to the backend repo `src`.
  - Safe mode: `QAPFLOW_SAFE_MODE=1` skips external imports.
  - Homotopy hint `use_homotopy` is forwarded where supported.
  - If the backend is unavailable, a fallback greedy assignment keeps pipelines running.

6. Meta-Learning & Selection
- `Librex.Meta` clusters instances, maintains Elo ratings (global + per-cluster), and uses UCB to balance exploration and exploitation.
- Tournament mode (Swiss-system) can simulate matches and produce rankings before running a chosen solver.

7. Data & Benchmarks
- ARFF/ASlib loader provides robust parsing for algorithm selection scenarios: quoted attributes, categorical values, missing values (`?`), extra whitespace, PAR10 mapping for timeout/memout/crash.
- QAPLIB bench: CLI supports multi-mode runs, compact summaries, CSV/JSONL outputs, and bucket-level aggregates.

8. Safety & Observability
- Alignment via schemas, explicit safe-mode, request size and time caps.
- Structured outputs with solution metadata (backend origin, mode, optional bounds if available).
- Logging hooks recommended at orchestration boundaries; current code prints compact bench summaries and exposes `/health` mode detection.

9. Configuration
- Environment variables:
  - `QAP_MODULAR_REPO_PATH`: path to modular QAP repo `src`.
  - `QAPFLOW_SAFE_MODE`: `1` to disable external backend.
  - `ORCH_MAX_BODY`: request body cap (bytes).
  - `ORCH_REQUEST_TIMEOUT`: per-request wall-time (seconds).

10. Extensibility
- Add new solvers by implementing a frontend under `libria_meta/solvers/` with schemas and a backend adapter if needed.
- Extend C2SC to include new subproblem types and dependency ordering.
- Integrate OptiBench registry/replay for drift checks and reproducibility (planned docs under benchmarking).

