# MEZAN

Meta-Equilibrium Zero-regret Assignment Network

Overview
- MEZAN is the orchestration layer that composes problem requests into solver subproblems (assignment, workflow, allocation, etc.), routes them to the appropriate Libria solvers, and aggregates results into an executable plan.
- Philosophy: balance (mīzān) across objectives and constraints, safety-first defaults, and transparent, measurable outcomes. “MEZAN” for academic writing; “Mezan” in product/code contexts.
- Scope: compatible with ORCHEX (research automation) and TURING (product automation). The Libria Suite supplies domain solvers, surfaced as “*Flow” frontends (QAPFlow, WorkFlow, AllocFlow, GraphFlow, MetaFlow, DualFlow, EvoFlow).

Architecture
- C2SC (Constraint-to-Solver Compiler): maps composite problems to subproblems.
- Frontends: `QAPFlow`, `WorkFlow`, `AllocFlow`, etc. live in `Libria/libria-meta/libria_meta/solvers/`.
- Backend adapters: translate standardized JSON schemas to specific solver backends (e.g., `Librex.QAP_backend` to a modular QAP solver repo).
- Orchestrator service: HTTP API in `Libria/libria-meta/services/mezan_orchestrator.py` with schema validation, safe-mode, and request timeouts.
- Meta-learning: `Librex.Meta` supports clustering, Elo ratings, and UCB selection for solver tournaments.

Key Features
- Standard schemas for inputs/outputs; strict validation.
- QAPFlow supports A/B QAP form and assignment-style (fit/interaction) inputs.
- External modular backend support for QAP with selectable modes (hybrid, fft, enhanced, nesterov, instance_adaptive, aggressive) and size-based `mode=auto` policy.
- Robust ARFF parsing and ASlib scenario loader for algorithm selection experiments.
- Bench harness and summaries (mode-level and size buckets) for QAPLIB.

Quick Start
1) Orchestrator service
```
python -m libria_meta.services.mezan_orchestrator
# POST /qapflow/solve with {"A": [[...]], "B": [[...]], "mode": "auto", "time_limit": 60}

# Web UI (local):
# - http://127.0.0.1:8081/ui (dashboard)
# - http://127.0.0.1:8081/bench/ui (bench jobs)
```

2) QAPFlow CLI (single problem)
```
python -m libria_meta.cli.qapflow_cli \
  --problem Libria/libria-meta/examples/problems/qapflow_minimal.json \
  --mode hybrid --time-limit 60
```

3) QAPFlow CLI (QAPLIB bench)
```
QAPLIB_DATA_DIR=/path/to/dat QAPFLOW_BACKEND=external \
python -m libria_meta.cli.qapflow_cli \
  --bench qaplib --modes hybrid,nesterov,enhanced --time-limit 60 --out bench.csv
```

Using Modular QAP Backend
- Set `QAP_MODULAR_REPO_PATH` to the `src` folder of your modular QAP repository.
- Optional safe mode: `QAPFLOW_SAFE_MODE=1` disables external backend and uses an internal fallback.
- Homotopy hint is forwarded to supported modes (`use_homotopy=true`).

Directory Pointers
- Orchestrator API: `Libria/libria-meta/services/mezan_orchestrator.py`
- Composite CLI: `Libria/libria-meta/libria_meta/mezan_engine/cli.py`
- Frontends: `Libria/libria-meta/libria_meta/solvers/`
- Backend adapter (QAP): `Libria/libria-meta/Librex.QAP_backend.py`
- Schemas: `Libria/libria-meta/libria_meta/schemas/`
- Bench utils: `Libria/libria-meta/libria_meta/common/bench_utils.py`
- ARFF/ASlib: `Libria/libria-meta/benchmark/aslib_parser.py`

Licensing & Dependencies
- Python 3.9+; numpy, pandas, scipy, scikit-learn.
- Optional: fastjsonschema (schema validation), requests (HTTP clients), matplotlib (figures).
