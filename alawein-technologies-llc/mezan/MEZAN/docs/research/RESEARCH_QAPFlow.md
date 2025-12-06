# QAPFlow Research Documentation

Problem Definition
- Quadratic Assignment Problem (QAP): assign n facilities to n locations to minimize Σ A[i,k] · B[j,l] over assignments (i→j, k→l). NP-hard; classical combinatorial optimization.

Research Context
- Classical methods: branch-and-bound, tabu search, simulated annealing; heuristics and metaheuristics widely used.
- Benchmarks: QAPLIB instances (various sizes, structured/unstructured).

Our Approach
- Frontend: standardized MEZAN schema accepting QAP (A,B) or assignment-style (fit/interaction) inputs.
- Backend: modular repository integration with selectable modes (hybrid, fft, enhanced, nesterov, instance_adaptive, aggressive) and `mode=auto` size policy.
- Metadata: returns objective, assignment, solve_time, and `bound` if provided by backend.

Implementation Status
- Frontend and adapter implemented (`libria_meta/solvers/qapflow.py`, `Librex.QAP_backend.py`).
- CLI and HTTP orchestration implemented with schema validation.
- Bench harness supports QAPLIB, multi-mode sweeps, CSV/JSONL outputs, bucket summaries.

Benchmarking Plan
- Datasets: QAPLIB `.dat` files.
- Metrics: objective value and solve_time; optional bound gap when available.
- Runs: time-limited sweeps across modes with reproducible seeds and environment capture.

MEZAN Integration
- Orchestrator endpoint: `/qapflow/solve`.
- Composite flows: assignment used by C2SC prior to workflow and allocation.

