# Multi-Feeder Architecture

Overview
- Parallel execution of subproblems and solver runs when dependencies allow; aggregation layer reconciles results with precedence constraints.

Execution Model
- Dependency graph from C2SC determines which feeders (assignment/workflow/allocation/…) can run concurrently.
- Each feeder reports structured results and timings; orchestrator applies conflict resolution and consistency checks.

Resource Control
- Global limits via AllocFlow; per-request time budgets; cancellation and fallback policies where possible.

Results Aggregation
- Stable merge of assignments, routes, and budgets into a final plan; consistent metadata for traceability.

