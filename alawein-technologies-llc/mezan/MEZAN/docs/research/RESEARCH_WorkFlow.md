# WorkFlow Research Documentation

Problem Definition
- Workflow routing/sequencing over a DAG of stages (e.g., Designer → Critic → Refactorer → Validator), with costs and expected benefits per transition.

Research Context
- Related to job-shop/flow-shop scheduling and DAG scheduling; objectives include latency, throughput, and quality.

Our Approach
- Frontend schema for stage graphs and constraints; orchestrator adapter calls the WorkFlow solver.
- Baseline policies (spec_only): shortest-path on expected cost, greedy improvements; RL-based structure selection possible in future.

Implementation Status
- Frontend/schema implemented; solver baseline under development; orchestrator endpoint `/workflow/solve` present.

Benchmarking Plan
- Synthetic DAGs and replayed orchestration traces; measure completion time and output quality proxies.

Integration
- C2SC composes assignment → workflow → allocation; WorkFlow routes items through stages based on constraints and policies.

