# AllocFlow Research Documentation

Problem Definition
- Allocate finite budget/compute/time to competing agents/tasks to maximize utility under constraints.

Research Context
- Knapsack/portfolio optimization and bandit-style allocation with non-stationary rewards.

Our Approach
- Frontend schema for requests and budgets; planned baselines include UCB/Thompson Sampling with constraints and simple knapsack solvers.

Implementation Status
- Frontend/schema and orchestrator endpoint `/allocflow/solve` implemented; solver policies planned.

Benchmarking Plan
- Synthetic and replayed workloads; measure total reward and fairness metrics.

Integration
- Follows assignment/workflow; feeds resource limits back to execution plan.

