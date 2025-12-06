# MetaFlow Research Documentation

Problem Definition
- Select solvers/strategies dynamically based on instance features and historical performance.

Research Context
- Meta-learning for algorithm selection; clustering; tournament methods; contextual bandits.

Our Approach
- Librex.Meta: KMeans clustering, Elo ratings (global + per-cluster), UCB-based selection, Swiss-system tournament simulation.

Implementation Status
- Implemented in `libria_meta/meta_solver.py` (Librex.Meta) and used by baselines; integration points exposed for orchestrator.

Benchmarking Plan
- ASlib scenarios (robust ARFF parsing); measure regret/accuracy and selection latency.

Integration
- Guides C2SC and budget allocation by recommending strategies and expected values per instance cluster.

