# Balance Philosophy (mīzān)

Principles
- Balance speed, quality, and safety explicitly in objectives and constraints.
- Make alignment a first-class requirement: schemas, safe-mode, timeouts.
- Favor stable, reproducible behaviors over marginal speedups.

Practice
- Multi-objective trade-offs made explicit at C2SC; encoded as weights or constraints.
- Robustness tests (e.g., noise perturbations) before promoting strategies.
- Clear fallbacks when backends are unavailable.

