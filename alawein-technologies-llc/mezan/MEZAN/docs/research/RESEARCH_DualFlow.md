# DualFlow Research Documentation

Problem Definition
- Robust optimization under adversarial perturbations; min–max formulations for pipeline plans or assignments.

Research Context
- Robust combinatorial optimization; adversarial testing and red-teaming.

Our Approach
- Orchestrator adapter to evaluate candidate solutions under defined perturbation sets; integrate with Librex.Meta for robustness signals.

Implementation Status
- Spec_only; orchestrator wiring present.

Benchmarking Plan
- Synthetic adversarial perturbations; report worst-case objective vs baseline.

Integration
- Supports “Nightmare Mode” style stress tests before promotion of strategies.

