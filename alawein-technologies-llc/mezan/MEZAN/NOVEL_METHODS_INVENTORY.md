# MEZAN Novel Methods Inventory

Scope
- Inventory of techniques implemented or planned across the Libria Suite, surfaced as *Flow frontends. Status tags: implemented, embedded, spec_only, baseline.

QAPFlow (Assignment / QAP)
- Modular backend modes (implemented): hybrid, fft, enhanced, nesterov, instance_adaptive, aggressive.
- Homotopy continuation hint (embedded): forwarded where supported.
- Robustness probe (embedded): optional uniform perturbation for A/B (bench-only).
- Bounds passthrough (embedded): captures `bound` if backend provides it.

WorkFlow (Routing / DAG Sequencing)
- Frontend & schema (implemented); solver stub (baseline); integration through orchestrator.

AllocFlow (Budget / Portfolio)
- Frontend & schema (implemented); baseline bandit/knapsack patterns planned (spec_only).

GraphFlow (Topology / Architecture)
- Adapter stub through orchestrator (embedded); research notes exist (spec_only).

MetaFlow (Meta-learning / Strategy Selection)
- Librex.Meta clustering + Elo + UCB (implemented) for solver selection.

DualFlow (Robust / Adversarial)
- Adapter stub through orchestrator (spec_only); robust objective design under discussion.

EvoFlow (Evolution / Search)
- Adapter stub through orchestrator (spec_only).

Summary
- Implemented: QAPFlow frontend/backend adapter and bench; Librex.Meta selection; WorkFlow/AllocFlow frontends; orchestrator service.
- Embedded: Homotopy hint, robustness tests, bounds passthrough, health endpoint, request timeout.
- Spec_only: GraphFlow/DualFlow/EvoFlow adapters and problem specs.

