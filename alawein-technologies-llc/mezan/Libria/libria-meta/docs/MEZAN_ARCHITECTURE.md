# MEZAN Architecture (Visual)

```mermaid
flowchart LR
  U[User/Client] -->|Composite JSON| ORCH[/MEZAN Orchestrator API/]
  ORCH --> C2SC[C2SC Compiler]
  C2SC -->|assignment| QAP[QAPFlow]
  C2SC -->|workflow| WF[WorkFlow]
  C2SC -->|allocation| AL[AllocFlow]
  QAP -->|A/B or fit/interaction| Backend[Backend Adapter]
  Backend -->|Modes: hybrid,fft,enhanced,nesterov,instance_adaptive,aggressive| Modular[Modular Repo]
  Modular --> Backend
  QAP --> ORCH
  WF --> ORCH
  AL --> ORCH
  ORCH -->|Aggregated Plan| U
```

- C2SC maps composite problem into subproblems.
- Backend adapter transforms schemas and calls modular solvers.
- Orchestrator aggregates and returns a unified response.

