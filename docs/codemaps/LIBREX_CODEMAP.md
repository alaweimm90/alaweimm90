# Librex Codemap

> **Location:** `.archive/organizations/AlaweinOS/Librex/` + `Librex.QAP/`
> **Status:** Development
> **Domain:** librex.dev (to register)

---

## Architecture Overview

```mermaid
flowchart TB
    subgraph Core["Librex Core"]
        LIB[librex-core]
        API[Python API]
        CLI[CLI Interface]
    end

    subgraph Solvers["Solver Modules"]
        QAP[QAP Solver]
        FLOW[Flow Solver]
        ALLOC[Allocation Solver]
        EVO[Evolutionary]
        GRAPH[Graph Solver]
    end

    subgraph AI["AI Integration"]
        ML[ML Optimization]
        RL[Reinforcement Learning]
        HYBRID[Hybrid Solvers]
    end

    subgraph Output["Outputs"]
        VIZ[Visualization]
        BENCH[Benchmarks]
        REPORTS[Reports]
    end

    CLI --> Core
    API --> Core
    Core --> Solvers
    Solvers --> AI
    AI --> Output

    style Core fill:#3B82F6,color:#fff
    style Solvers fill:#10B981,color:#fff
    style AI fill:#8B5CF6,color:#fff
```

---

## Directory Structure

### Librex Core

```
Librex/
├── src/                         # Core library source
├── ai/                          # AI/ML integration
├── benchmarks/                  # Performance benchmarks
├── data/                        # Test datasets
├── docs/                        # Documentation
├── equilibria/                  # Equilibrium solvers
│   └── quantum/                 # Quantum optimization
├── examples/                    # Usage examples
├── governance/                  # Project governance
├── reports/                     # Generated reports
├── scripts/                     # Utility scripts
├── tests/                       # Test suites
└── tools/                       # Development tools
```

### Librex.QAP (Separate Module)

```
Librex.QAP/
├── src/                         # QAP solver source
├── docs/
│   ├── community/               # Community docs
│   ├── deployment/              # Deployment guides
│   ├── launch/                  # Launch playbooks
│   ├── research/                # Research docs
│   └── templates/               # Templates
├── tests/
└── .archive/                    # Historical docs
```

---

## Solver Inventory

| Solver | Problem Domain | Status | Location |
|--------|----------------|--------|----------|
| **QAP** | Quadratic Assignment | Active | `Librex.QAP/` |
| **Flow** | Network Flow | Planned | `Librex/` |
| **Alloc** | Resource Allocation | Planned | `Librex/` |
| **Evo** | Evolutionary | Planned | `Librex/` |
| **Graph** | Graph Optimization | Planned | `Librex/` |
| **Dual** | Dual Problems | Planned | `Librex/` |
| **Meta** | Meta-optimization | Planned | `Librex/` |

---

## Key Files

| File | Purpose |
|------|---------|
| `src/librex/__init__.py` | Main package |
| `src/librex/solvers/` | Solver implementations |
| `src/librex/models/` | Problem models |
| `ai/` | ML-enhanced solvers |
| `benchmarks/` | Performance tests |

---

## Integration with MEZAN

```mermaid
flowchart LR
    MEZAN[MEZAN Orchestrator] --> Librex[Librex Solvers]
    Librex --> QAP[QAP]
    Librex --> Flow[Flow]
    Librex --> Alloc[Alloc]

    style MEZAN fill:#8B5CF6,color:#fff
    style Librex fill:#10B981,color:#fff
```

MEZAN uses Librex as its solver backend, orchestrating multiple solvers for complex optimization problems.

---

## Deployment

- **Package:** PyPI (`pip install librex`)
- **Docs:** GitHub Pages
- **Benchmarks:** Automated CI

---

_Last Updated: December 5, 2025_
