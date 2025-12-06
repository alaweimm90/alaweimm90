# MEZAN Codemap

> **Location:** `.archive/organizations/AlaweinOS/MEZAN/`
> **Components:** ATLAS, Libria, MEZAN Core
> **Status:** Development

---

## Architecture Overview

```mermaid
flowchart TB
    subgraph MEZAN["MEZAN Meta-Solver"]
        ORCH[Orchestrator]
        SELECTOR[Solver Selector]
        COMBINER[Result Combiner]
    end

    subgraph ATLAS["ATLAS (→ Orchex)"]
        CORE[Atlas Core]
        AGENTS[Agent System]
        WORKFLOW[Workflow Engine]
    end

    subgraph Libria["Libria Solvers"]
        QAP[libria-qap]
        FLOW[libria-flow]
        ALLOC[libria-alloc]
        EVO[libria-evo]
        GRAPH[libria-graph]
        DUAL[libria-dual]
        META[libria-meta]
    end

    MEZAN --> ATLAS
    MEZAN --> Libria
    ATLAS --> AGENTS
    ATLAS --> WORKFLOW

    style MEZAN fill:#8B5CF6,color:#fff
    style ATLAS fill:#3B82F6,color:#fff
    style Libria fill:#10B981,color:#fff
```

---

## Directory Structure

```
MEZAN/
├── ATLAS/                       # Orchestration system (→ Orchex)
│   ├── atlas-core/              # Core orchestration
│   ├── docs/                    # ATLAS documentation
│   ├── examples/                # Usage examples
│   ├── src/                     # Source code
│   └── tests/                   # Test suites
│
├── Libria/                      # Solver library
│   ├── libria-qap/              # Quadratic Assignment
│   ├── libria-flow/             # Network Flow
│   ├── libria-alloc/            # Resource Allocation
│   ├── libria-evo/              # Evolutionary
│   ├── libria-graph/            # Graph Optimization
│   ├── libria-dual/             # Dual Problems
│   ├── libria-meta/             # Meta-optimization
│   ├── libria-integration/      # Integration layer
│   ├── benchmarking/            # Benchmark suite
│   ├── consolidation/           # Consolidation tools
│   ├── meta/                    # Meta-information
│   ├── prompts/                 # AI prompts
│   └── quick_reference/         # Quick reference docs
│
├── MEZAN/                       # MEZAN core
│   └── docs/                    # Core documentation
│
├── core/                        # Shared core utilities
│   └── tests/                   # Core tests
│
├── docs/                        # Project documentation
│   └── specs/                   # Specifications
│
├── scripts/                     # Utility scripts
├── src/                         # Main source
├── tests/                       # Integration tests
└── visualization/               # Visualization tools
```

---

## Component Relationships

### ATLAS → Orchex Migration

ATLAS was renamed to Orchex and moved to `tools/orchex/` in the main repo. The archive contains the original ATLAS code.

```mermaid
flowchart LR
    OLD[MEZAN/ATLAS/] -->|Renamed| NEW[tools/orchex/]
    
    style OLD fill:#EF4444,color:#fff
    style NEW fill:#10B981,color:#fff
```

### Libria Solver Details

| Solver | Purpose | Algorithms |
|--------|---------|------------|
| `libria-qap` | Quadratic Assignment | Simulated Annealing, Genetic |
| `libria-flow` | Network Flow | Ford-Fulkerson, Push-Relabel |
| `libria-alloc` | Resource Allocation | Linear Programming |
| `libria-evo` | Evolutionary | GA, ES, DE |
| `libria-graph` | Graph Problems | Dijkstra, A*, TSP |
| `libria-dual` | Dual Problems | Lagrangian Relaxation |
| `libria-meta` | Meta-optimization | Hyper-parameter tuning |

---

## Key Files

| File | Purpose |
|------|---------|
| `MEZAN/src/mezan.py` | Main orchestrator |
| `Libria/libria-integration/` | Solver integration |
| `ATLAS/src/` | Orchestration engine |
| `visualization/` | Result visualization |

---

## Integration Points

1. **Librex** - Uses Libria solvers as backend
2. **TalAI** - Uses MEZAN for research optimization
3. **Orchex** - Evolved from ATLAS

---

_Last Updated: December 5, 2025_
