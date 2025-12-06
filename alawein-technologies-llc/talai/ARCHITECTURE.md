# System Architecture

Technical architecture documentation for IDEAS framework components.

## System Overview

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐
│             │     │              │     │              │
│  IdeaForge  │────▶│  BuildForge  │────▶│   Turingo    │
│             │     │              │     │              │
└─────────────┘     └──────────────┘     └──────────────┘
  Idea Gen.          Validation           Optimization
  15 frameworks      5 gates              14 agents
  17 agents          YAML config          3 paradigms
```

## Component Architecture

### IdeaForge

```
ideaforge.py (Main CLI)
    │
    ├─▶ frameworks.py (15 thinking frameworks)
    │   ├─ FirstPrinciplesFramework
    │   ├─ InversionFramework
    │   ├─ CrossDomainAnalogyFramework
    │   └─ ... (12 more)
    │
    └─▶ agents.py (17 specialist agents)
        ├─ MetaOrchestrator
        ├─ Planner
        ├─ Researcher
        └─ ... (14 more)

Data Flow:
Input → KnowledgeGraph → MultiFrameworkProcessor → IdeaGenerator → Scored Ideas → JSON
```

**Key Classes:**
- `KnowledgeGraph`: Represents extracted concepts and relationships
- `MultiFrameworkProcessor`: Coordinates parallel framework execution
- `IdeaGenerator`: Main idea generation engine
- `Idea`: Data class for generated ideas with scores

**Process:**
1. Extract knowledge graph from input text
2. Process through 15 frameworks in parallel
3. Generate ideas from framework insights
4. Score ideas (novelty × impact × feasibility)
5. Rank and filter results

### BuildForge

```
buildforge.py (Main Orchestrator)
    │
    ├─▶ DomainConfig (YAML configuration)
    │
    └─▶ Gates
        ├─ Gate1_NoveltyAssessment
        ├─ Gate2_TheoryValidation
        ├─ Gate3_FeasibilityAnalysis
        ├─ Gate4_BenchmarkValidation
        └─ Gate5_PublicationDeployment

Data Flow:
Idea + Config → Gate 1 → Gate 2 → Gate 3 → Gate 4 → Gate 5 → GateResult[]
                   ↓        ↓        ↓        ↓        ↓
              Novelty   Theory   Feasible  Benchmarks Published
              ≥65?      Sound?   Cost/Time  Beat SOTA?  Docs
```

**Gate Logic:**
- Each gate returns: GateResult(verdict, score, insights, risks, recommendations)
- Verdict: PROCEED | PIVOT | STOP
- Gates execute sequentially
- STOP halts pipeline; PIVOT allows continuation with warnings

**Configuration:**
```yaml
domain: string
keywords: list
novelty_threshold: float (0-100)
sota_baselines: list
benchmarks: dict
evaluation_metrics: list
release_criteria: list
```

### Turingo

```
turingo.py (Main Orchestrator)
    │
    ├─▶ Executive Core (4 agents)
    │   ├─ Ringmaster (orchestration)
    │   ├─ BlueprintBoss (architecture)
    │   ├─ DealMaker (ROI evaluation)
    │   └─ EthicsEnforcer (validation)
    │
    ├─▶ Specialists (10 agents)
    │   ├─ PuzzleProdigy (problem analysis)
    │   ├─ QuantumQuokka (quantum algorithms)
    │   ├─ MLMagician (ML solutions)
    │   ├─ ... (7 more)
    │
    └─▶ Workflows (5 SOPs)
        ├─ SOP_01_Selection
        ├─ SOP_02_Stampede
        ├─ SOP_03_Validation
        ├─ SOP_04_Publication
        └─ SOP_05_MetaLearning

Data Flow:
Problem → SOP-01 → SOP-02 → SOP-03 → SOP-04 → SOP-05
          Select   Solve    Validate  Publish  Learn
            ↓         ↓         ↓
         ROI      3 paradigms Benchmark
         Analysis  parallel    testing
```

**Agent Communication:**
```
Agent → send_message() → Message Queue → Async Processing → Response
```

**Problem Solving Flow:**
```
1. Problem Analysis (Puzzle Prodigy)
2. Ethics Check (Ethics Enforcer)
3. Architecture Design (Blueprint Boss)
4. Multi-Paradigm Stampede:
   ├─ Alpha Squad (Quantum)
   ├─ Beta Brigade (ML)
   └─ Gamma Gang (Classical)
5. Solution Selection (best objective)
6. Validation (Benchmark Bandit, Novelty Ninja, Skeptic Sorcerer)
7. Optional: Proof Verification
8. Result Archival
```

## Data Structures

### IdeaForge

```python
@dataclass
class Idea:
    id: str
    title: str
    description: str
    framework: str
    novelty_score: float  # 0-10
    impact_score: float   # 0-10
    feasibility_score: float  # 0-10
    combined_score: float  # product of scores
    domain: str
    keywords: List[str]
    implementation_hints: List[str]
    risks: List[str]
    generated_at: str
    agent: str
```

### BuildForge

```python
@dataclass
class GateResult:
    gate_number: int
    gate_name: str
    verdict: str  # PROCEED | PIVOT | STOP
    score: float
    confidence: float
    insights: List[str]
    risks: List[str]
    recommendations: List[str]
    timestamp: str
    duration_seconds: float
```

### Turingo

```python
@dataclass
class TuringoResult:
    problem_type: str
    instance_name: str
    objective_value: float
    solution: List[int]
    winning_paradigm: str
    improvement_over_sota: float
    computation_time_seconds: float
    validation_status: str
    novelty_score: float
    proof_verified: bool
    agents_used: List[str]
    timestamp: str
```

## Concurrency Model

### IdeaForge
```python
# Frameworks processed in parallel
async def process_parallel(self, graph):
    tasks = [framework.analyze(graph) for framework in self.frameworks.values()]
    results = await asyncio.gather(*tasks)
    return results
```

### Turingo
```python
# Multi-paradigm parallel solving
async def execute(self, problem_type, instance, architecture, time_limit):
    tasks = []
    if "quantum" in paradigms:
        tasks.append(self._alpha_squad(problem_type, instance))
    if "ml" in paradigms:
        tasks.append(self._beta_brigade(problem_type, instance))
    if "classical" in paradigms:
        tasks.append(self._gamma_gang(problem_type, instance))

    solutions = await asyncio.gather(*tasks)
    return solutions
```

## Configuration Management

### IdeaForge
Runtime configuration via CLI arguments and default config dict.

### BuildForge
YAML files in `domains/` directory:
```
domains/
├── example/
│   ├── config.yaml
│   └── README.md
└── my_project/
    └── config.yaml
```

### Turingo
Configuration via CLI and optional JSON config file.

## Error Handling

All components use Python exceptions:
- `ValueError`: Invalid configuration or input
- `FileNotFoundError`: Missing required files
- `RuntimeError`: Execution failures

Results include status fields:
- BuildForge: `verdict` field (PROCEED/PIVOT/STOP)
- Turingo: `validation_status` field (PASS/CONDITIONAL/FAIL)

## Logging

Current implementation uses `print()` statements.
Production would use Python `logging` module with configurable levels.

## Testing Strategy

### Unit Testing
Test individual frameworks, agents, and gates in isolation.

### Integration Testing
Test complete workflows:
- IdeaForge: generate → rank → filter
- BuildForge: all 5 gates sequentially
- Turingo: solve single problem, autonomous rodeo

### Performance Testing
Measure execution times for typical workloads:
- IdeaForge: 10-50 ideas
- BuildForge: Full pipeline
- Turingo: Problem instances of varying sizes

## Deployment Considerations

### Current State
- Single-machine execution
- No external dependencies (except Python libraries)
- File-based result storage
- CLI-only interface

### Future Considerations
- Distributed agent execution
- Database for results/knowledge base
- REST API interface
- Web UI
- Cloud deployment (AWS/GCP)
- Container orchestration (Docker/Kubernetes)

## Performance Characteristics

Based on prototype measurements:

| Operation | Time | Notes |
|-----------|------|-------|
| IdeaForge generate (10 ideas) | <1s | Simulated frameworks |
| BuildForge full pipeline | 15s | 5 gates sequential |
| Turingo solve (single) | 1.6s | 3 paradigms parallel |
| Turingo rodeo (5 problems) | 8s | Includes selection |

Real performance will vary with:
- Actual ML model training time
- Quantum hardware queue time
- Problem complexity
- Hardware specifications

## Security Considerations

### Current Implementation
- No authentication/authorization
- Local file system access only
- No network communication
- No sensitive data handling

### For Production
- Input validation required
- Sandboxed execution environments
- Rate limiting for resource-intensive operations
- Audit logging
- Secrets management for API keys

## Extensibility

### Adding New Frameworks (IdeaForge)
1. Subclass `ThinkingFramework`
2. Implement `analyze()` method
3. Register in `ALL_FRAMEWORKS` dict

### Adding New Gates (BuildForge)
1. Create new gate class
2. Implement `execute()` method
3. Add to gates list in main orchestrator

### Adding New Agents (Turingo)
1. Subclass `TuringoAgent` (Executive/Specialist/Consultant)
2. Implement `execute_task()` method
3. Register in agent initialization

### Adding Problem Types (Turingo)
1. Add problem analysis in PuzzleProdigy
2. Implement formulations in paradigm agents
3. Add SOTA baselines in BenchmarkBandit
4. Update configuration schema

---

Last updated: 2025-11-15
