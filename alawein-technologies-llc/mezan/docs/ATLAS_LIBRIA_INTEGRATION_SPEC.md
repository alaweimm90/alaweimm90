# ORCHEX ↔ Libria Integration Specification

**Version**: 1.0
**Date**: 2026-01-17
**Purpose**: Define integration contract between ORCHEX Engine (Claude Instance 2) and Libria Suite (Claude Instance 1)

---

## 1. Architecture Overview

```
┌─────────────────────────────────────────┐
│          ORCHEX Engine                    │
│    (Claude Instance 2 - ORCHEX/)         │
│                                          │
│  ┌────────────────────────────────────┐ │
│  │  40+ Research Agents               │ │
│  │  - Synthesis Agent                 │ │
│  │  - Literature Review Agent         │ │
│  │  - Hypothesis Generation Agent     │ │
│  │  - Critical Analysis Agent         │ │
│  │  - ...                             │ │
│  └────────────────────────────────────┘ │
│                                          │
│  ┌────────────────────────────────────┐ │
│  │  Dialectical Workflows             │ │
│  │  - Thesis-Antithesis-Synthesis     │ │
│  │  - Multi-perspective Analysis      │ │
│  │  - Quality Gates                   │ │
│  └────────────────────────────────────┘ │
│                                          │
│  ┌────────────────────────────────────┐ │
│  │  ORCHEX Orchestrator                │ │
│  │  - Task Assignment                  │ │
│  │  - Workflow Routing                │ │
│  │  - Resource Allocation             │ │
│  │  - Topology Management             │ │
│  └───────────┬────────────────────────┘ │
└──────────────│──────────────────────────┘
               │
               │ LibriaRouter API
               │
               ▼
┌─────────────────────────────────────────┐
│        Libria Integration Layer          │
│          (Shared Interface)              │
│                                          │
│  ┌────────────────────────────────────┐ │
│  │  LibriaRouter                      │ │
│  │  - solve_assignment()              │ │
│  │  - route_workflow_step()           │ │
│  │  - allocate_resources()            │ │
│  │  - optimize_topology()             │ │
│  │  - validate_workflow()             │ │
│  │  - evolve_architecture()           │ │
│  └────────────────────────────────────┘ │
└──────────────│──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│         Libria Solvers                   │
│   (Claude Instance 1 - Libria/)         │
│                                          │
│  ┌──────┬──────┬──────┬──────┬──────┐  │
│  │ QAP  │ Meta │ Flow │Alloc │Graph │  │
│  └──────┴──────┴──────┴──────┴──────┘  │
│  ┌──────┬──────┐                        │
│  │ Dual │ Evo  │                        │
│  └──────┴──────┘                        │
└─────────────────────────────────────────┘
```

---

## 2. Integration Points

### 2.1 Agent-Task Assignment (Librex.QAP)

**ORCHEX Use Case**: Assign research tasks to 40+ agents optimally

**API Call**:
```python
# In ORCHEX Engine (Claude Instance 2)
from libria_integration import LibriaRouter

router = LibriaRouter()
assignment = router.solve_assignment(
    agents=list(atlas_engine.agents.values()),
    tasks=pending_tasks
)
```

**Data Flow**:
1. ORCHEX provides list of agents + tasks
2. LibriaRouter extracts agent features (skill, workload, history)
3. Librex.QAP computes optimal assignment
4. ORCHEX receives agent-task mapping
5. ORCHEX assigns tasks accordingly

**Interface Contract**:
```python
def solve_assignment(
    agents: List[ResearchAgent],
    tasks: List[Dict]
) -> Dict:
    """
    Args:
        agents: List of ORCHEX research agents
        tasks: List of pending tasks

    Returns:
        {
            'agent_id': str,  # Assigned agent
            'assignment': np.ndarray,  # Full assignment matrix
            'cost': float  # Total assignment cost
        }
    """
```

### 2.2 Workflow Routing (Librex.Flow)

**ORCHEX Use Case**: Route dialectical workflow steps to best agents with confidence

**API Call**:
```python
# In ORCHEX dialectical workflow
agent_id = router.route_workflow_step(
    workflow_state={
        'step': 'antithesis',  # Current step
        'thesis': thesis_result,
        'available_agents': antithesis_capable_agents,
        'execution_path': previous_steps,
        'confidence_history': confidence_scores
    }
)
```

**Data Flow**:
1. ORCHEX provides workflow state + available agents
2. Librex.Flow extracts workflow features
3. LinUCB selects agent maximizing quality + calibration
4. Returns agent_id + confidence estimate
5. ORCHEX routes to selected agent

**Interface Contract**:
```python
def route_workflow_step(
    workflow_state: Dict
) -> str:
    """
    Args:
        workflow_state: {
            'step': str,  # Current workflow step
            'execution_path': List[Dict],  # Previous steps
            'available_agents': List[ResearchAgent],
            'thesis': Optional[Dict],  # For dialectical workflows
            ...
        }

    Returns:
        agent_id: str  # Selected agent ID
    """
```

### 2.3 Resource Allocation (Librex.Alloc)

**ORCHEX Use Case**: Allocate compute/memory/bandwidth to agents under budget

**API Call**:
```python
allocation = router.allocate_resources(
    agents=atlas_engine.agents.values(),
    total_budget={
        'compute': 10.0,  # CPU cores
        'memory': 32.0,   # GB RAM
        'bandwidth': 1000.0  # Mbps
    }
)
```

**Data Flow**:
1. ORCHEX provides agents + total budget
2. Librex.Alloc builds resource request matrix
3. Constrained Thompson Sampling solves allocation
4. Returns resource allocation per agent
5. ORCHEX applies resource limits

**Interface Contract**:
```python
def allocate_resources(
    agents: List[ResearchAgent],
    total_budget: Dict[str, float]
) -> Dict[int, np.ndarray]:
    """
    Args:
        agents: List of ORCHEX agents
        total_budget: {'compute': float, 'memory': float, 'bandwidth': float}

    Returns:
        allocation: {agent_idx: np.array([compute, memory, bandwidth])}
    """
```

### 2.4 Topology Optimization (Librex.Graph)

**ORCHEX Use Case**: Optimize agent communication network for information flow

**API Call**:
```python
# Periodic topology optimization
agent_states = atlas_engine.get_agent_state_history(window=100)
optimal_topology = router.optimize_topology(agent_states)
atlas_engine.apply_communication_topology(optimal_topology)
```

**Data Flow**:
1. ORCHEX provides agent state history (T × n × d)
2. Librex.Graph estimates pairwise mutual information
3. Spectral optimization finds best topology
4. Returns adjacency matrix
5. ORCHEX updates communication graph

**Interface Contract**:
```python
def optimize_topology(
    agent_states: np.ndarray  # (T × n × d)
) -> np.ndarray:
    """
    Args:
        agent_states: Agent state time series (T timesteps, n agents, d dimensions)

    Returns:
        adjacency_matrix: (n × n) binary communication topology
    """
```

### 2.5 Workflow Validation (Librex.Dual)

**ORCHEX Use Case**: Pre-deployment adversarial testing of dialectical workflows

**API Call**:
```python
validation_report = router.validate_workflow(
    workflow_spec={
        'workflow': thesis_antithesis_synthesis_workflow,
        'safety_spec': lambda output: output['quality'] > 0.8,
        'test_cases': test_inputs
    }
)
```

**Data Flow**:
1. ORCHEX provides workflow + safety spec + test cases
2. Librex.Dual runs red-team attacks (PGD, genetic, beam)
3. Blue team robustifies workflow
4. Returns validation report + robustified workflow
5. ORCHEX deploys robustified version

**Interface Contract**:
```python
def validate_workflow(
    workflow_spec: Dict
) -> Dict:
    """
    Args:
        workflow_spec: {
            'workflow': Callable,
            'safety_spec': Callable[[Dict], bool],
            'test_cases': List[Dict]
        }

    Returns:
        {
            'attack_success_rate': float,
            'failed_cases': List[Dict],
            'robustified_workflow': Callable,
            'certified_radius': float
        }
    """
```

### 2.6 Architecture Evolution (Librex.Evo)

**ORCHEX Use Case**: Discover optimal agent team architectures via MAP-Elites

**API Call**:
```python
results = router.evolve_architecture(
    task_distribution=atlas_benchmark_tasks
)

# Get best architecture for specific behavior
best_arch = results['archive'][(communication_bin, specialization_bin)]
atlas_engine.reconfigure(best_arch)
```

**Data Flow**:
1. ORCHEX provides task distribution
2. Librex.Evo evolves agent architectures via MAP-Elites
3. Returns diverse archive of architectures
4. ORCHEX selects architecture for current needs

**Interface Contract**:
```python
def evolve_architecture(
    task_distribution: List[Dict]
) -> Dict:
    """
    Args:
        task_distribution: Sample of ORCHEX tasks for evaluation

    Returns:
        {
            'archive': Dict[Tuple, AgentArchitecture],
            'coverage': float,
            'max_quality': float
        }
    """
```

---

## 3. Data Schemas

### 3.1 ResearchAgent Schema

```python
from dataclasses import dataclass

@dataclass
class ResearchAgent:
    """ORCHEX research agent representation"""
    agent_id: str
    agent_type: str  # "synthesis", "literature_review", etc.
    specialization: str
    skill_level: float  # 0.0 - 1.0
    current_workload: int
    max_tasks: int
    execution_history: List[Dict]
    model: str  # "claude-3-opus", "gpt-4-turbo", etc.

    def to_features(self) -> np.ndarray:
        """Convert to feature vector for Libria solvers"""
        return np.array([
            self.skill_level,
            self.current_workload / self.max_tasks,
            len(self.execution_history),
            hash(self.specialization) % 100 / 100.0,  # Normalized hash
            # Add more features as needed
        ])
```

### 3.2 Task Schema

```python
@dataclass
class ResearchTask:
    """ORCHEX research task representation"""
    task_id: str
    task_type: str  # "synthesis", "literature_review", etc.
    complexity: float  # 0.0 - 1.0
    priority: float  # 0.0 - 1.0
    deadline: float  # Hours remaining
    dependencies: List[str]  # Task IDs
    requirements: Dict[str, Any]

    def to_features(self) -> np.ndarray:
        """Convert to feature vector for Libria solvers"""
        return np.array([
            self.complexity,
            self.priority,
            self.deadline / 24.0,  # Normalize to days
            len(self.dependencies),
            # Add more features as needed
        ])
```

### 3.3 Agent State Schema (for Librex.Graph)

```python
@dataclass
class AgentState:
    """Agent state at a given timestep"""
    agent_id: str
    timestamp: float
    active_tasks: int
    recent_success_rate: float
    communication_frequency: float
    specialization_vector: np.ndarray
    workload_vector: np.ndarray

    def to_array(self) -> np.ndarray:
        """Convert to d-dimensional state vector"""
        return np.concatenate([
            [self.active_tasks / 10.0],  # Normalized
            [self.recent_success_rate],
            [self.communication_frequency],
            self.specialization_vector,
            self.workload_vector
        ])
```

---

## 4. Redis Blackboard Schema

Both ORCHEX and Libria share state via Redis:

### 4.1 Agent State Keys

```
# Agent metadata
ORCHEX:agent:{agent_id}:type = "synthesis"
ORCHEX:agent:{agent_id}:skill_level = 0.85
ORCHEX:agent:{agent_id}:workload = 3
ORCHEX:agent:{agent_id}:available = True

# Agent history (for learning)
ORCHEX:agent:{agent_id}:history = List[execution_record]

# Agent connections (from Librex.Graph topology)
ORCHEX:agent:{agent_id}:connections = Set[agent_ids]
```

### 4.2 Libria Solver State Keys

```
# Librex.QAP cost predictions
libria:qap:cost_matrix = serialized_matrix

# Librex.Meta Elo ratings
libria:meta:elo:{solver_id} = 1542.3

# Librex.Flow LinUCB parameters
libria:flow:A:{agent_id} = serialized_matrix
libria:flow:b:{agent_id} = serialized_vector

# Librex.Graph topology
libria:graph:topology = serialized_adjacency_matrix
libria:graph:fiedler_value = 0.342
```

### 4.3 Shared Execution Logs

```
# Execution records
execution:{execution_id}:agent = agent_id
execution:{execution_id}:task = task_id
execution:{execution_id}:duration = 125.3
execution:{execution_id}:success = True
execution:{execution_id}:quality = 0.92
```

---

## 5. Implementation Checklist

### 5.1 For Claude Instance 2 (ORCHEX)

**Week 1**:
- [ ] Implement `ORCHEX-core/atlas_core/engine.py` with ATLASEngine class
- [ ] Implement `ORCHEX-core/atlas_core/agent.py` with ResearchAgent base class
- [ ] Create 5-10 initial research agents (synthesis, literature_review, hypothesis_gen, critical_analysis, validation)
- [ ] Implement Redis blackboard connection
- [ ] Create agent state tracking

**Week 2**:
- [ ] Implement dialectical workflows (thesis-antithesis-synthesis)
- [ ] Create quality gates
- [ ] Implement LibriaRouter integration (import from libria-integration/)
- [ ] Test agent-task assignment via Librex.QAP

**Week 3**:
- [ ] Implement workflow routing via Librex.Flow
- [ ] Add resource allocation via Librex.Alloc
- [ ] Test end-to-end: ORCHEX task → Libria solver → ORCHEX execution

**Week 4**:
- [ ] Add topology optimization via Librex.Graph
- [ ] Implement periodic topology updates
- [ ] Add workflow validation via Librex.Dual

**Week 5-6**:
- [ ] Expand to 40+ research agents
- [ ] Implement architecture evolution via Librex.Evo
- [ ] Full integration testing
- [ ] Performance benchmarking

### 5.2 For Claude Instance 1 (Libria)

**Week 1**:
- [ ] Implement `libria-core/libria_core/base.py` with LibriaSolver base class
- [ ] Implement `libria-integration/libria_integration/atlas_adapter.py` with LibriaRouter
- [ ] Create feature extraction helpers for ResearchAgent → features
- [ ] Implement Librex.Meta (PRIORITY for March 31)

**Week 2**:
- [ ] Implement Librex.QAP with agent-task assignment features
- [ ] Test integration with ORCHEX mock data
- [ ] Implement Librex.Flow with workflow routing

**Week 3-4**:
- [ ] Implement Librex.Alloc with resource allocation
- [ ] Implement Librex.Graph with topology optimization
- [ ] Test all integrations with ORCHEX Engine

**Week 5-12**:
- [ ] Complete remaining solvers (Librex.Dual, Librex.Evo)
- [ ] Benchmarking on ORCHEX workloads
- [ ] Publication preparation

---

## 6. Communication Protocol

### 6.1 Synchronization Points

**Daily Standups** (async via shared document):
- What did you implement yesterday?
- What will you implement today?
- Any blockers?
- Any API changes needed?

**Weekly Integration Tests**:
- Run full ORCHEX + Libria stack
- Verify all integration points working
- Performance benchmarking
- Update shared schemas if needed

### 6.2 Shared Files to Monitor

Both Claude instances should watch these files for changes:

```
shared/api-contracts/atlas_libria_interface.yaml
shared/schemas/agent_schema.json
shared/schemas/task_schema.json
shared/docs/INTEGRATION_CHANGES.md  # Log all API changes here
```

### 6.3 Conflict Resolution

If API needs to change:
1. Update `shared/docs/INTEGRATION_CHANGES.md` with proposed change
2. Other instance reviews and approves
3. Both instances update implementations
4. Run integration tests
5. Merge changes

---

## 7. Testing Strategy

### 7.1 Unit Tests (Each Instance)

**ORCHEX (Claude 2)**:
- Test ATLASEngine agent registration
- Test workflow execution
- Test LibriaRouter calls (with mocks)

**Libria (Claude 1)**:
- Test each solver independently
- Test LibriaRouter integration logic
- Test feature extraction from ORCHEX schemas

### 7.2 Integration Tests (Coordinated)

**Scenarios**:
1. **Agent Assignment**: ORCHEX creates 10 agents + 5 tasks → Librex.QAP assigns → ORCHEX executes
2. **Workflow Routing**: ORCHEX runs thesis-antithesis-synthesis → Librex.Flow routes each step
3. **Resource Allocation**: ORCHEX requests resources → Librex.Alloc allocates → ORCHEX applies limits
4. **Topology Optimization**: ORCHEX collects state history → Librex.Graph optimizes → ORCHEX updates graph
5. **End-to-End**: Full research workflow using all Libria solvers

### 7.3 Performance Benchmarks

**Metrics to Track**:
- Assignment quality (cost reduction vs. greedy)
- Routing confidence calibration (ECE)
- Resource utilization efficiency
- Topology algebraic connectivity
- End-to-end research task quality

---

## 8. Deployment

### 8.1 Docker Compose Startup

```bash
cd /mnt/c/Users/mesha/Downloads/Important
docker-compose up -d

# Verify services
docker-compose ps

# Check logs
docker-compose logs -f ORCHEX-core
docker-compose logs -f libria-api

# Access services
# ORCHEX: http://localhost:8000
# Libria: http://localhost:8001
# Grafana: http://localhost:3000
# Prometheus: http://localhost:9090
# Jupyter: http://localhost:8888
```

### 8.2 Health Checks

```bash
# Check Redis
redis-cli ping

# Check PostgreSQL
psql -h localhost -U libria -d libria -c "SELECT 1;"

# Check ORCHEX API
curl http://localhost:8000/health

# Check Libria API
curl http://localhost:8001/health
```

---

## 9. Example: Complete Integration Flow

```python
# ============================================
# In ORCHEX Engine (Claude Instance 2)
# ============================================

from atlas_core.engine import ATLASEngine, ResearchAgent
from libria_integration import LibriaRouter

# Initialize
ORCHEX = ATLASEngine(libria_enabled=True)
router = LibriaRouter()

# Register agents
for i in range(40):
    agent = ResearchAgent(
        agent_id=f"agent_{i}",
        agent_type=random.choice(["synthesis", "literature_review", "hypothesis_gen"]),
        specialization="machine_learning",
        skill_level=random.uniform(0.5, 1.0),
        max_tasks=5,
        model="claude-3-opus"
    )
    ORCHEX.register_agent(agent)

# Assign task using Librex.QAP
task = {
    'task_id': 'task_001',
    'task_type': 'synthesis',
    'complexity': 0.7,
    'priority': 0.9,
    'deadline': 24.0
}

assignment = router.solve_assignment(
    agents=list(ORCHEX.agents.values()),
    tasks=[task]
)

assigned_agent = ORCHEX.agents[assignment['agent_id']]

# Execute dialectical workflow using Librex.Flow routing
workflow_result = ORCHEX.execute_workflow(
    workflow_type="thesis_antithesis_synthesis",
    inputs={'topic': 'neural_architecture_search'}
)

# Optimize topology weekly using Librex.Graph
if week_number % 1 == 0:
    agent_states = ORCHEX.get_agent_state_history(window=100)
    optimal_topology = router.optimize_topology(agent_states)
    ORCHEX.apply_communication_topology(optimal_topology)

# ============================================
# In Libria Integration (Both Instances)
# ============================================

# libria-integration/libria_integration/atlas_adapter.py
# (This file created by Claude Instance 1)

class LibriaRouter:
    def __init__(self):
        from libria_qap import Librex.QAP
        from libria_flow import Librex.Flow
        from libria_graph import Librex.Graph

        self.qap_solver = Librex.QAP()
        self.flow_solver = Librex.Flow(n_agents=40)
        self.graph_solver = Librex.Graph(n_agents=40)

        # Train on historical ORCHEX data
        self._load_historical_data()

    def solve_assignment(self, agents, tasks):
        agent_features = self._extract_agent_features(agents)
        task_features = self._extract_task_features(tasks)
        context = self._get_system_context()

        result = self.qap_solver.solve({
            'agent_features': agent_features,
            'task_features': task_features,
            'context': context
        }, features_extracted=True)

        return {
            'agent_id': agents[result['assignment'][0]].agent_id,
            'assignment': result['assignment'],
            'cost': result['cost']
        }

    # ... (other methods as specified above)
```

---

## 10. Success Metrics

**Integration Quality**:
- ✅ All 6 integration points working end-to-end
- ✅ < 100ms latency for routing calls
- ✅ < 500ms latency for assignment calls
- ✅ > 95% uptime for Libria API

**Research Quality** (via ORCHEX):
- 20-30% improvement in task assignment quality (vs. greedy)
- 15-25% improvement in workflow routing quality
- 10-20% improvement in resource utilization
- 20-30% better agent communication (higher Fiedler value)

**Coordination Success**:
- Both Claude instances can develop in parallel
- < 1 day to resolve integration conflicts
- Shared test suite passes > 95%

---

**END OF INTEGRATION SPECIFICATION**

**Next Steps**:
1. Claude Instance 2: Begin ORCHEX Engine implementation (Week 1)
2. Claude Instance 1: Begin Libria core + Librex.Meta (Week 1)
3. Both: Daily sync via `shared/docs/INTEGRATION_CHANGES.md`
4. Week 2: First integration test (agent assignment)

---

**Document Owner**: Both Claude Instances
**Last Updated**: 2026-01-17
**Version**: 1.0
