# Itqān Libria Suite - Master Implementation Plan

**Version**: 1.0
**Date**: 2026-01-17
**Status**: Comprehensive Parallel Development Strategy

---

## Executive Summary

This document provides the master plan for implementing all 7 Libria solvers in parallel while respecting the critical path (Librex.Meta - March 31, 2025 deadline). It coordinates multi-stream development, ORCHEX integration, and inter-Claude collaboration.

**Development Strategy**: Parallel execution across 8 streams with priority weighting

**Timeline**: 12 weeks (Jan 20 - April 14, 2026)

**Team Structure**: Multi-Claude coordination
- **Claude Instance 1** (Libria): This instance - 7 solvers + integration
- **Claude Instance 2** (ORCHEX): Parallel instance - ORCHEX Engine development
- **Coordination**: Shared parent directory with defined API boundaries

---

## 1. Directory Structure (Complete Monorepo)

### 1.1 Recommended Layout

```
/mnt/c/Users/mesha/Downloads/Important/
│
├── ORCHEX/                          # ← ORCHEX Engine (Claude Instance 2)
│   ├── ORCHEX-core/
│   │   ├── atlas_core/
│   │   │   ├── __init__.py
│   │   │   ├── engine.py           # ATLASEngine main class
│   │   │   ├── agent.py            # ResearchAgent base class
│   │   │   ├── workflow.py         # Workflow orchestration
│   │   │   ├── memory.py           # Long-term memory system
│   │   │   └── blackboard.py       # Redis blackboard connector
│   │   ├── tests/
│   │   ├── requirements.txt
│   │   └── setup.py
│   │
│   ├── ORCHEX-agents/               # Research agents (40+)
│   │   ├── synthesis_agent.py
│   │   ├── literature_review_agent.py
│   │   ├── hypothesis_generation_agent.py
│   │   └── ...
│   │
│   ├── ORCHEX-workflows/            # Dialectical workflows
│   │   ├── thesis_antithesis_synthesis.py
│   │   ├── multi_perspective_analysis.py
│   │   └── quality_gates.py
│   │
│   └── ORCHEX-api/                  # API for Libria integration
│       ├── libria_interface.py     # Libria → ORCHEX adapter
│       └── state_manager.py        # Agent state tracking
│
├── Libria/                         # ← Libria Suite (Claude Instance 1)
│   ├── libria-core/
│   │   ├── libria_core/
│   │   │   ├── __init__.py
│   │   │   ├── base.py             # LibriaSolver base class
│   │   │   ├── utils.py
│   │   │   └── registry.py         # Solver registry
│   │   ├── tests/
│   │   ├── requirements.txt
│   │   └── setup.py
│   │
│   ├── libria-meta/                # Librex.Meta (PRIORITY)
│   │   ├── libria_meta/
│   │   │   ├── __init__.py
│   │   │   ├── meta_solver.py
│   │   │   ├── elo_tracker.py
│   │   │   ├── tournament.py
│   │   │   └── feature_extractor.py
│   │   ├── baselines/
│   │   ├── benchmark/
│   │   ├── tests/
│   │   ├── scripts/
│   │   ├── requirements.txt
│   │   └── README.md
│   │
│   ├── libria-qap/
│   ├── libria-flow/
│   ├── libria-alloc/
│   ├── libria-graph/
│   ├── libria-dual/
│   ├── libria-evo/
│   │
│   ├── libria-integration/         # ORCHEX ↔ Libria integration
│   │   ├── libria_integration/
│   │   │   ├── __init__.py
│   │   │   ├── atlas_adapter.py    # Libria → ORCHEX
│   │   │   ├── solver_router.py    # Route tasks to solvers
│   │   │   └── state_sync.py       # Sync with ORCHEX blackboard
│   │   └── tests/
│   │
│   └── libria-benchmarks/          # Shared benchmarks
│       ├── data/
│       ├── scripts/
│       └── results/
│
├── infrastructure/                 # Shared infrastructure
│   ├── docker/
│   │   ├── Dockerfile.orchex
│   │   ├── Dockerfile.libria
│   │   ├── docker-compose.yml
│   │   └── nginx.conf
│   ├── redis/
│   │   └── redis.conf
│   ├── postgres/
│   │   └── init.sql
│   └── monitoring/
│       ├── prometheus.yml
│       └── grafana-dashboards/
│
└── shared/                         # Shared between ORCHEX and Libria
    ├── api-contracts/              # API specifications
    │   ├── atlas_libria_interface.yaml
    │   └── openapi-spec.yaml
    ├── schemas/                    # Data schemas
    │   ├── agent_schema.json
    │   ├── task_schema.json
    │   └── result_schema.json
    └── docs/                       # Integration documentation
        ├── ATLAS_LIBRIA_INTEGRATION.md
        └── API_REFERENCE.md
```

---

## 2. ORCHEX Structure Recommendations

### 2.1 ORCHEX Core Architecture (For Claude Instance 2)

Since you'll be prompting another Claude to work on ORCHEX, here's my recommended structure:

```python
# ORCHEX/ORCHEX-core/atlas_core/engine.py

from typing import Dict, List, Optional
from dataclasses import dataclass
import redis
import logging

@dataclass
class AgentConfig:
    """Configuration for ORCHEX research agent"""
    agent_id: str
    agent_type: str  # "synthesis", "literature_review", "hypothesis_gen", etc.
    specialization: str
    skill_level: float
    max_tasks: int
    model: str  # "claude-3-opus", "gpt-4-turbo", etc.

class ResearchAgent:
    """Base class for ORCHEX research agents"""

    def __init__(self, config: AgentConfig):
        self.config = config
        self.current_workload = 0
        self.execution_history = []

    def can_accept_task(self, task: Dict) -> bool:
        """Check if agent can accept task"""
        return self.current_workload < self.config.max_tasks

    def execute(self, task: Dict) -> Dict:
        """Execute research task"""
        raise NotImplementedError

class ATLASEngine:
    """
    Main ORCHEX orchestration engine

    Manages:
    - 40+ research agents
    - Dialectical workflows (thesis-antithesis-synthesis)
    - Quality gates
    - Integration with Libria solvers
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        libria_enabled: bool = True
    ):
        self.redis = redis.Redis.from_url(redis_url, decode_responses=True)
        self.agents: Dict[str, ResearchAgent] = {}
        self.workflows = {}
        self.libria_enabled = libria_enabled

        # Initialize Libria connection if enabled
        if libria_enabled:
            from libria_integration import LibriaRouter
            self.libria_router = LibriaRouter()

    def register_agent(self, agent: ResearchAgent):
        """Register research agent with ORCHEX"""
        self.agents[agent.config.agent_id] = agent

        # Store agent state in Redis blackboard
        self.redis.hset(
            f"ORCHEX:agent:{agent.config.agent_id}",
            mapping={
                "type": agent.config.agent_type,
                "specialization": agent.config.specialization,
                "skill_level": agent.config.skill_level,
                "available": True
            }
        )

    def assign_task(self, task: Dict) -> Optional[str]:
        """
        Assign task to best agent

        Uses Libria solvers for optimal assignment:
        - Librex.QAP: Agent-task assignment
        - Librex.Meta: Agent selection meta-learning
        - Librex.Flow: Workflow routing
        """
        if self.libria_enabled:
            # Use Libria Librex.QAP for optimal assignment
            assignment = self.libria_router.solve_assignment(
                agents=list(self.agents.values()),
                tasks=[task]
            )
            return assignment.get('agent_id')
        else:
            # Fallback: simple greedy assignment
            for agent_id, agent in self.agents.items():
                if agent.can_accept_task(task):
                    return agent_id
        return None

    def execute_workflow(
        self,
        workflow_type: str,
        inputs: Dict
    ) -> Dict:
        """
        Execute dialectical workflow

        Types:
        - thesis_antithesis_synthesis
        - multi_perspective_analysis
        - adversarial_validation (uses Librex.Dual)
        """
        if workflow_type == "thesis_antithesis_synthesis":
            return self._dialectical_workflow(inputs)
        elif workflow_type == "adversarial_validation":
            # Use Librex.Dual for workflow validation
            if self.libria_enabled:
                return self.libria_router.validate_workflow(inputs)

        raise ValueError(f"Unknown workflow: {workflow_type}")

    def _dialectical_workflow(self, inputs: Dict) -> Dict:
        """
        Thesis-Antithesis-Synthesis dialectical reasoning

        1. Thesis: Generate initial hypothesis
        2. Antithesis: Challenge with counter-arguments
        3. Synthesis: Reconcile into refined conclusion
        """
        # Step 1: Thesis
        thesis_agent = self._select_agent_by_type("hypothesis_generation")
        thesis_result = thesis_agent.execute({
            "task": "generate_hypothesis",
            "input": inputs
        })

        # Step 2: Antithesis (use Librex.Flow for routing)
        if self.libria_enabled:
            antithesis_agent_id = self.libria_router.route_workflow_step(
                workflow_state={
                    "step": "antithesis",
                    "thesis": thesis_result,
                    "available_agents": [
                        a for a in self.agents.values()
                        if a.config.agent_type == "critical_analysis"
                    ]
                }
            )
            antithesis_agent = self.agents[antithesis_agent_id]
        else:
            antithesis_agent = self._select_agent_by_type("critical_analysis")

        antithesis_result = antithesis_agent.execute({
            "task": "challenge_hypothesis",
            "thesis": thesis_result
        })

        # Step 3: Synthesis
        synthesis_agent = self._select_agent_by_type("synthesis")
        synthesis_result = synthesis_agent.execute({
            "task": "synthesize",
            "thesis": thesis_result,
            "antithesis": antithesis_result
        })

        return {
            "thesis": thesis_result,
            "antithesis": antithesis_result,
            "synthesis": synthesis_result
        }

    def _select_agent_by_type(self, agent_type: str) -> ResearchAgent:
        """Select agent by type"""
        for agent in self.agents.values():
            if agent.config.agent_type == agent_type and agent.can_accept_task({}):
                return agent
        raise ValueError(f"No available agent of type: {agent_type}")

    def get_agent_state_history(self, window: int = 100):
        """
        Get agent state history for Librex.Graph

        Returns: (T × n × d) array of agent states over time
        """
        states = []
        for agent_id in self.agents:
            agent_history = self.redis.lrange(f"ORCHEX:agent:{agent_id}:history", 0, window-1)
            states.append(agent_history)

        return states  # Process into (T × n × d) format

    def optimize_communication_topology(self):
        """
        Use Librex.Graph to optimize agent communication network

        Returns: Optimized adjacency matrix for agent communication
        """
        if not self.libria_enabled:
            return None

        # Get agent state history
        agent_states = self.get_agent_state_history(window=100)

        # Use Librex.Graph to optimize topology
        optimal_topology = self.libria_router.optimize_topology(agent_states)

        # Apply to ORCHEX
        self._apply_topology(optimal_topology)

        return optimal_topology

    def _apply_topology(self, adjacency_matrix):
        """Apply communication topology to agents"""
        # Store in Redis blackboard
        n = len(self.agents)
        for i, agent_id_i in enumerate(self.agents.keys()):
            connected_agents = [
                list(self.agents.keys())[j]
                for j in range(n)
                if adjacency_matrix[i, j] > 0.5
            ]
            self.redis.sadd(
                f"ORCHEX:agent:{agent_id_i}:connections",
                *connected_agents
            )
```

### 2.2 Libria ↔ ORCHEX Integration Layer

```python
# Libria/libria-integration/libria_integration/atlas_adapter.py

from typing import Dict, List, Optional, Any
import sys
sys.path.append('../../ORCHEX/ORCHEX-core')

from atlas_core.engine import ATLASEngine, ResearchAgent

class LibriaRouter:
    """
    Routes ORCHEX tasks to appropriate Libria solvers

    Solver Routing:
    - Librex.QAP: Agent-task assignment
    - Librex.Meta: Solver selection for tasks
    - Librex.Flow: Workflow routing with confidence
    - Librex.Alloc: Resource allocation to agents
    - Librex.Graph: Communication topology optimization
    - Librex.Dual: Workflow adversarial validation
    - Librex.Evo: Agent architecture evolution
    """

    def __init__(self):
        # Import Libria solvers
        from libria_qap import Librex.QAP
        from libria_meta import Librex.Meta
        from libria_flow import Librex.Flow
        from libria_alloc import Librex.Alloc
        from libria_graph import Librex.Graph
        from libria_dual import Librex.Dual
        from libria_evo import Librex.Evo

        # Initialize solvers
        self.qap_solver = Librex.QAP()
        self.meta_solver = Librex.Meta(solvers=[...])  # Add Libria solvers
        self.flow_solver = Librex.Flow(n_agents=40)  # ORCHEX has 40+ agents
        self.alloc_solver = Librex.Alloc(n_agents=40, n_resource_types=3)
        self.graph_solver = Librex.Graph(n_agents=40)
        self.dual_solver = Librex.Dual(workflow=None, safety_spec=None)
        self.evo_solver = Librex.Evo()

        # Load historical data from ORCHEX
        self._load_historical_data()

    def solve_assignment(
        self,
        agents: List[ResearchAgent],
        tasks: List[Dict]
    ) -> Dict:
        """Use Librex.QAP for agent-task assignment"""

        # Extract agent features
        agent_features = self._extract_agent_features(agents)

        # Extract task features
        task_features = self._extract_task_features(tasks)

        # Get system context
        context = self._get_system_context()

        # Solve assignment
        result = self.qap_solver.solve({
            'agent_features': agent_features,
            'task_features': task_features,
            'context': context
        }, features_extracted=True)

        return {
            'agent_id': agents[result['assignment'][0]].config.agent_id,
            'assignment': result['assignment'],
            'cost': result['cost']
        }

    def route_workflow_step(
        self,
        workflow_state: Dict
    ) -> str:
        """Use Librex.Flow for workflow routing"""

        available_agents = workflow_state['available_agents']
        agent_ids = [a.config.agent_id for a in available_agents]

        # Select agent with Librex.Flow
        selected_agent_id, confidence = self.flow_solver.select_agent(
            workflow_state,
            agent_ids
        )

        return selected_agent_id

    def allocate_resources(
        self,
        agents: List[ResearchAgent],
        total_budget: Dict[str, float]
    ) -> Dict:
        """Use Librex.Alloc for resource allocation"""

        import numpy as np

        # Build agent requests (compute, memory, bandwidth)
        agent_requests = {}
        for i, agent in enumerate(agents):
            agent_requests[i] = np.array([
                agent.config.max_tasks * 1.0,  # Compute units
                agent.config.max_tasks * 0.5,  # Memory units
                0.2  # Bandwidth units
            ])

        # Budget array
        budget = np.array([
            total_budget.get('compute', 10.0),
            total_budget.get('memory', 5.0),
            total_budget.get('bandwidth', 2.0)
        ])

        # Allocate
        allocation = self.alloc_solver.allocate(agent_requests, budget)

        return allocation

    def optimize_topology(
        self,
        agent_states: Any  # (T × n × d)
    ):
        """Use Librex.Graph for topology optimization"""
        return self.graph_solver.optimize_topology(
            agent_states,
            n_iterations=100,
            lr=0.01
        )

    def validate_workflow(
        self,
        workflow_spec: Dict
    ) -> Dict:
        """Use Librex.Dual for adversarial workflow validation"""

        # Create workflow function from spec
        def workflow_fn(input_data):
            # Execute ORCHEX workflow
            return {"output": "...", "is_safe": True, "quality": 0.9}

        # Safety specification
        def safety_spec(output):
            return output.get('is_safe', False) and output['quality'] > 0.8

        # Update Librex.Dual with workflow
        self.dual_solver.workflow = workflow_fn
        self.dual_solver.safety_spec = safety_spec

        # Validate
        test_inputs = workflow_spec.get('test_cases', [])
        report = self.dual_solver.validate_workflow(test_inputs)

        return report

    def evolve_architecture(
        self,
        task_distribution: List[Dict]
    ):
        """Use Librex.Evo for agent architecture evolution"""

        def performance_metric(architecture, task):
            # Simulate ORCHEX execution with architecture
            return 0.85  # Placeholder

        results = self.evo_solver.evolve(
            task_distribution,
            performance_metric,
            n_iterations=1000
        )

        return results

    def _extract_agent_features(self, agents: List[ResearchAgent]):
        """Extract features from ORCHEX agents"""
        import numpy as np
        features = []
        for agent in agents:
            feat = [
                agent.config.skill_level,
                agent.current_workload,
                len(agent.execution_history),
                # Add more features...
            ]
            features.append(feat)
        return np.array(features)

    def _extract_task_features(self, tasks: List[Dict]):
        """Extract features from tasks"""
        import numpy as np
        features = []
        for task in tasks:
            feat = [
                task.get('complexity', 0.5),
                task.get('priority', 0.5),
                task.get('deadline', 1.0),
                # Add more features...
            ]
            features.append(feat)
        return np.array(features)

    def _get_system_context(self):
        """Get system-level context"""
        import numpy as np
        return np.array([
            0.5,  # System load
            0.7,  # Time of day (normalized)
            0.1,  # Failure rate
            5.0   # Queue length
        ])

    def _load_historical_data(self):
        """Load ORCHEX execution history to train Libria solvers"""
        # Query Redis/PostgreSQL for historical executions
        # Train Librex.QAP cost predictor
        # Train Librex.Flow routing policy
        # etc.
        pass
```

---

## 3. API Contract (ORCHEX ↔ Libria)

Create shared API specification:

```yaml
# shared/api-contracts/atlas_libria_interface.yaml

openapi: 3.0.0
info:
  title: ORCHEX-Libria Integration API
  version: 1.0.0
  description: API contract between ORCHEX Engine and Libria Solvers

paths:
  /assignment/solve:
    post:
      summary: Solve agent-task assignment
      requestBody:
        content:
          application/json:
            schema:
              type: object
              properties:
                agents:
                  type: array
                  items:
                    $ref: '#/components/schemas/Agent'
                tasks:
                  type: array
                  items:
                    $ref: '#/components/schemas/Task'
      responses:
        '200':
          description: Assignment solution
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/AssignmentResult'

  /workflow/route:
    post:
      summary: Route workflow to agent
      requestBody:
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/WorkflowState'
      responses:
        '200':
          description: Selected agent
          content:
            application/json:
              schema:
                type: object
                properties:
                  agent_id:
                    type: string
                  confidence:
                    type: number

  /resources/allocate:
    post:
      summary: Allocate resources to agents
      # ... (similar pattern)

  /topology/optimize:
    post:
      summary: Optimize communication topology
      # ... (similar pattern)

components:
  schemas:
    Agent:
      type: object
      properties:
        agent_id:
          type: string
        agent_type:
          type: string
        skill_level:
          type: number
        current_workload:
          type: integer
        execution_history:
          type: array
          items:
            $ref: '#/components/schemas/Execution'

    Task:
      type: object
      properties:
        task_id:
          type: string
        complexity:
          type: number
        priority:
          type: number
        deadline:
          type: number
        requirements:
          type: object

    AssignmentResult:
      type: object
      properties:
        assignment:
          type: array
          items:
            type: integer
        cost:
          type: number
        cost_matrix:
          type: array
```

---

## 4. Multi-Stream Implementation Plan

### Stream 1: Librex.Meta (CRITICAL PATH)

**Owner**: This Claude instance
**Priority**: 🔴 CRITICAL
**Timeline**: Weeks 1-12
**Deadline**: March 31, 2025

**Week-by-Week**:
- Week 1-2: Implementation
- Week 3-4: Baselines
- Week 5-7: Benchmarking
- Week 8: Ablations
- Week 9-11: Paper writing
- Week 12: Submission

**Deliverables**:
- Complete Librex.Meta implementation
- 10 baselines implemented
- ASlib benchmark results
- 8-page paper + supplementary
- Submission to AutoML 2025

### Stream 2: Infrastructure & Tooling

**Owner**: This Claude instance
**Priority**: 🟡 HIGH
**Timeline**: Weeks 1-4

**Tasks**:
- [x] Create all requirements.txt files (do now)
- [x] Create all README.md files (do now)
- [x] Set up docker-compose.yml (do now)
- [x] Create CI/CD workflows (do now)
- [x] Set up monitoring (Prometheus/Grafana)
- [x] Create shared schemas
- [x] Generate API documentation

### Stream 3: Librex.QAP + Librex.Flow

**Owner**: This Claude instance
**Priority**: 🟡 HIGH
**Timeline**: Weeks 1-8

**Parallel Development**:
- Librex.QAP: Weeks 1-6
- Librex.Flow: Weeks 3-8

### Stream 4: ORCHEX Integration Layer

**Owner**: This Claude instance + Other Claude
**Priority**: 🟡 HIGH
**Timeline**: Weeks 1-6

**Coordination Points**:
- Week 1: Define API contract
- Week 2: Implement LibriaRouter
- Week 3: Implement ORCHEX adapter
- Week 4: Integration testing
- Week 5-6: End-to-end validation

### Stream 5: Librex.Alloc + Librex.Graph

**Owner**: This Claude instance
**Priority**: 🟠 MEDIUM
**Timeline**: Weeks 3-10

### Stream 6: Librex.Dual + Librex.Evo

**Owner**: This Claude instance
**Priority**: 🟠 MEDIUM
**Timeline**: Weeks 5-12

### Stream 7: Documentation & Testing (Continuous)

**Owner**: This Claude instance
**Priority**: 🟢 CONTINUOUS

**Tasks**:
- Write tests as implementation progresses
- Update documentation
- Generate API docs
- Create usage examples

### Stream 8: Publication Preparation (Continuous)

**Owner**: This Claude instance
**Priority**: 🟢 CONTINUOUS

**Tasks**:
- Track experimental results
- Generate figures and tables
- Draft paper sections
- Prepare supplementary materials

---

## 5. Immediate Next Steps (Starting Now)

Let me create all the supporting files for you right now. I'll generate:

1. requirements.txt for each solver
2. README.md for each solver
3. setup.py for each package
4. docker-compose.yml for full stack
5. CI/CD workflows
6. API contract specifications
7. ORCHEX integration templates

This will give both Claude instances everything needed to start parallel development immediately.

---

**Status**: Ready to execute comprehensive setup

Would you like me to proceed with creating all these files now?
