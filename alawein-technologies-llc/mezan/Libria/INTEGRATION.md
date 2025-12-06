# Librex Suite - Integration Specifications

## Standard Solver Interface

### Base Solver Class
```python
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from dataclasses import dataclass
import json

@dataclass
class Solution:
    """Standard solution format for all solvers"""
    assignment: Any  # Solver-specific solution representation
    objective_value: float
    metadata: Dict[str, Any]
    execution_time: float
    iterations: int

    def to_dict(self) -> Dict:
        return {
            "assignment": self.assignment,
            "objective_value": self.objective_value,
            "metadata": self.metadata,
            "execution_time": self.execution_time,
            "iterations": self.iterations
        }

@dataclass
class Problem:
    """Standard problem format"""
    type: str  # "qap", "flow", "alloc", etc.
    data: Dict[str, Any]
    constraints: Optional[Dict[str, Any]] = None
    parameters: Optional[Dict[str, Any]] = None

class LibriaSolver(ABC):
    """Base class for all Librex solvers"""

    def __init__(self, mode: str = "hybrid", **kwargs):
        self.mode = mode
        self.config = kwargs
        self.performance_history = []

    @abstractmethod
    def solve(self, problem: Problem) -> Solution:
        """Core solving method"""
        pass

    @abstractmethod
    def validate(self, solution: Solution, problem: Problem) -> bool:
        """Validate solution feasibility"""
        pass

    def to_atlas_format(self, solution: Solution) -> Dict:
        """Convert to ORCHEX/TURING format"""
        return {
            "solver": self.__class__.__name__,
            "version": self.get_version(),
            "solution": solution.to_dict(),
            "timestamp": self.get_timestamp()
        }

    def get_version(self) -> str:
        return "1.0.0"

    def get_timestamp(self) -> str:
        from datetime import datetime
        return datetime.utcnow().isoformat()
```

## Input/Output Schemas

### Librex.QAP I/O Schema
```json
{
  "input_schema": {
    "type": "object",
    "properties": {
      "problem_type": {"const": "qap"},
      "cost_matrix": {
        "type": "array",
        "items": {"type": "array", "items": {"type": "number"}}
      },
      "synergy_matrix": {
        "type": "array",
        "description": "Optional synergy bonuses"
      },
      "conflict_matrix": {
        "type": "array",
        "description": "Optional conflict penalties"
      },
      "constraints": {
        "type": "object",
        "properties": {
          "must_assign": {"type": "array"},
          "cannot_assign": {"type": "array"}
        }
      }
    },
    "required": ["problem_type", "cost_matrix"]
  },
  "output_schema": {
    "type": "object",
    "properties": {
      "assignment": {
        "type": "array",
        "items": {"type": "integer"},
        "description": "Permutation vector"
      },
      "objective_value": {"type": "number"},
      "agent_task_pairs": {
        "type": "array",
        "items": {
          "type": "object",
          "properties": {
            "agent_id": {"type": "string"},
            "task_id": {"type": "string"},
            "cost": {"type": "number"}
          }
        }
      }
    }
  }
}
```

### Librex.Flow I/O Schema
```json
{
  "input_schema": {
    "type": "object",
    "properties": {
      "problem_type": {"const": "flow"},
      "stages": {
        "type": "array",
        "items": {
          "type": "object",
          "properties": {
            "id": {"type": "string"},
            "type": {"enum": ["designer", "critic", "validator", "refactorer"]},
            "required": {"type": "boolean"},
            "confidence_threshold": {"type": "number"}
          }
        }
      },
      "edges": {
        "type": "array",
        "items": {
          "type": "object",
          "properties": {
            "from": {"type": "string"},
            "to": {"type": "string"},
            "cost": {"type": "number"},
            "quality_gain": {"type": "number"}
          }
        }
      }
    }
  },
  "output_schema": {
    "type": "object",
    "properties": {
      "workflow": {
        "type": "array",
        "items": {"type": "string"},
        "description": "Ordered list of stages"
      },
      "skipped_stages": {"type": "array"},
      "total_cost": {"type": "number"},
      "expected_quality": {"type": "number"}
    }
  }
}
```

### Librex.Alloc I/O Schema
```json
{
  "input_schema": {
    "type": "object",
    "properties": {
      "problem_type": {"const": "allocation"},
      "agents": {
        "type": "array",
        "items": {
          "type": "object",
          "properties": {
            "id": {"type": "string"},
            "capability": {"type": "number"},
            "cost_per_unit": {"type": "number"},
            "performance_history": {"type": "array"}
          }
        }
      },
      "tasks": {
        "type": "array",
        "items": {
          "type": "object",
          "properties": {
            "id": {"type": "string"},
            "required_resources": {"type": "number"},
            "priority": {"type": "number"},
            "deadline": {"type": "string"}
          }
        }
      },
      "budget": {"type": "number"}
    }
  },
  "output_schema": {
    "type": "object",
    "properties": {
      "allocations": {
        "type": "array",
        "items": {
          "type": "object",
          "properties": {
            "agent_id": {"type": "string"},
            "task_id": {"type": "string"},
            "resources": {"type": "number"}
          }
        }
      },
      "total_cost": {"type": "number"},
      "utilization": {"type": "number"}
    }
  }
}
```

## ORCHEX/TURING Adapter Template

```python
class ATLASAdapter:
    """Adapter for integrating Librex solvers with ORCHEX/TURING"""

    def __init__(self):
        self.solvers = {
            'qap': Librex.QAP(),
            'flow': Librex.Flow(),
            'alloc': Librex.Alloc(),
            'graph': Librex.Graph(),
            'meta': Librex.Meta(),
            'dual': Librex.Dual()
        }
        self.cache = {}

    def process_request(self, atlas_request: Dict) -> Dict:
        """Process ORCHEX request and route to appropriate solver"""

        # 1. Identify problem type
        problem_type = self._identify_problem_type(atlas_request)

        # 2. Transform to solver format
        problem = self._transform_to_solver_format(atlas_request, problem_type)

        # 3. Check cache
        cache_key = self._get_cache_key(problem)
        if cache_key in self.cache:
            return self.cache[cache_key]

        # 4. Select solver (can use Librex.Meta)
        solver = self._select_solver(problem_type, problem)

        # 5. Solve
        solution = solver.solve(problem)

        # 6. Validate with Librex.Dual
        if self.solvers['dual']:
            validated = self.solvers['dual'].validate_adversarial(solution, problem)
            if not validated:
                # Try alternative solver or parameters
                solution = self._try_alternative(problem, solver)

        # 7. Transform back to ORCHEX format
        atlas_response = self._transform_to_atlas_format(solution, problem_type)

        # 8. Cache result
        self.cache[cache_key] = atlas_response

        # 9. Update Librex.Meta with performance
        self._update_meta_learner(problem, solution, solver)

        return atlas_response

    def _identify_problem_type(self, request: Dict) -> str:
        """Identify which solver to use"""
        if 'agents' in request and 'tasks' in request:
            if 'synergy' in request or 'conflict' in request:
                return 'qap'
            elif 'resources' in request:
                return 'alloc'
        elif 'workflow' in request or 'stages' in request:
            return 'flow'
        elif 'network' in request or 'topology' in request:
            return 'graph'
        else:
            # Use Librex.Meta to decide
            return self.solvers['meta'].classify(request)

    def _transform_to_solver_format(self, request: Dict, problem_type: str) -> Problem:
        """Convert ORCHEX format to solver-specific format"""
        transformers = {
            'qap': self._transform_to_qap,
            'flow': self._transform_to_flow,
            'alloc': self._transform_to_alloc,
            'graph': self._transform_to_graph
        }
        return transformers[problem_type](request)

    def _transform_to_qap(self, request: Dict) -> Problem:
        """Transform to QAP format"""
        n_agents = len(request['agents'])
        n_tasks = len(request['tasks'])

        # Build cost matrix
        cost_matrix = [[0] * n_tasks for _ in range(n_agents)]
        for i, agent in enumerate(request['agents']):
            for j, task in enumerate(request['tasks']):
                cost_matrix[i][j] = self._compute_cost(agent, task)

        # Build synergy/conflict matrices if provided
        synergy_matrix = request.get('synergy_matrix', None)
        conflict_matrix = request.get('conflict_matrix', None)

        return Problem(
            type='qap',
            data={
                'cost_matrix': cost_matrix,
                'synergy_matrix': synergy_matrix,
                'conflict_matrix': conflict_matrix
            },
            constraints=request.get('constraints')
        )

    def _compute_cost(self, agent: Dict, task: Dict) -> float:
        """Compute cost of assigning agent to task"""
        base_cost = task.get('difficulty', 1.0) / agent.get('capability', 1.0)
        skill_match = self._compute_skill_match(agent, task)
        return base_cost * (2.0 - skill_match)

    def _compute_skill_match(self, agent: Dict, task: Dict) -> float:
        """Compute skill match between agent and task"""
        agent_skills = set(agent.get('skills', []))
        task_requirements = set(task.get('requirements', []))
        if not task_requirements:
            return 1.0
        return len(agent_skills & task_requirements) / len(task_requirements)
```

## Data Flow Diagrams

### End-to-End Data Flow
```
User Request (JSON/YAML)
         ↓
┌─────────────────┐
│  ORCHEX Intake   │ - Validates request
│                 │ - Identifies problem types
└────────┬────────┘
         ↓
┌─────────────────┐
│  Task Splitter  │ - Decomposes into subproblems
│                 │ - Creates dependency graph
└────────┬────────┘
         ↓
┌─────────────────┐
│   Librex.Meta    │ - Selects solvers
│                 │ - Sets parameters
└────────┬────────┘
         ↓
     [Parallel]
    ╱    │    ╲
   ↓     ↓     ↓
┌────┐ ┌────┐ ┌────┐
│QAP │ │Flow│ │Alloc│ - Each solver works
│Lib │ │Lib │ │Lib  │   independently
└──┬─┘ └──┬─┘ └──┬──┘
   ↓     ↓     ↓
    ╲    │    ╱
     [Combine]
         ↓
┌─────────────────┐
│   Librex.Dual    │ - Adversarial validation
│                 │ - Robustness testing
└────────┬────────┘
         ↓
┌─────────────────┐
│  Result Merger  │ - Combines solutions
│                 │ - Resolves conflicts
└────────┬────────┘
         ↓
┌─────────────────┐
│  SSOT Update    │ - Updates knowledge base
│                 │ - Logs performance
└────────┬────────┘
         ↓
Response to User (JSON)
```

### Solver Communication Pattern
```
    Librex.Meta (Coordinator)
         ↑ ↓
    Performance
      Feedback
         ↑ ↓
┌────────────────────────┐
│                        │
│  ┌─────┐    ┌─────┐   │
│  │ QAP │←──→│Flow │   │ Solvers can
│  └─────┘    └─────┘   │ share partial
│     ↑ ↓      ↑ ↓      │ solutions
│  ┌─────┐    ┌─────┐   │
│  │Graph│←──→│Alloc│   │
│  └─────┘    └─────┘   │
│                        │
└────────────────────────┘
         ↑ ↓
     Librex.Dual
   (Validates all)
```

## CLI Integration

```bash
# Individual solver CLIs
Librex.QAP solve problem.json --mode hybrid --gpu
Librex.Flow optimize workflow.yaml --confidence 0.8
Librex.Alloc allocate resources.json --algorithm thompson

# Unified CLI
Librex solve problem.json --auto-select
Librex benchmark all --dataset qaplib
Librex integrate --with ORCHEX --config config.yaml

# ORCHEX/TURING integration
turing process request.json --solvers Librex
ORCHEX assign-agents --solver Librex.QAP --gpu
```

## Environment Variables

```bash
# Solver configuration
export Librex_HOME=/opt/Librex
export Librex.QAP_GPU_ENABLED=true
export Librex.Flow_CONFIDENCE_THRESHOLD=0.8
export Librex.Alloc_ALGORITHM=thompson
export Librex.Meta_LEARNING_RATE=0.01

# ORCHEX/TURING integration
export ATLAS_SOLVER_SUITE=Librex
export TURING_Librex_ENDPOINT=http://localhost:8080
export Librex_CACHE_SIZE=1000
export Librex_PARALLEL_SOLVERS=4

# Performance monitoring
export Librex_METRICS_ENABLED=true
export Librex_METRICS_ENDPOINT=http://prometheus:9090
```

## Docker Compose Integration

```yaml
version: '3.8'

services:
  Librex:
    image: Librex/suite:latest
    ports:
      - "8080:8080"
    environment:
      - MODE=server
      - SOLVERS=all
    volumes:
      - ./config:/opt/Librex/config
      - ./cache:/opt/Librex/cache
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  ORCHEX:
    image: turing/ORCHEX:latest
    depends_on:
      - Librex
    environment:
      - Librex_ENDPOINT=http://Librex:8080
      - SOLVER_SUITE=Librex

  turing:
    image: turing/platform:latest
    depends_on:
      - ORCHEX
      - Librex
    ports:
      - "3000:3000"
```

## Performance Monitoring

```python
class PerformanceMonitor:
    """Monitor solver performance for Librex.Meta"""

    def __init__(self):
        self.metrics = {
            'solve_times': [],
            'solution_quality': [],
            'solver_selections': {},
            'failure_rates': {}
        }

    def record_solve(self, solver: str, problem: Problem,
                    solution: Solution, time: float):
        """Record solver performance"""
        self.metrics['solve_times'].append({
            'solver': solver,
            'problem_size': self._get_problem_size(problem),
            'time': time,
            'timestamp': datetime.now()
        })

        self.metrics['solution_quality'].append({
            'solver': solver,
            'objective': solution.objective_value,
            'problem_type': problem.type
        })

        # Update solver selection counts
        self.metrics['solver_selections'][solver] = \
            self.metrics['solver_selections'].get(solver, 0) + 1

    def get_solver_recommendation(self, problem: Problem) -> str:
        """Recommend best solver based on history"""
        problem_features = self._extract_features(problem)

        # Find similar problems in history
        similar = self._find_similar_problems(problem_features)

        # Return solver with best performance on similar problems
        return self._best_solver_for_similar(similar)
```