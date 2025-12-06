# MEZAN Core - Integration Layer

**Version:** 4.1.0 (Integration Layer)
**Status:** Production-Ready
**Purpose:** Bridge between V4.0.0 distributed infrastructure and Vision optimization solvers

---

## Overview

The **MEZAN Core Integration Layer** provides a clean abstraction between:
- **V4.0.0 Infrastructure:** DeepThink 3+1, Redis, Event Bus, API Gateway
- **Libria Suite Solvers:** QAP, Flow, Alloc, Graph, Dual, Evo, Meta

This layer enables:
1. **Incremental rollout** of Libria solvers via feature flags
2. **Graceful degradation** to heuristic fallbacks if solvers fail
3. **Pluggable architecture** for swapping optimization backends
4. **Unified interface** for all optimization problems in MEZAN

---

## Architecture

```
┌─────────────────────────────────────────────────┐
│         MEZAN V4.0.0 Infrastructure             │
│   (ORCHEX, Redis, Event Bus, API Gateway)       │
└────────────────┬────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────┐
│        MEZAN Core Integration Layer             │
│  ┌─────────────────────────────────────────┐   │
│  │ OptimizerInterface (Abstract Base Class)│   │
│  └─────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────┐   │
│  │   OptimizerFactory (Solver Selection)   │   │
│  │   - Feature Flags                       │   │
│  │   - Dynamic Loading                     │   │
│  │   - Fallback Logic                      │   │
│  └─────────────────────────────────────────┘   │
└────────────────┬────────────────────────────────┘
                 │
        ┌────────┴────────┐
        ▼                 ▼
┌──────────────┐  ┌──────────────┐
│   Libria     │  │  Heuristic   │
│   Solvers    │  │   Fallback   │
│ (7 solvers)  │  │ (Simple/Fast)│
└──────────────┘  └──────────────┘
```

---

## Components

### 1. `optimizer_interface.py`

Defines the abstract interface all solvers must implement:

**Key Classes:**
- `OptimizerInterface` - Abstract base class for all solvers
- `OptimizationProblem` - Input problem specification
- `OptimizationResult` - Solver output with metadata
- `ProblemType` - Enum of supported problem types
- `HeuristicFallbackOptimizer` - Simple fallback solver

**Example Usage:**
```python
from MEZAN.core import OptimizerInterface, OptimizationProblem, ProblemType

# Define problem
problem = OptimizationProblem(
    problem_type=ProblemType.QAP,
    data={
        "distance_matrix": [[0, 1, 2], [1, 0, 1], [2, 1, 0]],
        "flow_matrix": [[0, 5, 3], [5, 0, 2], [3, 2, 0]],
    }
)

# Solver implements OptimizerInterface
solver = MySolver()
solver.initialize()
result = solver.solve(problem)

print(f"Status: {result.status}")
print(f"Solution: {result.solution}")
print(f"Time: {result.computation_time}s")
```

### 2. `optimizer_factory.py`

Factory pattern for creating solvers based on problem type and feature flags:

**Key Classes:**
- `OptimizerFactory` - Creates and manages solvers
- `FeatureFlag` - Enum of feature flags for rollout control

**Example Usage:**
```python
from MEZAN.core import OptimizerFactory, OptimizationProblem, ProblemType

# Create factory with config
factory = OptimizerFactory(config={
    "feature_flags": {
        "enable_qap_libria": True,  # Enable Librex.QAP
        "enable_gpu": False,
    }
})

# Create problem
problem = OptimizationProblem(
    problem_type=ProblemType.QAP,
    data={...}
)

# Factory selects appropriate solver
optimizer = factory.create_optimizer(problem)
result = optimizer.solve(problem)
```

### 3. `config.yaml`

Configuration file for feature flags and solver settings:

**Key Sections:**
- `feature_flags` - Control which Libria solvers are enabled
- `solvers` - Solver-specific algorithm parameters
- `fallback` - Fallback behavior configuration
- `monitoring` - Prometheus metrics settings

**Example:**
```yaml
feature_flags:
  enable_qap_libria: true   # Enable Librex.QAP
  enable_gpu: false         # Disable GPU for now

solvers:
  qap_libria:
    algorithm: "simulated_annealing"
    max_iterations: 1000
```

---

## Supported Problem Types

| Problem Type | Description | Libria Solver | Status |
|--------------|-------------|---------------|--------|
| `QAP` | Agent-task assignment with synergies | Librex.QAP | 🔧 In Development |
| `FLOW` | Confidence-aware workflow routing | Librex.Flow | 📋 Planned |
| `ALLOC` | Budget-constrained resource allocation | Librex.Alloc | 📋 Planned |
| `GRAPH` | Agent network topology optimization | Librex.Graph | 📋 Planned |
| `DUAL` | Adversarial min-max robust optimization | Librex.Dual | 📋 Planned |
| `EVO` | Multi-objective evolutionary optimization | Librex.Evo | 📋 Planned |
| `META` | Meta-learning for algorithm selection | Librex.Meta | 📋 Planned |

---

## Feature Flags

The integration layer uses feature flags for gradual Libria solver rollout:

### Individual Solver Flags

```yaml
enable_qap_libria: false      # Enable Librex.QAP
enable_flow_libria: false     # Enable Librex.Flow
enable_alloc_libria: false    # Enable Librex.Alloc
enable_graph_libria: false    # Enable Librex.Graph
enable_dual_libria: false     # Enable Librex.Dual
enable_evo_libria: false      # Enable Librex.Evo
enable_meta_libria: false     # Enable Librex.Meta
```

### Master Control Flags

```yaml
enable_all_libria: false      # Enable ALL Libria solvers
force_heuristic: false        # Force heuristic fallback (testing)
enable_gpu: false             # Enable GPU acceleration
```

### Rollout Strategy

**Phase 1:** Feature flags all `false`, use heuristic fallbacks
- ✅ Validates integration layer works
- ✅ No risk to V4.0.0 production

**Phase 2:** Enable `enable_qap_libria: true`
- 🔧 Librex.QAP handles agent assignment
- 🔧 Fallback to heuristic if Librex.QAP fails

**Phase 3:** Enable additional solvers incrementally
- 📋 Librex.Flow, Librex.Alloc, etc.

**Phase 4:** Enable `enable_all_libria: true`
- 🎯 Full Vision + V4.0.0 integration

---

## Graceful Degradation

The integration layer ensures MEZAN always works, even if Libria solvers fail:

### Fallback Hierarchy

```
1. Try Libria Solver (if enabled via feature flag)
   ↓ (if fails or times out)
2. Fall back to HeuristicFallbackOptimizer
   ↓ (if even heuristic fails)
3. Log error and return None
```

### Fallback Behavior

```python
# In optimizer_factory.py
try:
    optimizer = LibriaSolver(...)
    result = optimizer.solve(problem)
    if result.status == SolverStatus.SUCCESS:
        return result  # Use Libria solution
except Exception:
    logger.warning("Libria solver failed, using heuristic fallback")

# Fall back to heuristic
heuristic = HeuristicFallbackOptimizer()
result = heuristic.solve(problem)
result.status = SolverStatus.FALLBACK
return result
```

---

## Integration with V4.0.0

### ORCHEX Agent Assignment

**Before (V4.0.0):**
```python
# ORCHEX/ORCHEX-core/src/ORCHEX/intelligent_mezan.py
def assign_agents_to_tasks(agents, tasks):
    # Simple heuristic: random assignment
    return random.shuffle(agents)
```

**After (with Integration Layer):**
```python
from MEZAN.core import OptimizerFactory, OptimizationProblem, ProblemType

def assign_agents_to_tasks(agents, tasks):
    # Create QAP problem
    problem = OptimizationProblem(
        problem_type=ProblemType.QAP,
        data={
            "distance_matrix": compute_agent_distances(agents),
            "flow_matrix": compute_task_flows(tasks),
        }
    )

    # Use optimizer factory
    factory = OptimizerFactory()
    optimizer = factory.create_optimizer(problem)
    result = optimizer.solve(problem)

    # Use solution (or fallback heuristic)
    return result.solution["assignment"]
```

### Redis Integration

```python
# Cache optimization results in Redis
import redis

def solve_with_caching(problem):
    r = redis.Redis()
    cache_key = f"optimizer:{problem.problem_type}:{hash(problem)}"

    # Check cache
    cached = r.get(cache_key)
    if cached:
        return deserialize(cached)

    # Solve
    optimizer = factory.create_optimizer(problem)
    result = optimizer.solve(problem)

    # Cache result
    r.setex(cache_key, 3600, serialize(result))  # 1 hour TTL
    return result
```

### Event Bus Notifications

```python
# Publish optimization events
from MEZAN.orchex.orchex-core.src.orchex.event_bus import publish_event

def solve_with_events(problem):
    optimizer = factory.create_optimizer(problem)

    # Publish start event
    publish_event("optimization.started", {
        "problem_type": problem.problem_type.value,
        "solver": optimizer.__class__.__name__,
    })

    result = optimizer.solve(problem)

    # Publish completion event
    publish_event("optimization.completed", {
        "problem_type": problem.problem_type.value,
        "status": result.status.value,
        "computation_time": result.computation_time,
    })

    return result
```

---

## Testing

### Unit Tests

```bash
cd MEZAN/core
pytest tests/test_optimizer_interface.py
pytest tests/test_optimizer_factory.py
```

### Integration Tests

```bash
# Test with V4.0.0 infrastructure
cd MEZAN/ORCHEX
pytest tests/test_mezan_integration.py
```

---

## Monitoring

The integration layer exposes Prometheus metrics:

### Metrics Exposed

- `mezan_optimization_requests_total` - Total optimization requests
- `mezan_optimization_duration_seconds` - Optimization computation time
- `mezan_optimization_status` - Success/Timeout/Failed/Fallback counts
- `mezan_solver_selection` - Which solver was used for each problem type
- `mezan_gpu_utilization` - GPU utilization (if enabled)

### Grafana Dashboard

See `MEZAN/monitoring/grafana-dashboard.json` for pre-built dashboard.

---

## Development

### Adding a New Libria Solver

1. **Implement `OptimizerInterface`:**

```python
# Libria/libria-new/src/libria_new/solver.py
from MEZAN.core import OptimizerInterface, OptimizationProblem, OptimizationResult

class NewLibriaSolver(OptimizerInterface):
    def initialize(self):
        # Initialize solver
        pass

    def solve(self, problem: OptimizationProblem) -> OptimizationResult:
        # Solve problem
        solution = my_algorithm(problem.data)
        return OptimizationResult(
            status=SolverStatus.SUCCESS,
            solution=solution,
            objective_value=compute_objective(solution),
            computation_time=elapsed,
            metadata={"algorithm": "my_algorithm"}
        )

    def get_problem_types(self):
        return [ProblemType.NEW]

    def estimate_complexity(self, problem):
        return "medium"
```

2. **Register in Factory:**

```python
# optimizer_factory.py
if self._is_enabled(FeatureFlag.ENABLE_NEW_LIBRIA):
    from libria_new import NewLibriaSolver
    self.solver_registry[ProblemType.NEW] = NewLibriaSolver
```

3. **Add Feature Flag:**

```yaml
# config.yaml
feature_flags:
  enable_new_libria: false
```

4. **Test:**

```bash
pytest tests/test_new_libria.py
```

---

## See Also

- **MEZAN/MEZAN_COMPLETE_DUAL_DOCUMENTATION.md** - Vision vs V4.0.0
- **MEZAN/Libria/README.md** - Libria Suite overview
- **MEZAN/ORCHEX/START_HERE.md** - ORCHEX orchestration
- **Librex/README.md** - General optimization framework

---

**Last Updated:** 2025-11-19
**Version:** 4.1.0
**Status:** ✅ Integration Layer Complete, Ready for Solver Implementation
