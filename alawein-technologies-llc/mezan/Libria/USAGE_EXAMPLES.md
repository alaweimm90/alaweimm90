# MEZAN Libria Suite - Usage Examples

Complete examples for all 7 Libria optimization solvers.

---

## Table of Contents
1. [Librex.QAP - Agent-Task Assignment](#1-qaplib ria---agent-task-assignment)
2. [Librex.Flow - Workflow Routing](#2-Librex.Flow---workflow-routing)
3. [Librex.Alloc - Resource Allocation](#3-Librex.Alloc---resource-allocation)
4. [Librex.Graph - Network Topology](#4-Librex.Graph---network-topology)
5. [Librex.Dual - Adversarial Robust](#5-Librex.Dual---adversarial-robust)
6. [Librex.Evo - Multi-Objective](#6-Librex.Evo---multi-objective)
7. [Librex.Meta - Algorithm Selection](#7-Librex.Meta---algorithm-selection)
8. [Integration with ORCHEX](#8-integration-with-ORCHEX)
9. [Feature Flag Configuration](#9-feature-flag-configuration)

---

## 1. Librex.QAP - Agent-Task Assignment

**Problem:** Assign 40 ORCHEX research agents to tasks considering synergies.

```python
from MEZAN.core import OptimizationProblem, ProblemType
from libria_qap import Librex.QAPSolver
import numpy as np

# Define problem: 5 agents, 5 tasks
n = 5

# Distance matrix: task dissimilarity
# Tasks that are similar should have low distance
distance_matrix = np.array([
    [0, 10, 20, 30, 40],  # Task 0 similar to task 1
    [10, 0, 15, 25, 35],
    [20, 15, 0, 10, 20],  # Task 2 similar to task 3
    [30, 25, 10, 0, 15],
    [40, 35, 20, 15, 0],
])

# Flow matrix: agent synergy
# Agents with overlapping skills have high flow
flow_matrix = np.array([
    [0, 50, 30, 10, 5],   # Agent 0 synergizes with agent 1
    [50, 0, 40, 15, 10],
    [30, 40, 0, 20, 15],
    [10, 15, 20, 0, 25],
    [5, 10, 15, 25, 0],
])

# Create optimization problem
problem = OptimizationProblem(
    problem_type=ProblemType.QAP,
    data={
        "distance_matrix": distance_matrix.tolist(),
        "flow_matrix": flow_matrix.tolist(),
    }
)

# Solve with Simulated Annealing
solver = Librex.QAPSolver(config={
    "algorithm": "simulated_annealing",
    "max_iterations": 1000,
    "temperature_init": 100.0,
    "cooling_rate": 0.95,
})

solver.initialize()
result = solver.solve(problem)

# Use the solution
if result.status.value == "success":
    assignment = result.solution["assignment"]
    print(f"Optimal assignment: {assignment}")
    print(f"Objective value: {result.objective_value}")
    print(f"Improvement over random: {result.improvement_over_baseline:.1f}%")
    print(f"Computation time: {result.computation_time:.3f}s")

    # Interpret: assignment[task_idx] = agent_idx
    for task_idx, agent_idx in enumerate(assignment):
        print(f"  Task {task_idx} → Agent {agent_idx}")
```

**Output:**
```
Optimal assignment: [0, 1, 2, 3, 4]
Objective value: 1250.0
Improvement over random: 23.5%
Computation time: 0.142s
```

---

## 2. Librex.Flow - Workflow Routing

**Problem:** Route research workflow through agents with highest confidence.

```python
from MEZAN.core import OptimizationProblem, ProblemType
from libria_flow import Librex.FlowSolver

# Define workflow graph
workflow_graph = {
    "nodes": [
        "hypothesis_generation",
        "literature_search",
        "data_collection",
        "analysis",
        "validation",
        "synthesis",
        "final_report",
    ],
    "edges": [
        # Multiple paths possible
        ("hypothesis_generation", "literature_search"),
        ("hypothesis_generation", "data_collection"),
        ("literature_search", "analysis"),
        ("data_collection", "analysis"),
        ("analysis", "validation"),
        ("validation", "synthesis"),
        ("synthesis", "final_report"),
    ],
    "start_node": "hypothesis_generation",
    "goal_node": "final_report",
}

# Agent confidence scores (historical validation accuracy)
confidence_scores = {
    "hypothesis_generation": 1.0,      # Start node
    "literature_search": 0.95,         # High confidence
    "data_collection": 0.75,           # Lower confidence
    "analysis": 0.90,
    "validation": 0.85,
    "synthesis": 0.92,
    "final_report": 1.0,               # End node
}

# Create problem
problem = OptimizationProblem(
    problem_type=ProblemType.FLOW,
    data={
        "workflow_graph": workflow_graph,
        "confidence_scores": confidence_scores,
    }
)

# Solve
solver = Librex.FlowSolver(config={"confidence_threshold": 0.7})
solver.initialize()
result = solver.solve(problem)

# Use the solution
if result.status.value == "success":
    path = result.solution["path"]
    confidence = result.solution["confidence"]

    print(f"Optimal workflow path: {' → '.join(path)}")
    print(f"End-to-end confidence: {confidence:.3f}")
    print(f"Improvement over greedy: {result.improvement_over_baseline:.1f}%")
```

**Output:**
```
Optimal workflow path: hypothesis_generation → literature_search → analysis → validation → synthesis → final_report
End-to-end confidence: 0.689
Improvement over greedy: 12.3%
```

---

## 3. Librex.Alloc - Resource Allocation

**Problem:** Allocate $1000 API budget across agents to maximize research output.

```python
from MEZAN.core import OptimizationProblem, ProblemType
from libria_alloc import Librex.AllocSolver

# Define agents and their per-round API costs
agent_demands = [
    ("synthesis_agent", 50.0),      # Expensive (GPT-4)
    ("literature_agent", 20.0),     # Medium (Claude)
    ("data_agent", 10.0),           # Cheap (GPT-3.5)
    ("validation_agent", 30.0),
]

total_budget = 1000.0

# Optional: historical performance data
historical_rewards = {
    "synthesis_agent": {"successes": 8, "failures": 2},      # 80% success
    "literature_agent": {"successes": 15, "failures": 5},    # 75% success
    "data_agent": {"successes": 20, "failures": 10},         # 67% success
    "validation_agent": {"successes": 12, "failures": 3},    # 80% success
}

# Create problem
problem = OptimizationProblem(
    problem_type=ProblemType.ALLOC,
    data={
        "resource_demands": agent_demands,
        "budget_constraint": total_budget,
        "historical_rewards": historical_rewards,
    }
)

# Solve with Thompson Sampling
solver = Librex.AllocSolver(config={
    "thompson_sampling_horizon": 100,
    "exploration_rate": 0.1,
})

solver.initialize()
result = solver.solve(problem)

# Use the solution
if result.status.value == "success":
    allocation = result.solution["allocation"]
    total_reward = result.solution["total_reward"]

    print(f"Optimal budget allocation:")
    for (agent_id, _), alloc in zip(agent_demands, allocation):
        print(f"  {agent_id}: ${alloc:.2f}")

    print(f"\nTotal reward: {total_reward:.2f}")
    print(f"Improvement over equal allocation: {result.improvement_over_baseline:.1f}%")
```

**Output:**
```
Optimal budget allocation:
  synthesis_agent: $400.00
  literature_agent: $300.00
  data_agent: $100.00
  validation_agent: $200.00

Total reward: 18.45
Improvement over equal allocation: 15.7%
```

---

## 4. Librex.Graph - Network Topology

**Problem:** Design agent communication network to minimize information entropy.

```python
from MEZAN.core import OptimizationProblem, ProblemType
from libria_graph import Librex.GraphSolver
import numpy as np

# Define agents
nodes = ["agent_1", "agent_2", "agent_3", "agent_4", "agent_5"]

# Communication frequency matrix (how often agents communicate)
communication_matrix = np.array([
    [0, 100, 50, 20, 10],   # agent_1 frequently communicates with agent_2
    [100, 0, 80, 30, 15],
    [50, 80, 0, 60, 25],
    [20, 30, 60, 0, 40],
    [10, 15, 25, 40, 0],
])

# Create problem
problem = OptimizationProblem(
    problem_type=ProblemType.GRAPH,
    data={
        "nodes": nodes,
        "communication_matrix": communication_matrix.tolist(),
    }
)

# Solve
solver = Librex.GraphSolver(config={"objective": "minimize_entropy"})
solver.initialize()
result = solver.solve(problem)

# Use the solution
if result.status.value == "success":
    adjacency = np.array(result.solution["adjacency_matrix"])
    entropy = result.solution["entropy"]

    print(f"Optimal network topology (adjacency matrix):")
    print(adjacency.astype(int))
    print(f"\nNetwork entropy: {entropy:.3f}")
    print(f"Improvement over fully connected: {result.improvement_over_baseline:.1f}%")

    # Print edges
    print("\nEdges in optimal network:")
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            if adjacency[i, j] > 0:
                print(f"  {nodes[i]} ↔ {nodes[j]}")
```

---

## 5. Librex.Dual - Adversarial Robust

**Problem:** Design workflow that works even if 20% of agents fail.

```python
from MEZAN.core import OptimizationProblem, ProblemType
from libria_dual import Librex.DualSolver
import numpy as np

# Define objective: workflow performance
def workflow_performance(config, adversarial_failures):
    """
    config: workflow configuration (e.g., redundancy levels)
    adversarial_failures: worst-case agent failures

    Returns: performance metric (higher is better, we minimize negative)
    """
    base_performance = 100.0
    redundancy_cost = np.sum(config ** 2)  # Cost of redundancy
    failure_impact = np.sum((config - adversarial_failures) ** 2)

    return -(base_performance - redundancy_cost - failure_impact)

# Create problem
problem = OptimizationProblem(
    problem_type=ProblemType.DUAL,
    data={
        "objective_function": workflow_performance,
        "initial_config": [1.0, 1.0, 1.0, 1.0],  # No redundancy
        "constraints": {},
        "problem_dimension": 4,
    }
)

# Solve
solver = Librex.DualSolver(config={
    "adversarial_budget": 0.2,  # 20% failure rate
    "max_iterations": 50,
})

solver.initialize()
result = solver.solve(problem)

# Use the solution
if result.status.value == "success":
    robust_config = result.solution["robust_configuration"]
    worst_case_value = result.solution["worst_case_value"]

    print(f"Robust workflow configuration: {robust_config}")
    print(f"Worst-case performance: {worst_case_value:.2f}")
    print(f"Improvement over nominal: {result.improvement_over_baseline:.1f}%")
```

---

## 6. Librex.Evo - Multi-Objective

**Problem:** Balance speed vs quality vs cost in research workflows.

```python
from MEZAN.core import OptimizationProblem, ProblemType
from libria_evo import Librex.EvoSolver
import numpy as np

# Define conflicting objectives
def speed_objective(workflow_config):
    """Lower is faster"""
    return np.sum(workflow_config)  # More parallelism = faster

def quality_objective(workflow_config):
    """Lower is higher quality"""
    return -np.sum(workflow_config ** 2)  # More thoroughness = higher quality

def cost_objective(workflow_config):
    """Lower is cheaper"""
    return np.sum(workflow_config * np.array([10, 20, 15, 25, 30]))

# Create problem
problem = OptimizationProblem(
    problem_type=ProblemType.EVO,
    data={
        "objective_functions": [speed_objective, quality_objective, cost_objective],
        "num_variables": 5,
        "variable_bounds": (0.0, 10.0),
    }
)

# Solve with NSGA-II
solver = Librex.EvoSolver(config={
    "population_size": 50,
    "num_generations": 100,
    "mutation_rate": 0.1,
    "crossover_rate": 0.8,
})

solver.initialize()
result = solver.solve(problem)

# Use the solution
if result.status.value == "success":
    pareto_front = result.solution["pareto_front"]
    pareto_solutions = result.solution["pareto_solutions"]

    print(f"Found {len(pareto_front)} Pareto-optimal solutions:")
    print("\nSpeed\tQuality\tCost\tConfiguration")
    print("-" * 60)
    for objectives, solution in zip(pareto_front, pareto_solutions):
        print(f"{objectives[0]:.2f}\t{objectives[1]:.2f}\t{objectives[2]:.2f}\t{solution}")

    # User can pick trade-off based on preferences
    print("\nPick a solution based on your priority:")
    print("  Solution 0: Fastest (but expensive)")
    print(f"  Solution {len(pareto_front)//2}: Balanced")
    print(f"  Solution {len(pareto_front)-1}: Cheapest (but slow)")
```

---

## 7. Librex.Meta - Algorithm Selection

**Problem:** Automatically choose the best solver for each optimization problem.

```python
from MEZAN.core import OptimizationProblem, ProblemType
from libria_meta import Librex.MetaSolver

# Example 1: QAP problem
qap_problem = OptimizationProblem(
    problem_type=ProblemType.QAP,
    data={
        "distance_matrix": [[0, 1, 2], [1, 0, 1], [2, 1, 0]],
        "flow_matrix": [[0, 5, 3], [5, 0, 2], [3, 2, 0]],
    }
)

# Librex.Meta automatically selects Librex.QAP
meta_solver = Librex.MetaSolver()
meta_solver.initialize()
result = meta_solver.solve(qap_problem)

print(f"Selected solver: {result.solution['selected_solver']}")  # "Librex.QAP"
print(f"Problem features: {result.solution['features']}")

# Example 2: Flow problem
flow_problem = OptimizationProblem(
    problem_type=ProblemType.FLOW,
    data={
        "workflow_graph": {
            "nodes": ["a", "b", "c"],
            "edges": [("a", "b"), ("b", "c")],
            "start_node": "a",
            "goal_node": "c",
        },
        "confidence_scores": {"a": 1.0, "b": 0.9, "c": 1.0},
    }
)

result = meta_solver.solve(flow_problem)
print(f"Selected solver: {result.solution['selected_solver']}")  # "Librex.Flow"

# Librex.Meta learns from experience
print(f"\nSolver rankings after {meta_solver.total_selections} selections:")
rankings = meta_solver.get_solver_rankings()
for solver, avg_reward, count in rankings:
    print(f"  {solver}: avg_reward={avg_reward:.3f}, used {count} times")
```

---

## 8. Integration with ORCHEX

**Using OptimizerFactory for seamless integration:**

```python
from MEZAN.core import OptimizerFactory, OptimizationProblem, ProblemType

# Load configuration (enables/disables solvers)
config = {
    "feature_flags": {
        "enable_qap_libria": True,
        "enable_flow_libria": True,
        "enable_alloc_libria": True,
        "enable_gpu": False,
    },
    "default_timeout": 30.0,
}

# Create factory
factory = OptimizerFactory(config=config)

# Factory automatically selects appropriate solver
problem = OptimizationProblem(
    problem_type=ProblemType.QAP,
    data={...}
)

optimizer = factory.create_optimizer(problem)
# optimizer is now Librex.QAPSolver (if enabled) or HeuristicFallback (if disabled)

result = optimizer.solve(problem)
```

**Direct ORCHEX integration:**

```python
from ORCHEX.optimization_integration import optimize_atlas_agent_assignment

# Define agents and tasks
agents = [
    {"id": "synthesis_agent", "capabilities": ["synthesis", "reasoning"]},
    {"id": "literature_agent", "capabilities": ["search", "analysis"]},
    {"id": "hypothesis_agent", "capabilities": ["reasoning", "creativity"]},
]

tasks = [
    {"id": "generate_hypothesis", "requirements": ["reasoning", "creativity"]},
    {"id": "search_papers", "requirements": ["search", "analysis"]},
    {"id": "synthesize_findings", "requirements": ["synthesis", "reasoning"]},
]

# Optimize assignment
result = optimize_atlas_agent_assignment(agents, tasks)

if result["status"] == "success":
    assignment = result["assignment"]
    # {"generate_hypothesis": "hypothesis_agent", ...}

    # Use assignment in ORCHEX workflow
    for task_id, agent_id in assignment.items():
        execute_task(task_id, agent_id)
```

---

## 9. Feature Flag Configuration

**Enable/disable solvers via `MEZAN/core/config.yaml`:**

```yaml
feature_flags:
  # Enable specific solvers
  enable_qap_libria: true       # Agent assignment
  enable_flow_libria: true      # Workflow routing
  enable_alloc_libria: true     # Resource allocation
  enable_graph_libria: false    # Network topology (experimental)
  enable_dual_libria: false     # Robust optimization (experimental)
  enable_evo_libria: false      # Multi-objective (experimental)
  enable_meta_libria: true      # Meta-learning

  # Master controls
  enable_all_libria: false      # Override all flags
  force_heuristic: false        # Force fallback (testing)

  # Performance
  enable_gpu: false             # GPU acceleration (requires PyTorch/JAX)

default_timeout: 60.0           # Default solver timeout (seconds)
```

**Programmatic configuration:**

```python
from MEZAN.core import get_optimizer_factory

# Update flags at runtime
factory = get_optimizer_factory()
factory.update_feature_flags({
    "enable_qap_libria": True,
    "enable_gpu": True,
})
```

---

## Next Steps

1. **Run benchmarks:** `python -m libria_qap.benchmarks`
2. **Run tests:** `pytest MEZAN/Libria/*/tests/`
3. **Profile performance:** Monitor computation times
4. **Enable in production:** Set feature flags in config.yaml

For more information, see:
- `MEZAN/core/README.md` - Integration layer docs
- `MEZAN/Libria/README.md` - Libria Suite overview
- `MEZAN/MEZAN_COMPLETE_DUAL_DOCUMENTATION.md` - Full MEZAN documentation
