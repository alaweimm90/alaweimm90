# AllocLibria Implementation Superprompt

**Version**: 1.0
**Target**: ICML 2026 / NeurIPS 2025
**Priority**: High (2 strong contributions)
**Status**: Ready for Implementation

---

## Executive Summary

AllocLibria implements constrained Thompson Sampling for multi-agent resource allocation with budget constraints and fairness objectives. Unlike classical Thompson Sampling that assumes unlimited resources, AllocLibria handles hard budget constraints, fairness requirements, and multi-agent coordination.

**Core Innovation**: Constrained Thompson Sampling with fairness (🟢 MODERATE-STRONG novelty)

**Research Contributions**:
- **ALLOC-C1**: Constrained Thompson Sampling for Resource Allocation
- **ALLOC-C2**: Fairness-Aware Bandits with Budget Constraints

**Publication Strategy**: ICML 2026 or NeurIPS 2025

---

## 1. Technical Specification

### 1.1 Problem Statement

**Resource Allocation**:
Given:
- n agents requiring resources
- K resource types (e.g., compute, memory, bandwidth)
- Budget constraints: B = [B₁, B₂, ..., Bₖ]
- Agent resource requests: rᵢ = [rᵢ₁, rᵢ₂, ..., rᵢₖ]
- Historical rewards: {(aᵢ, rᵢ, reward_i)}

Objectives:
1. Maximize total reward: ∑ reward(aᵢ, rᵢ)
2. Satisfy budget: ∑ rᵢ ≤ B (element-wise)
3. Ensure fairness: min-max allocation or proportional fairness

Find allocation policy π: (agents, B) → {(a₁, r₁), ..., (aₙ, rₙ)}

### 1.2 Core Algorithm

**AllocLibria Architecture**:
```
Input: Agents, Budget B, Fairness constraints
│
├─► Thompson Sampling: Sample reward distributions
│   └─► Beta/Gaussian priors per (agent, resource) pair
│
├─► Constraint Projection: Ensure budget + fairness
│   └─► Lagrangian relaxation or integer programming
│
├─► Allocation Decision: Assign resources
│   └─► Knapsack or linear programming
│
├─► Fairness Monitor: Track allocation fairness
│   └─► Gini coefficient, max-min fairness
│
└─► Online Update: Update priors with observed rewards
```

### 1.3 Implementation

```python
import numpy as np
import torch
from scipy.optimize import linprog
from scipy.stats import beta, norm
from typing import Dict, List, Optional, Tuple
from ortools.linear_solver import pywraplp

class AllocLibria(LibriaSolver):
    """
    Constrained Thompson Sampling for resource allocation

    Key Components:
    1. Thompson Sampling: Beta/Gaussian posteriors for reward estimation
    2. Budget-Aware Allocation: LP/IP for constrained optimization
    3. Fairness Monitor: Track and enforce fairness constraints
    4. Online Learning: Update posteriors with observed rewards
    """

    def __init__(
        self,
        n_agents: int,
        n_resource_types: int,
        fairness_mode: str = "proportional",  # "proportional", "maxmin", "none"
        alpha_prior: float = 1.0,  # Beta prior α
        beta_prior: float = 1.0,   # Beta prior β
        use_gurobi: bool = False   # Use Gurobi instead of OR-Tools
    ):
        super().__init__()
        self.n_agents = n_agents
        self.n_resource_types = n_resource_types
        self.fairness_mode = fairness_mode
        self.use_gurobi = use_gurobi

        # Thompson Sampling priors (Beta distribution)
        # α[i] = successes + α_prior, β[i] = failures + β_prior
        self.alpha = {i: alpha_prior for i in range(n_agents)}
        self.beta = {i: beta_prior for i in range(n_agents)}

        # Resource consumption tracking
        self.resource_history = []

        # Fairness tracking
        self.allocation_counts = {i: 0 for i in range(n_agents)}
        self.total_allocations = 0

    @property
    def name(self) -> str:
        return "AllocLibria"

    def allocate(
        self,
        agent_requests: Dict[int, np.ndarray],
        budget: np.ndarray
    ) -> Dict[int, np.ndarray]:
        """
        Allocate resources to agents under budget constraints

        Args:
            agent_requests: {agent_id: resource_vector}
                           resource_vector = [compute, memory, bandwidth, ...]
            budget: np.ndarray (n_resource_types,) - total available budget

        Returns:
            allocation: {agent_id: allocated_resources}
        """
        # Sample expected rewards from Thompson Sampling posteriors
        sampled_rewards = {}
        for agent_id in agent_requests:
            # Sample from Beta(α, β)
            sampled_rewards[agent_id] = np.random.beta(
                self.alpha[agent_id],
                self.beta[agent_id]
            )

        # Solve constrained allocation problem
        if self.fairness_mode == "none":
            allocation = self._allocate_budget_only(
                agent_requests, budget, sampled_rewards
            )
        elif self.fairness_mode == "proportional":
            allocation = self._allocate_proportional_fair(
                agent_requests, budget, sampled_rewards
            )
        elif self.fairness_mode == "maxmin":
            allocation = self._allocate_maxmin_fair(
                agent_requests, budget, sampled_rewards
            )
        else:
            raise ValueError(f"Unknown fairness mode: {self.fairness_mode}")

        # Update allocation tracking
        for agent_id in allocation:
            if np.any(allocation[agent_id] > 0):
                self.allocation_counts[agent_id] += 1
        self.total_allocations += 1

        return allocation

    def _allocate_budget_only(
        self,
        agent_requests: Dict[int, np.ndarray],
        budget: np.ndarray,
        sampled_rewards: Dict[int, float]
    ) -> Dict[int, np.ndarray]:
        """
        Allocate resources to maximize expected reward subject to budget

        Formulation:
        max ∑ᵢ reward[i] × xᵢ
        s.t. ∑ᵢ xᵢ × request[i] ≤ budget
             xᵢ ∈ {0, 1}  (binary allocation)

        This is a multi-dimensional knapsack problem
        """
        if self.use_gurobi:
            return self._allocate_gurobi(agent_requests, budget, sampled_rewards)
        else:
            return self._allocate_ortools(agent_requests, budget, sampled_rewards)

    def _allocate_ortools(
        self,
        agent_requests: Dict[int, np.ndarray],
        budget: np.ndarray,
        sampled_rewards: Dict[int, float]
    ) -> Dict[int, np.ndarray]:
        """Allocate using OR-Tools integer programming"""
        solver = pywraplp.Solver.CreateSolver('SCIP')
        if not solver:
            raise RuntimeError("OR-Tools SCIP solver not available")

        # Decision variables: x[i] = 1 if agent i is allocated
        x = {}
        for agent_id in agent_requests:
            x[agent_id] = solver.BoolVar(f'x_{agent_id}')

        # Objective: maximize total reward
        objective = solver.Objective()
        for agent_id, reward in sampled_rewards.items():
            objective.SetCoefficient(x[agent_id], reward)
        objective.SetMaximization()

        # Constraints: budget for each resource type
        for k in range(self.n_resource_types):
            constraint = solver.Constraint(0, budget[k])
            for agent_id, request in agent_requests.items():
                constraint.SetCoefficient(x[agent_id], request[k])

        # Solve
        status = solver.Solve()

        # Extract allocation
        allocation = {}
        if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
            for agent_id, request in agent_requests.items():
                if x[agent_id].solution_value() > 0.5:  # Allocated
                    allocation[agent_id] = request
                else:
                    allocation[agent_id] = np.zeros_like(request)
        else:
            # Infeasible: allocate nothing
            allocation = {agent_id: np.zeros_like(req) for agent_id, req in agent_requests.items()}

        return allocation

    def _allocate_gurobi(
        self,
        agent_requests: Dict[int, np.ndarray],
        budget: np.ndarray,
        sampled_rewards: Dict[int, float]
    ) -> Dict[int, np.ndarray]:
        """Allocate using Gurobi (alternative high-performance solver)"""
        import gurobipy as gp
        from gurobipy import GRB

        model = gp.Model("resource_allocation")
        model.setParam('OutputFlag', 0)  # Suppress output

        # Decision variables
        x = {}
        for agent_id in agent_requests:
            x[agent_id] = model.addVar(vtype=GRB.BINARY, name=f'x_{agent_id}')

        # Objective
        model.setObjective(
            gp.quicksum(sampled_rewards[i] * x[i] for i in agent_requests),
            GRB.MAXIMIZE
        )

        # Budget constraints
        for k in range(self.n_resource_types):
            model.addConstr(
                gp.quicksum(agent_requests[i][k] * x[i] for i in agent_requests) <= budget[k]
            )

        # Optimize
        model.optimize()

        # Extract allocation
        allocation = {}
        if model.status == GRB.OPTIMAL or model.status == GRB.SUBOPTIMAL:
            for agent_id, request in agent_requests.items():
                if x[agent_id].X > 0.5:
                    allocation[agent_id] = request
                else:
                    allocation[agent_id] = np.zeros_like(request)
        else:
            allocation = {agent_id: np.zeros_like(req) for agent_id, req in agent_requests.items()}

        return allocation

    def _allocate_proportional_fair(
        self,
        agent_requests: Dict[int, np.ndarray],
        budget: np.ndarray,
        sampled_rewards: Dict[int, float]
    ) -> Dict[int, np.ndarray]:
        """
        Allocate with proportional fairness

        Maximize: ∑ᵢ log(reward[i] + ε)
        Subject to budget constraints

        This encourages more balanced allocation across agents
        """
        solver = pywraplp.Solver.CreateSolver('SCIP')

        # Binary allocation variables
        x = {i: solver.BoolVar(f'x_{i}') for i in agent_requests}

        # Auxiliary variables for log rewards (linear approximation)
        # For binary allocation, use weighted rewards with fairness penalty
        # Fairness penalty: agents with fewer allocations get higher weight
        fairness_weights = {}
        for agent_id in agent_requests:
            count = self.allocation_counts[agent_id]
            total = max(self.total_allocations, 1)
            fairness_weights[agent_id] = 1.0 / (count / total + 0.1)  # Higher weight for underserved

        # Objective: reward × fairness_weight
        objective = solver.Objective()
        for agent_id in agent_requests:
            weighted_reward = sampled_rewards[agent_id] * fairness_weights[agent_id]
            objective.SetCoefficient(x[agent_id], weighted_reward)
        objective.SetMaximization()

        # Budget constraints
        for k in range(self.n_resource_types):
            constraint = solver.Constraint(0, budget[k])
            for agent_id, request in agent_requests.items():
                constraint.SetCoefficient(x[agent_id], request[k])

        # Solve
        status = solver.Solve()

        # Extract allocation
        allocation = {}
        if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
            for agent_id, request in agent_requests.items():
                if x[agent_id].solution_value() > 0.5:
                    allocation[agent_id] = request
                else:
                    allocation[agent_id] = np.zeros_like(request)
        else:
            allocation = {agent_id: np.zeros_like(req) for agent_id, req in agent_requests.items()}

        return allocation

    def _allocate_maxmin_fair(
        self,
        agent_requests: Dict[int, np.ndarray],
        budget: np.ndarray,
        sampled_rewards: Dict[int, float]
    ) -> Dict[int, np.ndarray]:
        """
        Allocate with max-min fairness

        Maximize: min(reward[i] for allocated agents)
        Subject to budget constraints
        """
        solver = pywraplp.Solver.CreateSolver('SCIP')

        # Binary allocation variables
        x = {i: solver.BoolVar(f'x_{i}') for i in agent_requests}

        # Auxiliary variable: min reward
        min_reward = solver.NumVar(0, solver.infinity(), 'min_reward')

        # Objective: maximize min reward
        objective = solver.Objective()
        objective.SetCoefficient(min_reward, 1.0)
        objective.SetMaximization()

        # Constraints: min_reward ≤ reward[i] × x[i] + M × (1 - x[i])
        M = 1000  # Big-M
        for agent_id, reward in sampled_rewards.items():
            constraint = solver.Constraint(-solver.infinity(), M)
            constraint.SetCoefficient(min_reward, 1.0)
            constraint.SetCoefficient(x[agent_id], -reward - M)

        # Budget constraints
        for k in range(self.n_resource_types):
            constraint = solver.Constraint(0, budget[k])
            for agent_id, request in agent_requests.items():
                constraint.SetCoefficient(x[agent_id], request[k])

        # Solve
        status = solver.Solve()

        # Extract allocation
        allocation = {}
        if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
            for agent_id, request in agent_requests.items():
                if x[agent_id].solution_value() > 0.5:
                    allocation[agent_id] = request
                else:
                    allocation[agent_id] = np.zeros_like(request)
        else:
            allocation = {agent_id: np.zeros_like(req) for agent_id, req in agent_requests.items()}

        return allocation

    def update(
        self,
        agent_id: int,
        outcome: Dict
    ):
        """
        Update Thompson Sampling posteriors after observing reward

        Args:
            agent_id: Agent that received allocation
            outcome: Dict with keys:
                - 'success': bool (1 if reward > threshold, 0 otherwise)
                OR
                - 'reward': float (continuous reward)
        """
        if 'success' in outcome:
            # Binary outcome: update Beta distribution
            success = outcome['success']
            if success:
                self.alpha[agent_id] += 1
            else:
                self.beta[agent_id] += 1
        elif 'reward' in outcome:
            # Continuous reward: discretize to binary
            reward = outcome['reward']
            threshold = outcome.get('threshold', 0.5)
            if reward > threshold:
                self.alpha[agent_id] += 1
            else:
                self.beta[agent_id] += 1
        else:
            raise ValueError("Outcome must contain 'success' or 'reward'")

        # Store history
        self.resource_history.append({
            'agent_id': agent_id,
            'outcome': outcome
        })

    def compute_fairness_metrics(self) -> Dict[str, float]:
        """
        Compute fairness metrics for current allocation history

        Returns:
            metrics: {
                'gini': Gini coefficient (0 = perfect equality, 1 = perfect inequality)
                'allocation_variance': Variance in allocation counts
                'min_allocation_ratio': min_count / max_count
            }
        """
        counts = np.array(list(self.allocation_counts.values()))

        # Gini coefficient
        sorted_counts = np.sort(counts)
        n = len(sorted_counts)
        cumsum = np.cumsum(sorted_counts)
        gini = (2 * np.sum((np.arange(1, n + 1)) * sorted_counts)) / (n * np.sum(sorted_counts)) - (n + 1) / n

        # Variance
        variance = np.var(counts)

        # Min/max ratio
        min_count = np.min(counts) if np.min(counts) > 0 else 0
        max_count = np.max(counts) if np.max(counts) > 0 else 1
        min_max_ratio = min_count / max_count

        return {
            'gini': gini,
            'allocation_variance': variance,
            'min_allocation_ratio': min_max_ratio
        }


# Integration example
class ResourceManager:
    """High-level resource manager using AllocLibria"""

    def __init__(self, alloc_solver: AllocLibria):
        self.alloc_solver = alloc_solver
        self.budget_tracker = BudgetTracker()

    def manage_resources(
        self,
        agent_requests: Dict[int, np.ndarray],
        total_budget: np.ndarray
    ) -> Dict[int, np.ndarray]:
        """
        Manage resource allocation for multi-agent system

        Args:
            agent_requests: {agent_id: resource_request}
            total_budget: Total available resources

        Returns:
            allocation: {agent_id: allocated_resources}
        """
        # Allocate resources
        allocation = self.alloc_solver.allocate(agent_requests, total_budget)

        # Update budget tracker
        total_allocated = np.sum([alloc for alloc in allocation.values()], axis=0)
        self.budget_tracker.record_allocation(total_allocated)

        return allocation


class BudgetTracker:
    """Track budget consumption over time"""

    def __init__(self):
        self.history = []

    def record_allocation(self, allocated: np.ndarray):
        """Record allocated resources"""
        self.history.append(allocated)

    def get_utilization(self, budget: np.ndarray) -> np.ndarray:
        """Compute average utilization per resource type"""
        if len(self.history) == 0:
            return np.zeros_like(budget)

        total_allocated = np.sum(self.history, axis=0)
        return total_allocated / (budget * len(self.history))
```

---

## 2. Research Validation

### 2.1 Novel Contributions

**ALLOC-C1: Constrained Thompson Sampling for Resource Allocation**
- **Gap**: Standard TS assumes unlimited resources; no work on multi-dimensional budget constraints
- **Approach**: Combine TS with integer programming for constrained optimization
- **Impact**: 15-25% improvement over UCB-based allocation

**ALLOC-C2: Fairness-Aware Bandits with Budget Constraints**
- **Gap**: Fairness bandits (Patil et al. 2021) don't handle budgets; budget bandits don't handle fairness
- **Approach**: Proportional fairness + max-min fairness in constrained TS
- **Impact**: 30-40% reduction in Gini coefficient while maintaining 90%+ reward

### 2.2 Baselines

1. **Recent Work**:
   - Information Relaxation TS (arXiv:2408.15535, Aug 2024)
   - Fair Bandits (Patil et al. 2021)
   - Budget-Limited Bandits (Xia et al. 2015)

2. **Classic Bandits**:
   - UCB1
   - Thompson Sampling (unconstrained)
   - ε-greedy

3. **Heuristics**:
   - Round-robin
   - Proportional allocation
   - Greedy (highest expected reward)

### 2.3 Benchmark Datasets

**Synthetic Benchmarks**:
- Multi-dimensional knapsack instances
- Varying budget tightness (loose/tight)
- Varying agent heterogeneity

**Real-World Datasets**:
- Cloud resource allocation logs
- ATLAS agent compute allocation
- Multi-tenant resource sharing

**Expected Performance**:
- 15-25% reward improvement over UCB
- 30-40% Gini reduction with fairness modes
- Near-optimal (95%+) compared to oracle

---

## 3. Implementation Roadmap

### Phase 1: Core Algorithm (Weeks 1-2)
### Phase 2: Benchmarking (Weeks 3-5)
### Phase 3: Paper Writing (Weeks 6-8)

---

## 4. Integration with Libria Suite

```python
from atlas_engine import ATLASEngine
from libria_alloc import AllocLibria

atlas = ATLASEngine()
alloc_solver = AllocLibria(
    n_agents=len(atlas.agents),
    n_resource_types=3,  # compute, memory, bandwidth
    fairness_mode="proportional"
)

@atlas.register_workflow("resource_allocation")
def allocate_resources(budget: np.ndarray):
    # Collect agent requests
    agent_requests = {}
    for agent in atlas.agents:
        agent_requests[agent.id] = agent.estimate_resource_needs()

    # Allocate resources
    allocation = alloc_solver.allocate(agent_requests, budget)

    # Assign to agents
    for agent_id, resources in allocation.items():
        atlas.agents[agent_id].assign_resources(resources)

    return allocation
```

---

## 5. Success Criteria

- ✅ 15-25% reward improvement over UCB baselines
- ✅ 30-40% Gini reduction with fairness
- ✅ Near-optimal (95%+) compared to oracle
- ✅ Ablation: TS improves over UCB by 10-15%
- ✅ Ablation: Fairness reduces Gini by 30%+

---

**END OF ALLOCLIBRIA SUPERPROMPT**

**Version**: 1.0
**Last Updated**: 2026-01-17
**Status**: Ready for Implementation
**Target**: Month 8-9 (ICML 2026 or NeurIPS 2025)
