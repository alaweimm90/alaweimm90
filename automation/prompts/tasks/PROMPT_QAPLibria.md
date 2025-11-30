# QAPLibria Implementation Superprompt

**Version**: 1.0
**Target**: European Journal of Operational Research (EJOR) / INFORMS Journal on Computing
**Priority**: High (3 strong contributions)
**Status**: Ready for Implementation

---

## Executive Summary

QAPLibria implements contextual Quadratic Assignment Problem (QAP) solving with learned cost functions for multi-agent task assignment. Unlike classical QAP solvers that assume static cost matrices, QAPLibria learns agent-task affinity from historical execution data and uses spectral initialization + Sinkhorn projection for efficient assignment.

**Core Innovation**: Contextual QAP with learned costs (🟢 STRONG novelty)

**Research Contributions**:
- **QAP-C1**: Contextual QAP with Learned Cost Functions
- **QAP-C2**: Spectral Initialization for Assignment Problems
- **QAP-C3**: Online Learning for Dynamic Agent Reassignment

**Publication Strategy**: EJOR (IF: 6.4) or INFORMS Journal on Computing

---

## 1. Technical Specification

### 1.1 Problem Statement

**Classical QAP**:
Given:
- n agents, m tasks
- Flow matrix F (n × n): interaction strength between agents
- Distance matrix D (m × m): interaction cost between tasks

Find permutation π: {1,...,n} → {1,...,m} minimizing:
```
min ∑ᵢ ∑ⱼ F[i,j] × D[π(i), π(j)]
```

**Contextual QAP (QAPLibria)**:
Given:
- Agent features: A ∈ ℝⁿˣᵈ (skills, workload, history)
- Task features: B ∈ ℝᵐˣᵈ (complexity, requirements, deadline)
- Context: x ∈ ℝᵏ (system state, time, workload)
- Historical data: {(Aᵢ, Bᵢ, xᵢ, πᵢ, rᵢ)}ᵢ₌₁ᴺ

Learn cost function C(A, B, x) and find assignment minimizing:
```
min ∑ᵢ C(aᵢ, b_π(i), x)
```

where C is learned from historical execution outcomes.

### 1.2 Core Algorithm

**QAPLibria Architecture**:
```
Input: Agent features A, Task features B, Context x
│
├─► Cost Predictor: C(aᵢ, bⱼ, x) → ℝ
│   └─► Neural network or gradient boosting
│
├─► Spectral Initialization: X₀ = SpectralAlign(A, B)
│   └─► Eigendecomposition + Sinkhorn projection
│
├─► Frank-Wolfe Solver: X* = FW(C, X₀)
│   └─► Alternating linearization + Hungarian
│
└─► Output: Assignment matrix X* (or permutation π*)
```

### 1.3 Implementation

```python
import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
from scipy.linalg import eigh
from sklearn.ensemble import GradientBoostingRegressor
from typing import Dict, List, Optional, Tuple

class QAPLibria(LibriaSolver):
    """
    Contextual Quadratic Assignment Problem solver with learned costs

    Key Components:
    1. Cost Predictor: Learn C(agent, task, context) from historical data
    2. Spectral Initializer: Efficient initialization via eigendecomposition
    3. Frank-Wolfe Solver: Iterative refinement with Hungarian subroutine
    4. Online Updater: Continuous learning from new executions
    """

    def __init__(
        self,
        cost_predictor: str = "neural",  # "neural" or "gbdt"
        hidden_dim: int = 128,
        n_iterations: int = 50,
        convergence_tol: float = 1e-4,
        learning_rate: float = 0.001,
        spectral_rank: int = 10
    ):
        super().__init__()
        self.cost_predictor_type = cost_predictor
        self.hidden_dim = hidden_dim
        self.n_iterations = n_iterations
        self.convergence_tol = convergence_tol
        self.learning_rate = learning_rate
        self.spectral_rank = spectral_rank

        # Initialize cost predictor
        if cost_predictor == "neural":
            self.cost_model = NeuralCostPredictor(hidden_dim=hidden_dim)
        elif cost_predictor == "gbdt":
            self.cost_model = GBDTCostPredictor()
        else:
            raise ValueError(f"Unknown cost predictor: {cost_predictor}")

        # Training history
        self.history = []

    @property
    def name(self) -> str:
        return "QAPLibria"

    def fit(self, training_data: List[Dict]):
        """
        Train cost predictor on historical assignment data

        Args:
            training_data: List of dicts with keys:
                - 'agent_features': np.ndarray (n_agents × d_agent)
                - 'task_features': np.ndarray (n_tasks × d_task)
                - 'context': np.ndarray (d_context,)
                - 'assignment': np.ndarray (n_agents,) - π[i] = task assigned to agent i
                - 'costs': np.ndarray (n_agents,) - actual cost for each agent
        """
        print(f"Training QAPLibria on {len(training_data)} historical instances...")

        # Prepare training dataset
        X_train = []  # (agent_feat, task_feat, context)
        y_train = []  # actual costs

        for data in training_data:
            agent_feats = data['agent_features']
            task_feats = data['task_features']
            context = data['context']
            assignment = data['assignment']
            costs = data['costs']

            n_agents = len(agent_feats)
            for i in range(n_agents):
                j = assignment[i]  # Task assigned to agent i
                x = np.concatenate([agent_feats[i], task_feats[j], context])
                X_train.append(x)
                y_train.append(costs[i])

        X_train = np.array(X_train)
        y_train = np.array(y_train)

        # Train cost predictor
        self.cost_model.fit(X_train, y_train)

        print(f"✓ Cost predictor trained on {len(X_train)} agent-task pairs")

    def solve(
        self,
        instance: Dict,
        features_extracted: bool = False
    ) -> Dict[str, any]:
        """
        Solve contextual QAP instance

        Args:
            instance: Dict with keys:
                - 'agent_features': np.ndarray (n × d_agent)
                - 'task_features': np.ndarray (m × d_task)
                - 'context': np.ndarray (d_context,)
                OR
                - 'agents': List of agent objects (if features_extracted=False)
                - 'tasks': List of task objects (if features_extracted=False)

        Returns:
            result: Dict with keys:
                - 'assignment': np.ndarray (n,) - assignment[i] = task for agent i
                - 'cost': float - total assignment cost
                - 'cost_matrix': np.ndarray (n × m) - predicted costs
        """
        # Extract features if needed
        if not features_extracted:
            agent_feats = self._extract_agent_features(instance['agents'])
            task_feats = self._extract_task_features(instance['tasks'])
            context = self._extract_context(instance.get('context', {}))
        else:
            agent_feats = instance['agent_features']
            task_feats = instance['task_features']
            context = instance['context']

        n_agents = len(agent_feats)
        n_tasks = len(task_feats)

        # Predict cost matrix
        cost_matrix = self._predict_cost_matrix(agent_feats, task_feats, context)

        # Spectral initialization
        X0 = self._spectral_init(agent_feats, task_feats, rank=self.spectral_rank)

        # Frank-Wolfe solver
        X_star = self._frank_wolfe(cost_matrix, X0)

        # Extract assignment from X_star
        assignment = self._extract_assignment(X_star)

        # Compute total cost
        total_cost = sum(cost_matrix[i, assignment[i]] for i in range(n_agents))

        return {
            'assignment': assignment,
            'cost': total_cost,
            'cost_matrix': cost_matrix,
            'assignment_matrix': X_star
        }

    def _predict_cost_matrix(
        self,
        agent_feats: np.ndarray,
        task_feats: np.ndarray,
        context: np.ndarray
    ) -> np.ndarray:
        """
        Predict cost matrix C[i,j] = cost of assigning task j to agent i

        Args:
            agent_feats: (n × d_agent)
            task_feats: (m × d_task)
            context: (d_context,)

        Returns:
            C: (n × m) cost matrix
        """
        n_agents = len(agent_feats)
        n_tasks = len(task_feats)

        # Construct input features for all (agent, task) pairs
        X = []
        for i in range(n_agents):
            for j in range(n_tasks):
                x = np.concatenate([agent_feats[i], task_feats[j], context])
                X.append(x)

        X = np.array(X)

        # Predict costs
        costs = self.cost_model.predict(X)

        # Reshape to matrix
        C = costs.reshape(n_agents, n_tasks)

        return C

    def _spectral_init(
        self,
        agent_feats: np.ndarray,
        task_feats: np.ndarray,
        rank: int
    ) -> np.ndarray:
        """
        Spectral initialization using eigendecomposition + Sinkhorn projection

        Algorithm:
        1. Compute agent similarity matrix: A = agent_feats @ agent_feats.T
        2. Compute task similarity matrix: B = task_feats @ task_feats.T
        3. Eigendecompose A and B
        4. Align top-r eigenvectors: X0 = U_A @ diag(√λ_A ⊙ √λ_B) @ U_B.T
        5. Project to doubly stochastic: X0 ← Sinkhorn(X0)

        Args:
            agent_feats: (n × d)
            task_feats: (m × d)
            rank: truncation rank

        Returns:
            X0: (n × m) initial assignment matrix
        """
        # Similarity matrices
        A = agent_feats @ agent_feats.T  # (n × n)
        B = task_feats @ task_feats.T    # (m × m)

        # Eigendecomposition
        λ_A, U_A = eigh(A)
        λ_B, U_B = eigh(B)

        # Sort in descending order
        λ_A = λ_A[::-1]
        U_A = U_A[:, ::-1]
        λ_B = λ_B[::-1]
        U_B = U_B[:, ::-1]

        # Truncate to top-r
        r = min(rank, len(λ_A), len(λ_B))
        λ_A = λ_A[:r]
        U_A = U_A[:, :r]
        λ_B = λ_B[:r]
        U_B = U_B[:, :r]

        # Alignment weights
        weights = np.sqrt(np.abs(λ_A) * np.abs(λ_B))

        # Initial assignment matrix
        X0 = U_A @ np.diag(weights) @ U_B.T

        # Sinkhorn projection to doubly stochastic
        X0 = self._sinkhorn(X0, n_iters=100)

        return X0

    def _sinkhorn(
        self,
        X: np.ndarray,
        n_iters: int = 100,
        epsilon: float = 1e-6
    ) -> np.ndarray:
        """
        Sinkhorn-Knopp algorithm for doubly stochastic projection

        Args:
            X: (n × m) matrix
            n_iters: number of iterations
            epsilon: numerical stability

        Returns:
            X_proj: (n × m) doubly stochastic matrix
        """
        X = np.abs(X)  # Ensure non-negative
        X = X + epsilon  # Numerical stability

        for _ in range(n_iters):
            # Row normalization
            X = X / (np.sum(X, axis=1, keepdims=True) + epsilon)
            # Column normalization
            X = X / (np.sum(X, axis=0, keepdims=True) + epsilon)

        return X

    def _frank_wolfe(
        self,
        C: np.ndarray,
        X0: np.ndarray
    ) -> np.ndarray:
        """
        Frank-Wolfe algorithm for QAP

        Iteratively:
        1. Linearize objective around current X_t
        2. Solve linear assignment problem (Hungarian)
        3. Update X_{t+1} = (1 - γ_t) X_t + γ_t S_t

        Args:
            C: (n × m) cost matrix
            X0: (n × m) initial assignment

        Returns:
            X_star: (n × m) optimal assignment matrix
        """
        X = X0.copy()
        n_agents, n_tasks = C.shape

        for t in range(self.n_iterations):
            # Compute gradient: ∇f(X) ≈ C for linear objective
            grad = C

            # Solve linear assignment (Hungarian)
            row_ind, col_ind = linear_sum_assignment(grad)

            # Construct sparse assignment matrix
            S = np.zeros((n_agents, n_tasks))
            S[row_ind, col_ind] = 1.0

            # Step size (line search or fixed schedule)
            gamma = 2.0 / (t + 2.0)  # Standard FW schedule

            # Update
            X_new = (1 - gamma) * X + gamma * S

            # Check convergence
            if np.linalg.norm(X_new - X, 'fro') < self.convergence_tol:
                break

            X = X_new

        return X

    def _extract_assignment(self, X: np.ndarray) -> np.ndarray:
        """
        Extract discrete assignment from fractional matrix

        Args:
            X: (n × m) assignment matrix

        Returns:
            assignment: (n,) where assignment[i] = task for agent i
        """
        row_ind, col_ind = linear_sum_assignment(-X)  # Max weight matching
        assignment = np.zeros(len(row_ind), dtype=int)
        assignment[row_ind] = col_ind
        return assignment

    def update(
        self,
        instance: Dict,
        assignment: np.ndarray,
        actual_costs: np.ndarray
    ):
        """
        Online update: retrain cost predictor with new observations

        Args:
            instance: Problem instance (same format as solve())
            assignment: Executed assignment
            actual_costs: Observed costs for each agent
        """
        # Extract features
        if 'agent_features' not in instance:
            agent_feats = self._extract_agent_features(instance['agents'])
            task_feats = self._extract_task_features(instance['tasks'])
            context = self._extract_context(instance.get('context', {}))
        else:
            agent_feats = instance['agent_features']
            task_feats = instance['task_features']
            context = instance['context']

        # Store in history
        self.history.append({
            'agent_features': agent_feats,
            'task_features': task_feats,
            'context': context,
            'assignment': assignment,
            'costs': actual_costs
        })

        # Periodically retrain (e.g., every 100 instances)
        if len(self.history) % 100 == 0:
            print(f"Retraining cost predictor with {len(self.history)} instances...")
            self.fit(self.history)

    def _extract_agent_features(self, agents: List) -> np.ndarray:
        """Extract feature vector for each agent"""
        # Example features: skills, workload, success_rate, avg_latency
        features = []
        for agent in agents:
            feat = [
                agent.skill_level,
                agent.current_workload,
                agent.historical_success_rate,
                agent.avg_response_time,
                agent.capacity
            ]
            features.append(feat)
        return np.array(features)

    def _extract_task_features(self, tasks: List) -> np.ndarray:
        """Extract feature vector for each task"""
        # Example features: complexity, priority, deadline, dependencies
        features = []
        for task in tasks:
            feat = [
                task.complexity_score,
                task.priority,
                task.time_to_deadline,
                len(task.dependencies),
                task.estimated_duration
            ]
            features.append(feat)
        return np.array(features)

    def _extract_context(self, context: Dict) -> np.ndarray:
        """Extract global context features"""
        # Example: system load, time of day, failure rate
        return np.array([
            context.get('system_load', 0.5),
            context.get('hour_of_day', 12) / 24.0,
            context.get('recent_failure_rate', 0.0),
            context.get('queue_length', 0)
        ])


class NeuralCostPredictor(nn.Module):
    """Neural network for cost prediction"""

    def __init__(self, input_dim: int = None, hidden_dim: int = 128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.model = None  # Initialized after seeing first data

    def fit(self, X: np.ndarray, y: np.ndarray, n_epochs: int = 100):
        """Train neural network"""
        if self.model is None:
            input_dim = X.shape[1]
            self.model = nn.Sequential(
                nn.Linear(input_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(self.hidden_dim, 1)
            )

        # Convert to tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y).unsqueeze(1)

        # Training loop
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        for epoch in range(n_epochs):
            optimizer.zero_grad()
            y_pred = self.model(X_tensor)
            loss = criterion(y_pred, y_tensor)
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 20 == 0:
                print(f"  Epoch {epoch+1}/{n_epochs}, Loss: {loss.item():.4f}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict costs"""
        if self.model is None:
            raise ValueError("Model not trained yet")

        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            y_pred = self.model(X_tensor)
            return y_pred.numpy().flatten()


class GBDTCostPredictor:
    """Gradient Boosting Decision Tree for cost prediction"""

    def __init__(self):
        self.model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train GBDT"""
        self.model.fit(X, y)
        print(f"  GBDT trained, R² score: {self.model.score(X, y):.4f}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict costs"""
        return self.model.predict(X)
```

---

## 2. Research Validation

### 2.1 Novel Contributions

**QAP-C1: Contextual QAP with Learned Cost Functions**
- **Gap**: Classical QAP assumes static costs; no learning from historical data
- **Approach**: Learn C(agent, task, context) from execution outcomes
- **Impact**: 20-30% cost reduction via learned agent-task affinity

**QAP-C2: Spectral Initialization for Assignment Problems**
- **Gap**: Random initialization leads to poor local optima
- **Approach**: Spectral alignment of agent/task similarity matrices
- **Impact**: 10-15% faster convergence + better solution quality

**QAP-C3: Online Learning for Dynamic Agent Reassignment**
- **Gap**: Static assignments don't adapt to changing workloads
- **Approach**: Continuous cost model updates + periodic reassignment
- **Impact**: 15-20% improvement in dynamic environments

### 2.2 Baselines

1. **Classical QAP Solvers**:
   - RoTS (Robust Tabu Search)
   - Simulated Annealing
   - Genetic Algorithm
   - Branch-and-Bound (small instances)

2. **Learning-Based**:
   - Random Forest cost prediction + Hungarian
   - Neural assignment (Pointer Networks)
   - Reinforcement Learning (A3C)

3. **Heuristics**:
   - Greedy assignment
   - Hungarian algorithm (linear cost)
   - Load balancing heuristics

### 2.3 Benchmark Datasets

**QAPLIB**: 136 standard QAP instances
- Small: n < 20 (26 instances)
- Medium: 20 ≤ n < 50 (48 instances)
- Large: n ≥ 50 (62 instances)

**Multi-Agent Task Assignment** (custom dataset):
- ATLAS execution logs (500+ assignments)
- Agent features: skill, workload, latency
- Task features: complexity, deadline, dependencies
- Ground truth: actual execution costs

**Expected Performance**:
- QAPLIB: Within 5% of best known solution
- Multi-agent: 25% cost reduction vs. greedy baseline

---

## 3. Implementation Roadmap

### Phase 1: Core Algorithm (Weeks 1-3)

**Week 1**:
- [ ] Implement `QAPLibria` base class
- [ ] Implement `NeuralCostPredictor`
- [ ] Implement `GBDTCostPredictor`
- [ ] Unit tests for cost prediction

**Week 2**:
- [ ] Implement spectral initialization
- [ ] Implement Sinkhorn projection
- [ ] Unit tests for initialization
- [ ] Visualization of spectral alignment

**Week 3**:
- [ ] Implement Frank-Wolfe solver
- [ ] Hungarian algorithm integration (scipy)
- [ ] Convergence analysis
- [ ] End-to-end integration test

### Phase 2: Benchmarking (Weeks 4-6)

**Week 4**:
- [ ] Download QAPLIB instances
- [ ] Implement QAPLIB parser
- [ ] Implement baseline solvers (RoTS, SA, GA)
- [ ] Evaluation harness

**Week 5**:
- [ ] Run QAPLibria on QAPLIB (136 instances)
- [ ] Run baselines on QAPLIB
- [ ] Statistical comparison
- [ ] Performance plots

**Week 6**:
- [ ] Collect ATLAS execution logs
- [ ] Train cost predictor on historical data
- [ ] Evaluate on multi-agent task assignment
- [ ] Ablation studies

### Phase 3: Paper Writing (Weeks 7-9)

**Week 7**: Draft
- [ ] Introduction + Related Work
- [ ] Method section with algorithms

**Week 8**: Results
- [ ] QAPLIB results
- [ ] Multi-agent results
- [ ] Ablation studies

**Week 9**: Finalization
- [ ] Discussion + Conclusion
- [ ] Submission to EJOR or IJC

---

## 4. Integration with Libria Suite

### 4.1 ATLAS Integration

```python
# ATLAS workflow: Assign tasks to research agents
from atlas_engine import ATLASEngine
from libria_qap import QAPLibria

atlas = ATLASEngine()
qap_solver = QAPLibria()

# Train on historical assignments
historical_data = atlas.get_execution_logs(limit=500)
qap_solver.fit(historical_data)

@atlas.register_workflow("task_assignment")
def assign_tasks_to_agents(tasks: List[Task]) -> Dict[Agent, Task]:
    # Get available agents
    agents = atlas.get_available_agents()

    # Solve assignment problem
    instance = {
        'agents': agents,
        'tasks': tasks,
        'context': atlas.get_system_state()
    }

    result = qap_solver.solve(instance)
    assignment = result['assignment']

    # Map agents to tasks
    agent_task_map = {agents[i]: tasks[assignment[i]] for i in range(len(agents))}

    return agent_task_map
```

### 4.2 Online Learning

```python
# Update cost model after task execution
@atlas.on_task_complete
def update_qap_costs(task: Task, agent: Agent, execution_result: Dict):
    actual_cost = execution_result['cost']  # e.g., latency, error_rate
    assignment = [task.id]  # Simplified
    actual_costs = [actual_cost]

    qap_solver.update(
        instance={'agents': [agent], 'tasks': [task]},
        assignment=assignment,
        actual_costs=actual_costs
    )
```

---

## 5. Testing Protocol

### 5.1 Unit Tests

```python
# tests/test_qaplibria.py
import pytest
import numpy as np
from libria_qap import QAPLibria

def test_spectral_init():
    """Test spectral initialization"""
    agent_feats = np.random.randn(10, 5)
    task_feats = np.random.randn(15, 5)

    qap = QAPLibria()
    X0 = qap._spectral_init(agent_feats, task_feats, rank=5)

    assert X0.shape == (10, 15)
    assert np.allclose(np.sum(X0, axis=1), 1.0, atol=0.1)  # Row stochastic
    assert np.all(X0 >= 0)  # Non-negative

def test_sinkhorn_projection():
    """Test Sinkhorn projection to doubly stochastic"""
    X = np.random.rand(10, 10)
    qap = QAPLibria()
    X_proj = qap._sinkhorn(X, n_iters=100)

    # Check doubly stochastic
    assert np.allclose(np.sum(X_proj, axis=0), 1.0, atol=0.01)
    assert np.allclose(np.sum(X_proj, axis=1), 1.0, atol=0.01)

def test_frank_wolfe_convergence():
    """Test Frank-Wolfe convergence"""
    C = np.random.rand(10, 10)
    X0 = np.ones((10, 10)) / 10

    qap = QAPLibria(n_iterations=100)
    X_star = qap._frank_wolfe(C, X0)

    # Extract assignment
    assignment = qap._extract_assignment(X_star)

    # Verify valid assignment
    assert len(assignment) == 10
    assert len(set(assignment)) == 10  # All tasks assigned

def test_cost_prediction():
    """Test neural cost predictor"""
    X_train = np.random.randn(100, 10)
    y_train = np.random.rand(100)

    predictor = NeuralCostPredictor(hidden_dim=64)
    predictor.fit(X_train, y_train, n_epochs=50)

    X_test = np.random.randn(20, 10)
    y_pred = predictor.predict(X_test)

    assert len(y_pred) == 20
```

### 5.2 Integration Tests

```python
# tests/test_qap_integration.py
def test_end_to_end_assignment():
    """Test full QAPLibria pipeline"""
    # Mock training data
    training_data = [
        {
            'agent_features': np.random.randn(5, 3),
            'task_features': np.random.randn(5, 3),
            'context': np.random.randn(2),
            'assignment': np.array([0, 1, 2, 3, 4]),
            'costs': np.random.rand(5)
        }
        for _ in range(50)
    ]

    # Train QAPLibria
    qap = QAPLibria()
    qap.fit(training_data)

    # Solve new instance
    instance = {
        'agent_features': np.random.randn(5, 3),
        'task_features': np.random.randn(5, 3),
        'context': np.random.randn(2)
    }

    result = qap.solve(instance, features_extracted=True)

    assert 'assignment' in result
    assert 'cost' in result
    assert len(result['assignment']) == 5
```

---

## 6. Benchmark Setup

### 6.1 QAPLIB Download

```bash
# Download QAPLIB instances
wget https://qaplib.mgi.polymtl.ca/inst.tar.gz
tar -xzf inst.tar.gz

# Directory structure:
# qaplib/
#   chr12a.dat
#   chr15a.dat
#   ...
#   tai100a.dat
```

### 6.2 QAPLIB Parser

```python
# benchmark/qaplib_loader.py
import numpy as np
from pathlib import Path

def load_qaplib_instance(filepath: str) -> Dict:
    """
    Load QAPLIB instance from .dat file

    Returns:
        instance: {'n': int, 'F': np.ndarray, 'D': np.ndarray}
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()

    n = int(lines[0].strip())

    # Parse flow matrix F
    F_lines = []
    idx = 1
    while len(F_lines) < n:
        F_lines.extend(lines[idx].strip().split())
        idx += 1
    F = np.array([int(x) for x in F_lines]).reshape(n, n)

    # Parse distance matrix D
    D_lines = []
    while len(D_lines) < n * n:
        D_lines.extend(lines[idx].strip().split())
        idx += 1
    D = np.array([int(x) for x in D_lines]).reshape(n, n)

    return {'n': n, 'F': F, 'D': D}
```

### 6.3 Evaluation Harness

```python
# benchmark/evaluate_qap.py
def evaluate_qaplib(solver, qaplib_root="qaplib"):
    """Evaluate solver on QAPLIB benchmark"""
    instances = list(Path(qaplib_root).glob("*.dat"))

    results = []
    for inst_path in instances:
        instance = load_qaplib_instance(inst_path)
        result = solver.solve(instance)

        # Load best known solution
        bks = load_best_known_solution(inst_path.stem)

        results.append({
            'instance': inst_path.stem,
            'n': instance['n'],
            'objective': result['cost'],
            'bks': bks,
            'gap': (result['cost'] - bks) / bks * 100  # Percentage gap
        })

    return pd.DataFrame(results)
```

---

## 7. Publication Strategy

### 7.1 Target Venues

**Primary**: European Journal of Operational Research (EJOR)
- Impact Factor: 6.4
- Scope: Optimization, OR, decision sciences
- Timeline: 3-4 months review

**Backup**: INFORMS Journal on Computing (IJC)
- Impact Factor: 2.7
- Scope: Computational OR, algorithms

**Conference Option**: AAAI 2026

### 7.2 Paper Outline

1. Introduction (2 pages)
2. Related Work (1.5 pages)
   - Classical QAP algorithms
   - Learning-based optimization
   - Multi-agent task assignment
3. Method (4 pages)
   - Contextual QAP formulation
   - Cost learning (neural + GBDT)
   - Spectral initialization
   - Frank-Wolfe solver
4. Experiments (4 pages)
   - QAPLIB benchmark
   - Multi-agent task assignment
   - Ablation studies
5. Discussion (1 page)
6. Conclusion (0.5 pages)

**Appendix**: Convergence proof, hyperparameters

---

## 8. Code Repository Structure

```
libria-qap/
├── README.md
├── setup.py
├── requirements.txt
├── libria_qap/
│   ├── __init__.py
│   ├── qap_solver.py          # Main QAPLibria class
│   ├── cost_predictor.py      # Neural + GBDT predictors
│   ├── spectral_init.py       # Spectral initialization
│   ├── frank_wolfe.py         # FW solver
│   └── sinkhorn.py            # Sinkhorn projection
├── baselines/
│   ├── rots.py                # Robust Tabu Search
│   ├── simulated_annealing.py
│   ├── genetic_algorithm.py
│   └── hungarian.py
├── benchmark/
│   ├── qaplib_loader.py
│   ├── evaluator.py
│   └── visualization.py
├── tests/
│   ├── test_qap_solver.py
│   ├── test_spectral_init.py
│   └── test_integration.py
└── scripts/
    ├── download_qaplib.sh
    ├── train_qap.py
    └── run_benchmarks.py
```

---

## 9. Success Criteria

**QAPLIB Benchmark**:
- ✅ Within 5% of BKS on small instances (n < 20)
- ✅ Within 10% of BKS on medium instances (20 ≤ n < 50)
- ✅ Competitive with RoTS on large instances

**Multi-Agent Task Assignment**:
- ✅ 25% cost reduction vs. greedy baseline
- ✅ 15% improvement vs. Hungarian (linear cost)
- ✅ Faster than RoTS (< 10% overhead)

**Ablation Studies**:
- ✅ Spectral init improves over random by 10-15%
- ✅ Learned costs improve over static by 20-30%
- ✅ Online updates improve over offline by 5-10%

---

## 10. Next Actions

**Immediate**:
1. Set up repository
2. Implement core QAPLibria class
3. Implement cost predictors
4. Unit tests

**Short-term**:
1. Implement spectral initialization
2. Implement Frank-Wolfe solver
3. Download QAPLIB
4. Baseline implementations

**Mid-term**:
1. Run QAPLIB benchmarks
2. Multi-agent experiments
3. Statistical analysis

**Long-term**:
1. Write paper
2. Submit to EJOR
3. Open-source release

---

**END OF QAPLIBRIA SUPERPROMPT**

**Version**: 1.0
**Last Updated**: 2026-01-17
**Status**: Ready for Implementation
**Target**: Month 9 (September 2026)
