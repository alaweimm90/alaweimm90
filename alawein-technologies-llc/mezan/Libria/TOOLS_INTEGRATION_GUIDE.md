# Itqān Libria Suite: Tools Integration Guide

**Version**: 1.0.0
**Date**: November 14, 2025
**Purpose**: Comprehensive guide for integrating external tools and libraries into the Libria Suite

---

## Document Overview

This guide provides detailed integration instructions for all external tools and libraries used across the 7 Libria solvers. Each section includes:
- Tool purpose and when to use it
- Installation and setup
- Integration patterns with code examples
- Performance considerations
- Troubleshooting

---

## Table of Contents

1. [Core Dependencies](#1-core-dependencies)
2. [OR-Tools Integration](#2-or-tools-integration) (Librex.QAP, Librex.Alloc, Librex.Meta)
3. [Gurobi Integration](#3-gurobi-integration) (Alternative to OR-Tools)
4. [NetworkX Integration](#4-networkx-integration) (Librex.Graph)
5. [PyTorch Integration](#5-pytorch-integration) (Librex.Dual, Librex.Graph)
6. [scikit-learn Integration](#6-scikit-learn-integration) (All Solvers)
7. [Redis Integration](#7-redis-integration) (SSOT/Blackboard)
8. [PostgreSQL Integration](#8-postgresql-integration) (Execution Logs)
9. [Additional Libraries](#9-additional-libraries)
10. [Docker Integration](#10-docker-integration)

---

## 1. Core Dependencies

### 1.1 Python Environment

**Requirement**: Python 3.10+

**Setup**:
```bash
# Using pyenv (recommended)
pyenv install 3.10.12
pyenv local 3.10.12

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate  # Windows

# Upgrade pip
pip install --upgrade pip setuptools wheel
```

### 1.2 Package Management

**Using Poetry** (recommended for monorepo):
```bash
# Install Poetry
curl -sSL https://install.python-poetry.org | python3 -

# Initialize project
poetry init
poetry install

# Add dependency
poetry add numpy scipy scikit-learn
poetry add --group dev pytest pytest-cov
```

**Using pip + requirements.txt**:
```bash
# Install dependencies
pip install -r requirements.txt

# Generate requirements.txt
pip freeze > requirements.txt
```

### 1.3 Core Scientific Stack

**Installation**:
```bash
# NumPy, SciPy, pandas
pip install numpy==1.24.3 scipy==1.10.1 pandas==2.0.2

# Visualization
pip install matplotlib==3.7.1 seaborn==0.12.2

# Jupyter (for development)
pip install jupyter==1.0.0 ipython==8.14.0
```

**Example Usage**:
```python
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse.linalg import eigsh

# Example: Solve linear assignment problem
cost_matrix = np.random.rand(10, 10)
row_ind, col_ind = linear_sum_assignment(cost_matrix)
optimal_assignment = np.zeros_like(cost_matrix)
optimal_assignment[row_ind, col_ind] = 1
```

---

## 2. OR-Tools Integration

**Purpose**: Integer programming, linear assignment, constraint programming

**Used By**: Librex.QAP (assignment), Librex.Alloc (resource allocation), Librex.Meta (solver selection)

### 2.1 Installation

```bash
pip install ortools==9.6.2534
```

### 2.2 Linear Assignment (Hungarian Algorithm)

**Librex.QAP Usage**:
```python
from ortools.graph.python import min_cost_flow

def solve_linear_assignment(cost_matrix):
    """
    Solve linear assignment problem using OR-Tools

    Args:
        cost_matrix: n × m matrix of assignment costs

    Returns:
        assignment: n × m binary matrix (x_ij = 1 if agent i assigned to task j)
    """
    n, m = cost_matrix.shape

    # Create min cost flow solver
    smcf = min_cost_flow.SimpleMinCostFlow()

    # Add arcs from source to agents (supply = 1)
    source = 0
    for i in range(n):
        agent_node = i + 1
        smcf.add_arc_with_capacity_and_unit_cost(
            source, agent_node,
            capacity=1,  # Each agent assigned once
            unit_cost=0
        )

    # Add arcs from agents to tasks
    for i in range(n):
        agent_node = i + 1
        for j in range(m):
            task_node = n + j + 1
            smcf.add_arc_with_capacity_and_unit_cost(
                agent_node, task_node,
                capacity=1,
                unit_cost=int(cost_matrix[i, j] * 1000)  # Scale costs
            )

    # Add arcs from tasks to sink (demand = 1)
    sink = n + m + 1
    for j in range(m):
        task_node = n + j + 1
        smcf.add_arc_with_capacity_and_unit_cost(
            task_node, sink,
            capacity=1,
            unit_cost=0
        )

    # Set supply/demand
    smcf.set_nodes_supplies([n] + [0] * (n + m) + [-n])

    # Solve
    status = smcf.solve()

    if status != smcf.OPTIMAL:
        raise ValueError("No optimal solution found")

    # Extract assignment
    assignment = np.zeros((n, m), dtype=int)
    for arc in range(smcf.num_arcs()):
        if smcf.flow(arc) > 0:
            tail = smcf.tail(arc)
            head = smcf.head(arc)
            if 1 <= tail <= n and n + 1 <= head <= n + m:
                agent_idx = tail - 1
                task_idx = head - n - 1
                assignment[agent_idx, task_idx] = 1

    return assignment
```

### 2.3 Integer Programming for Resource Allocation

**Librex.Alloc Usage**:
```python
from ortools.linear_solver import pywraplp

def solve_resource_allocation(demands, budget, costs, fairness_min):
    """
    Solve constrained resource allocation

    max Σ utility_i · x_i
    s.t. Σ cost_i · x_i ≤ budget
         x_i ≥ fairness_min ∀i
         x_i ≤ demand_i ∀i
    """
    n = len(demands)

    # Create solver
    solver = pywraplp.Solver.CreateSolver('SCIP')  # Or 'GLOP' for LP

    # Decision variables
    x = [solver.NumVar(fairness_min, demands[i], f'x_{i}') for i in range(n)]

    # Objective: Maximize utility (assume utility = x_i for simplicity)
    objective = solver.Objective()
    for i in range(n):
        objective.SetCoefficient(x[i], 1.0)
    objective.SetMaximization()

    # Budget constraint
    budget_constraint = solver.Constraint(-solver.infinity(), budget)
    for i in range(n):
        budget_constraint.SetCoefficient(x[i], costs[i])

    # Solve
    status = solver.Solve()

    if status == pywraplp.Solver.OPTIMAL:
        allocation = [x[i].solution_value() for i in range(n)]
        return np.array(allocation)
    else:
        raise ValueError("No optimal solution found")
```

### 2.4 Constraint Programming for Tournament Scheduling

**Librex.Meta Usage**:
```python
from ortools.sat.python import cp_model

def schedule_tournament_rounds(solvers, num_rounds):
    """
    Schedule Swiss-system tournament rounds

    Constraints:
      - Each solver plays exactly once per round
      - Solvers with similar scores paired together
      - No solver plays same opponent twice
    """
    model = cp_model.CpModel()
    n = len(solvers)

    # Variables: match[r][i][j] = 1 if solver i plays solver j in round r
    match = {}
    for r in range(num_rounds):
        for i in range(n):
            for j in range(n):
                if i < j:  # Avoid duplicate pairs
                    match[(r, i, j)] = model.NewBoolVar(f'match_r{r}_i{i}_j{j}')

    # Constraint 1: Each solver plays exactly once per round
    for r in range(num_rounds):
        for i in range(n):
            model.Add(
                sum(match.get((r, min(i, j), max(i, j)), 0) for j in range(n) if j != i) == 1
            )

    # Constraint 2: No solver plays same opponent twice
    for i in range(n):
        for j in range(n):
            if i < j:
                model.Add(sum(match.get((r, i, j), 0) for r in range(num_rounds)) <= 1)

    # Solve
    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        schedule = []
        for r in range(num_rounds):
            round_matches = []
            for i in range(n):
                for j in range(n):
                    if i < j and solver.Value(match.get((r, i, j), 0)):
                        round_matches.append((solvers[i], solvers[j]))
            schedule.append(round_matches)
        return schedule
    else:
        raise ValueError("No feasible tournament schedule found")
```

### 2.5 Performance Optimization

**Tips**:
- Use `SCIP` for mixed-integer programming (free, high performance)
- Use `GLOP` for pure linear programming (faster than SCIP for LP)
- For very large problems, consider Gurobi (commercial, faster)
- Warm-start: Pass initial solution to solver

```python
# Warm-start example
solver = pywraplp.Solver.CreateSolver('SCIP')
# ... define variables and constraints ...

# Set initial values
for i in range(n):
    x[i].SetStartValue(initial_solution[i])

status = solver.Solve()
```

---

## 3. Gurobi Integration

**Purpose**: High-performance optimization (alternative to OR-Tools for large-scale problems)

**License**: Commercial (free academic license available)

### 3.1 Installation

```bash
# Install Gurobi
pip install gurobipy==10.0.3

# Activate license (requires license file)
grbgetkey [LICENSE_KEY]
```

### 3.2 Linear Assignment with Gurobi

```python
import gurobipy as gp
from gurobipy import GRB

def solve_assignment_gurobi(cost_matrix):
    """
    Solve assignment problem using Gurobi

    Typically 2-5× faster than OR-Tools for large instances (n > 100)
    """
    n, m = cost_matrix.shape

    # Create model
    model = gp.Model("assignment")
    model.setParam('OutputFlag', 0)  # Suppress output

    # Decision variables: x[i, j] = 1 if agent i assigned to task j
    x = model.addVars(n, m, vtype=GRB.BINARY, name="x")

    # Objective: Minimize total cost
    model.setObjective(
        gp.quicksum(cost_matrix[i, j] * x[i, j] for i in range(n) for j in range(m)),
        GRB.MINIMIZE
    )

    # Constraint: Each agent assigned to exactly one task
    for i in range(n):
        model.addConstr(gp.quicksum(x[i, j] for j in range(m)) == 1)

    # Constraint: Each task assigned to at most one agent (if n ≤ m)
    for j in range(m):
        model.addConstr(gp.quicksum(x[i, j] for i in range(n)) <= 1)

    # Solve
    model.optimize()

    if model.status == GRB.OPTIMAL:
        assignment = np.array([[x[i, j].X for j in range(m)] for i in range(n)])
        return assignment
    else:
        raise ValueError("No optimal solution found")
```

### 3.3 Quadratic Programming for QAP Relaxation

```python
def solve_qap_relaxation_gurobi(A, B, c, s, λ):
    """
    Solve QAP relaxation using Gurobi (quadratic objective)

    min Σ c_ij x_ij - λ Σ s_ik x_ij x_kj
    s.t. Σ_j x_ij = 1 ∀i
         Σ_i x_ij ≤ capacity_j ∀j
         x_ij ∈ [0, 1]
    """
    n, m = c.shape

    model = gp.Model("qap_relaxation")
    model.setParam('OutputFlag', 0)

    # Variables: x[i, j] continuous in [0, 1]
    x = model.addVars(n, m, lb=0.0, ub=1.0, name="x")

    # Quadratic objective
    obj = gp.QuadExpr()
    # Linear term
    for i in range(n):
        for j in range(m):
            obj += c[i, j] * x[i, j]

    # Quadratic term (synergy)
    for i in range(n):
        for j in range(m):
            for k in range(n):
                if k != i:
                    obj -= λ * s[i, k] * x[i, j] * x[k, j]

    model.setObjective(obj, GRB.MINIMIZE)

    # Constraints
    for i in range(n):
        model.addConstr(gp.quicksum(x[i, j] for j in range(m)) == 1)

    # Solve
    model.optimize()

    if model.status == GRB.OPTIMAL:
        X = np.array([[x[i, j].X for j in range(m)] for i in range(n)])
        return X
    else:
        raise ValueError("No optimal solution found")
```

### 3.4 When to Use Gurobi vs. OR-Tools

| Scenario | Recommended | Reason |
|----------|-------------|--------|
| Small problems (n < 50) | OR-Tools | Free, sufficient performance |
| Large problems (n > 100) | Gurobi | 2-5× faster, better scaling |
| Quadratic objectives | Gurobi | Native QP support |
| Academic use | Gurobi (free license) | Free for academics |
| Commercial use | OR-Tools or purchase Gurobi | OR-Tools free, Gurobi expensive |

---

## 4. NetworkX Integration

**Purpose**: Graph algorithms, topology optimization, visualization

**Used By**: Librex.Graph (topology optimization), Librex.Evo (workflow graphs)

### 4.1 Installation

```bash
pip install networkx==3.1
```

### 4.2 Graph Creation and Manipulation

```python
import networkx as nx

# Create empty graph
G = nx.Graph()  # Undirected
D = nx.DiGraph()  # Directed (for workflows)

# Add nodes and edges
G.add_nodes_from([1, 2, 3, 4, 5])
G.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 5), (5, 1)])

# Graph properties
print(f"Nodes: {G.number_of_nodes()}")
print(f"Edges: {G.number_of_edges()}")
print(f"Density: {nx.density(G)}")
print(f"Is connected: {nx.is_connected(G)}")

# Degree
degrees = dict(G.degree())
avg_degree = sum(degrees.values()) / len(degrees)
```

### 4.3 Spectral Graph Theory (Librex.Graph)

```python
import numpy as np

def compute_laplacian_eigenvalues(G):
    """
    Compute eigenvalues of graph Laplacian

    λ_2 (algebraic connectivity) measures how well-connected the graph is
    """
    L = nx.laplacian_matrix(G).toarray()
    eigenvalues, eigenvectors = np.linalg.eigh(L)

    # Sort eigenvalues (ascending)
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    return eigenvalues, eigenvectors

# Example: Compute algebraic connectivity
G = nx.karate_club_graph()  # Example graph
eigenvalues, _ = compute_laplacian_eigenvalues(G)
algebraic_connectivity = eigenvalues[1]  # λ_2
print(f"Algebraic connectivity: {algebraic_connectivity:.4f}")
```

### 4.4 Graph Algorithms

```python
# Shortest path
path = nx.shortest_path(G, source=1, target=5)
path_length = nx.shortest_path_length(G, source=1, target=5)

# Centrality measures
betweenness = nx.betweenness_centrality(G)
closeness = nx.closeness_centrality(G)
eigenvector_centrality = nx.eigenvector_centrality(G)

# Clustering
clustering_coeffs = nx.clustering(G)
avg_clustering = nx.average_clustering(G)

# Connected components
components = list(nx.connected_components(G))
largest_component = max(components, key=len)
```

### 4.5 Workflow DAG Operations (Librex.Evo)

```python
def validate_workflow_dag(workflow):
    """
    Validate that workflow is a valid DAG (no cycles)
    """
    if not nx.is_directed_acyclic_graph(workflow):
        raise ValueError("Workflow contains cycles")

    # Check for isolated nodes
    isolated = list(nx.isolates(workflow))
    if isolated:
        print(f"Warning: Isolated nodes {isolated}")

    return True

def topological_sort_workflow(workflow):
    """
    Topologically sort workflow (execution order)
    """
    try:
        return list(nx.topological_sort(workflow))
    except nx.NetworkXError:
        raise ValueError("Workflow contains cycles")

def workflow_depth(workflow):
    """
    Compute workflow depth (longest path in DAG)
    """
    return nx.dag_longest_path_length(workflow)
```

### 4.6 Visualization

```python
import matplotlib.pyplot as plt

def visualize_topology(G, title="Communication Topology"):
    """
    Visualize graph topology
    """
    pos = nx.spring_layout(G)  # Force-directed layout

    plt.figure(figsize=(10, 8))
    nx.draw_networkx_nodes(G, pos, node_size=500, node_color='lightblue')
    nx.draw_networkx_edges(G, pos, width=2.0, alpha=0.6)
    nx.draw_networkx_labels(G, pos, font_size=12)

    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('topology.png', dpi=300, bbox_inches='tight')
    plt.show()
```

---

## 5. PyTorch Integration

**Purpose**: Deep learning, gradient-based optimization, GPU acceleration

**Used By**: Librex.Dual (adversarial attack generation), Librex.Graph (neural MI estimation)

### 5.1 Installation

```bash
# CPU-only
pip install torch==2.0.1

# With CUDA 11.8 (for GPU acceleration)
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
```

### 5.2 Adversarial Attack Generation (Librex.Dual)

```python
import torch
import torch.nn.functional as F

def generate_adversarial_prompt(model, input_ids, target_loss, num_steps=100, lr=0.01):
    """
    Generate adversarial prompt using gradient-based optimization

    Args:
        model: LLM or workflow model
        input_ids: Token IDs of original prompt
        target_loss: Loss to maximize (e.g., error rate)
        num_steps: Optimization iterations
        lr: Learning rate

    Returns:
        adversarial_input_ids: Adversarial prompt token IDs
    """
    # Convert to embeddings (continuous optimization)
    embeddings = model.get_input_embeddings()(input_ids)

    # Make embeddings require grad
    embeddings = embeddings.detach().requires_grad_(True)

    optimizer = torch.optim.Adam([embeddings], lr=lr)

    for step in range(num_steps):
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs_embeds=embeddings)
        loss = target_loss(outputs)

        # Maximize loss (gradient ascent)
        (-loss).backward()
        optimizer.step()

        if step % 10 == 0:
            print(f"Step {step}: Loss = {loss.item():.4f}")

    # Convert embeddings back to token IDs (project to nearest token)
    embedding_matrix = model.get_input_embeddings().weight
    distances = torch.cdist(embeddings, embedding_matrix)
    adversarial_input_ids = distances.argmin(dim=-1)

    return adversarial_input_ids
```

### 5.3 Neural Mutual Information Estimation (Librex.Graph)

```python
import torch.nn as nn

class MutualInformationEstimator(nn.Module):
    """
    Neural estimator for mutual information I(X; Y)

    Uses MINE (Mutual Information Neural Estimation) approach
    """
    def __init__(self, input_dim_x, input_dim_y, hidden_dim=128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim_x + input_dim_y, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, y):
        """
        Args:
            x: Agent observations (batch_size, input_dim_x)
            y: Communication messages (batch_size, input_dim_y)

        Returns:
            MI estimate: I(X; Y)
        """
        # Joint distribution
        joint = torch.cat([x, y], dim=1)
        joint_scores = self.network(joint)

        # Marginal distribution (shuffle y)
        y_shuffle = y[torch.randperm(y.size(0))]
        marginal = torch.cat([x, y_shuffle], dim=1)
        marginal_scores = self.network(marginal)

        # MINE lower bound
        mi = joint_scores.mean() - torch.log(marginal_scores.exp().mean())

        return mi

def estimate_mi_neural(observations_x, observations_y, num_epochs=1000):
    """
    Estimate MI using neural network

    Args:
        observations_x: Agent X observations (N, d_x)
        observations_y: Agent Y observations (N, d_y)

    Returns:
        mi_estimate: I(X; Y)
    """
    x = torch.tensor(observations_x, dtype=torch.float32)
    y = torch.tensor(observations_y, dtype=torch.float32)

    model = MutualInformationEstimator(x.size(1), y.size(1))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        mi = model(x, y)
        (-mi).backward()  # Maximize MI
        optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch {epoch}: MI = {mi.item():.4f}")

    return mi.item()
```

### 5.4 GPU Acceleration

```python
# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Move model and data to GPU
model = model.to(device)
x = x.to(device)
y = y.to(device)

# Example: Batch processing on GPU
batch_size = 256
num_batches = len(dataset) // batch_size

for batch_idx in range(num_batches):
    batch_x = x[batch_idx * batch_size:(batch_idx + 1) * batch_size].to(device)
    batch_y = y[batch_idx * batch_size:(batch_idx + 1) * batch_size].to(device)

    # Forward pass (on GPU)
    output = model(batch_x, batch_y)

    # ... optimization step ...
```

### 5.5 Performance Tips

- **Use GPU for large models** (>1M parameters)
- **Batch processing**: Process multiple inputs simultaneously
- **Mixed precision training**: Use `torch.cuda.amp` for 2× speedup
- **DataLoader**: Use `torch.utils.data.DataLoader` with `num_workers > 0` for parallel data loading

---

## 6. scikit-learn Integration

**Purpose**: Machine learning models (regression, classification, clustering)

**Used By**: All solvers (cost prediction, performance modeling, feature engineering)

### 6.1 Installation

```bash
pip install scikit-learn==1.3.0
```

### 6.2 Cost Prediction (Librex.QAP)

```python
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

class ContextualCostPredictor:
    def __init__(self):
        self.model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        self.history = []

    def fit(self, X, y):
        """Train cost predictor"""
        self.model.fit(X, y)

    def predict(self, X):
        """Predict costs for new agent-task pairs"""
        return self.model.predict(X)

    def update_online(self, agent_id, task_id, context, actual_cost):
        """Online learning from new observation"""
        features = self.encode_features(agent_id, task_id, context)
        self.history.append((features, actual_cost))

        # Retrain every 20 observations
        if len(self.history) % 20 == 0:
            X = np.array([h[0] for h in self.history[-100:]])  # Last 100
            y = np.array([h[1] for h in self.history[-100:]])
            self.fit(X, y)

    def encode_features(self, agent_id, task_id, context):
        """Feature engineering"""
        return np.concatenate([
            agent_embeddings[agent_id],  # Pre-computed agent features
            task_embeddings[task_id],    # Pre-computed task features
            context_features(context)     # Context: complexity, priority, etc.
        ])

# Usage example
predictor = ContextualCostPredictor()

# Train on historical data
X_train, X_test, y_train, y_test = train_test_split(features, costs, test_size=0.2)
predictor.fit(X_train, y_train)

# Evaluate
y_pred = predictor.predict(X_test)
print(f"R² score: {r2_score(y_test, y_pred):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
```

### 6.3 Performance Modeling (Librex.Meta)

```python
from sklearn.ensemble import RandomForestRegressor

class SolverPerformanceModel:
    def __init__(self, solver_name):
        self.solver_name = solver_name
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )

    def fit(self, instance_features, runtimes):
        """
        Train performance model

        Args:
            instance_features: (N, d) array of problem instance features
            runtimes: (N,) array of solver runtimes
        """
        self.model.fit(instance_features, runtimes)

    def predict_runtime(self, instance_features):
        """Predict solver runtime on new instance"""
        return self.model.predict([instance_features])[0]

# Usage: Train one model per solver
models = {}
for solver in ["Librex.QAP", "Librex.Flow", "Librex.Alloc"]:
    model = SolverPerformanceModel(solver)
    model.fit(instance_features_train[solver], runtimes_train[solver])
    models[solver] = model

# Predict and select best solver
new_instance_features = extract_features(new_instance)
predicted_runtimes = {
    solver: model.predict_runtime(new_instance_features)
    for solver, model in models.items()
}
best_solver = min(predicted_runtimes, key=predicted_runtimes.get)
print(f"Best solver: {best_solver} (predicted runtime: {predicted_runtimes[best_solver]:.2f}s)")
```

### 6.4 Feature Engineering

```python
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def create_feature_pipeline(numerical_features, categorical_features):
    """
    Create preprocessing pipeline for features

    Args:
        numerical_features: List of numerical feature names
        categorical_features: List of categorical feature names
    """
    # Numerical: Standardize (mean=0, std=1)
    numerical_transformer = StandardScaler()

    # Categorical: One-hot encode
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    # Combine
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    return preprocessor

# Example
numerical_features = ['task_complexity', 'agent_workload', 'priority']
categorical_features = ['task_domain', 'agent_role']

preprocessor = create_feature_pipeline(numerical_features, categorical_features)
X_transformed = preprocessor.fit_transform(X_raw)
```

### 6.5 Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV

def tune_hyperparameters(model, X, y, param_grid):
    """
    Tune hyperparameters using grid search with cross-validation

    Args:
        model: scikit-learn model
        X: Features
        y: Targets
        param_grid: Dictionary of hyperparameter ranges

    Returns:
        best_model: Model with best hyperparameters
    """
    grid_search = GridSearchCV(
        model,
        param_grid,
        cv=5,  # 5-fold cross-validation
        scoring='neg_mean_squared_error',
        n_jobs=-1,  # Use all CPU cores
        verbose=1
    )

    grid_search.fit(X, y)

    print(f"Best hyperparameters: {grid_search.best_params_}")
    print(f"Best CV score: {-grid_search.best_score_:.4f}")

    return grid_search.best_estimator_

# Example
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15],
    'learning_rate': [0.01, 0.1, 0.2]
}

best_model = tune_hyperparameters(
    GradientBoostingRegressor(),
    X_train, y_train,
    param_grid
)
```

---

## 7. Redis Integration

**Purpose**: SSOT (Single Source of Truth) / Blackboard architecture

**Used By**: All solvers (shared state, pub/sub communication)

### 7.1 Installation

```bash
# Install Redis server
# Ubuntu/Debian
sudo apt-get install redis-server

# macOS
brew install redis

# Start Redis
redis-server

# Install Python client
pip install redis==4.6.0
```

### 7.2 SSOT/Blackboard Implementation

```python
import redis
import json

class LibriaBlackboard:
    """
    Shared state for all Libria solvers

    Pattern: Blackboard architecture for multi-agent coordination
    """
    def __init__(self, redis_url="redis://localhost:6379/0"):
        self.redis = redis.Redis.from_url(redis_url, decode_responses=True)
        self.pubsub = self.redis.pubsub()

    # Write operations
    def write_assignment(self, assignment):
        """Librex.QAP writes agent-task assignments"""
        key = "libria:assignment:current"
        value = json.dumps(assignment.tolist())  # Convert numpy array
        self.redis.set(key, value)
        self.publish("assignment_updated", {"assignment": assignment.tolist()})

    def write_workflow_route(self, route):
        """Librex.Flow writes workflow routing decisions"""
        key = "libria:workflow:route"
        value = json.dumps(route)
        self.redis.set(key, value)
        self.publish("route_updated", route)

    def write_resource_allocation(self, allocation):
        """Librex.Alloc writes resource allocations"""
        key = "libria:allocation:current"
        value = json.dumps(allocation)
        self.redis.set(key, value)
        self.publish("allocation_updated", allocation)

    def write_topology(self, topology):
        """Librex.Graph writes communication topology"""
        key = "libria:topology:current"
        # Store as adjacency list
        adj_list = {str(i): list(topology.neighbors(i)) for i in topology.nodes()}
        value = json.dumps(adj_list)
        self.redis.set(key, value)
        self.publish("topology_updated", adj_list)

    # Read operations
    def read_assignment(self):
        """Read current assignment"""
        value = self.redis.get("libria:assignment:current")
        return json.loads(value) if value else None

    def read_workflow_route(self):
        """Read current workflow route"""
        value = self.redis.get("libria:workflow:route")
        return json.loads(value) if value else None

    def read_resource_allocation(self):
        """Read current resource allocation"""
        value = self.redis.get("libria:allocation:current")
        return json.loads(value) if value else None

    def read_topology(self):
        """Read current communication topology"""
        value = self.redis.get("libria:topology:current")
        if value:
            adj_list = json.loads(value)
            import networkx as nx
            G = nx.Graph()
            for node, neighbors in adj_list.items():
                G.add_edges_from([(int(node), n) for n in neighbors])
            return G
        return None

    # Pub/Sub for event-driven coordination
    def publish(self, channel, message):
        """Publish event to all subscribers"""
        self.redis.publish(channel, json.dumps(message))

    def subscribe(self, channel, callback):
        """
        Subscribe to events

        Args:
            channel: Event channel (e.g., "assignment_updated")
            callback: Function to call when event received
        """
        def message_handler(message):
            if message['type'] == 'message':
                data = json.loads(message['data'])
                callback(data)

        self.pubsub.subscribe(**{channel: message_handler})
        thread = self.pubsub.run_in_thread(sleep_time=0.01)
        return thread

    # Atomic operations
    def increment_counter(self, key):
        """Atomic counter increment"""
        return self.redis.incr(key)

    def acquire_lock(self, lock_name, timeout=10):
        """Distributed lock for mutual exclusion"""
        lock = self.redis.lock(lock_name, timeout=timeout)
        return lock

# Usage example
blackboard = LibriaBlackboard()

# Write assignment (Librex.QAP)
assignment = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
blackboard.write_assignment(assignment)

# Read assignment (Librex.Flow)
current_assignment = blackboard.read_assignment()
print(f"Current assignment: {current_assignment}")

# Subscribe to events (Librex.Alloc)
def on_assignment_updated(data):
    print(f"Assignment updated: {data}")

thread = blackboard.subscribe("assignment_updated", on_assignment_updated)
```

### 7.3 Performance Optimization

```python
# Connection pooling for better performance
from redis import ConnectionPool

pool = ConnectionPool.from_url("redis://localhost:6379/0", max_connections=50)
redis_client = redis.Redis(connection_pool=pool, decode_responses=True)

# Pipeline for batch operations (reduces network round-trips)
pipe = redis_client.pipeline()
pipe.set("key1", "value1")
pipe.set("key2", "value2")
pipe.set("key3", "value3")
pipe.execute()  # Execute all commands at once

# Lua scripts for atomic multi-operation transactions
lua_script = """
local assignment = redis.call('GET', KEYS[1])
if assignment then
    redis.call('SET', KEYS[2], assignment)
    redis.call('PUBLISH', KEYS[3], assignment)
    return assignment
else
    return nil
end
"""

script = redis_client.register_script(lua_script)
result = script(keys=["libria:assignment:current", "libria:assignment:backup", "assignment_channel"])
```

---

## 8. PostgreSQL Integration

**Purpose**: Persistent storage for execution logs, performance history, benchmarks

**Used By**: All solvers (long-term data storage, analytics)

### 8.1 Installation

```bash
# Install PostgreSQL
# Ubuntu/Debian
sudo apt-get install postgresql postgresql-contrib

# macOS
brew install postgresql

# Start PostgreSQL
sudo service postgresql start  # Linux
brew services start postgresql  # macOS

# Install Python client
pip install psycopg2-binary==2.9.6 sqlalchemy==2.0.18
```

### 8.2 Database Schema

```sql
-- Create database
CREATE DATABASE libria;

-- Execution logs table
CREATE TABLE execution_logs (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    solver VARCHAR(50) NOT NULL,
    task_id VARCHAR(100),
    instance_features JSONB,
    solution JSONB,
    objective_value FLOAT,
    iterations INTEGER,
    time_seconds FLOAT,
    metadata JSONB
);

-- Performance history table (for Librex.Meta)
CREATE TABLE solver_performance (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    solver VARCHAR(50) NOT NULL,
    instance_id VARCHAR(100),
    instance_features JSONB,
    runtime FLOAT,
    quality_score FLOAT,
    success BOOLEAN
);

-- Assignment history (for Librex.QAP online learning)
CREATE TABLE assignment_history (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    agent_id INTEGER,
    task_id INTEGER,
    context JSONB,
    predicted_cost FLOAT,
    actual_cost FLOAT
);

-- Create indexes
CREATE INDEX idx_execution_logs_solver ON execution_logs(solver);
CREATE INDEX idx_execution_logs_timestamp ON execution_logs(timestamp);
CREATE INDEX idx_solver_performance_solver ON solver_performance(solver);
CREATE INDEX idx_assignment_history_agent ON assignment_history(agent_id);
```

### 8.3 SQLAlchemy ORM

```python
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, JSON, TIMESTAMP
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

# Database connection
DATABASE_URL = "postgresql://user:password@localhost:5432/libria"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

# ORM models
class ExecutionLog(Base):
    __tablename__ = "execution_logs"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(TIMESTAMP, default=datetime.utcnow)
    solver = Column(String(50), nullable=False)
    task_id = Column(String(100))
    instance_features = Column(JSON)
    solution = Column(JSON)
    objective_value = Column(Float)
    iterations = Column(Integer)
    time_seconds = Column(Float)
    metadata = Column(JSON)

class SolverPerformance(Base):
    __tablename__ = "solver_performance"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(TIMESTAMP, default=datetime.utcnow)
    solver = Column(String(50), nullable=False)
    instance_id = Column(String(100))
    instance_features = Column(JSON)
    runtime = Column(Float)
    quality_score = Column(Float)
    success = Column(Boolean)

# Create tables
Base.metadata.create_all(bind=engine)

# Usage: Log execution
session = SessionLocal()

log = ExecutionLog(
    solver="Librex.QAP",
    task_id="task_123",
    instance_features={"n_agents": 10, "n_tasks": 10, "complexity": 0.7},
    solution={"assignment": assignment.tolist()},
    objective_value=123.45,
    iterations=100,
    time_seconds=1.23,
    metadata={"convergence": "spectral_init"}
)

session.add(log)
session.commit()
session.close()
```

### 8.4 Query Performance History

```python
from sqlalchemy import func

def query_solver_performance(solver_name, limit=100):
    """
    Query recent performance of a solver

    Returns:
        List of (instance_features, runtime) tuples
    """
    session = SessionLocal()

    results = session.query(
        SolverPerformance.instance_features,
        SolverPerformance.runtime
    ).filter(
        SolverPerformance.solver == solver_name,
        SolverPerformance.success == True
    ).order_by(
        SolverPerformance.timestamp.desc()
    ).limit(limit).all()

    session.close()

    return [(r.instance_features, r.runtime) for r in results]

# Usage for Librex.Meta training
performance_data = query_solver_performance("Librex.QAP", limit=1000)
X = np.array([features for features, _ in performance_data])
y = np.array([runtime for _, runtime in performance_data])

# Train performance model
model = RandomForestRegressor()
model.fit(X, y)
```

---

## 9. Additional Libraries

### 9.1 CuPy (GPU-accelerated NumPy)

```bash
# Install CuPy for CUDA 11.8
pip install cupy-cuda11x==12.1.0
```

```python
import cupy as cp

# GPU-accelerated matrix operations (Librex.QAP)
A_cpu = np.random.rand(1000, 1000)
B_cpu = np.random.rand(1000, 1000)

# Move to GPU
A_gpu = cp.asarray(A_cpu)
B_gpu = cp.asarray(B_cpu)

# Matrix multiplication on GPU (10-50× faster for large matrices)
C_gpu = cp.dot(A_gpu, B_gpu)

# Move back to CPU
C_cpu = cp.asnumpy(C_gpu)
```

### 9.2 Numba (JIT compilation)

```bash
pip install numba==0.57.1
```

```python
from numba import jit

@jit(nopython=True)
def compute_objective_fast(X, c, s, λ):
    """
    JIT-compiled objective function for QAP

    5-10× faster than pure Python
    """
    n, m = X.shape
    obj = 0.0

    # Linear term
    for i in range(n):
        for j in range(m):
            obj += c[i, j] * X[i, j]

    # Quadratic term
    for i in range(n):
        for j in range(m):
            for k in range(n):
                if k != i:
                    obj -= λ * s[i, k] * X[i, j] * X[k, j]

    return obj

# Usage
obj_value = compute_objective_fast(X, c, s, λ)
```

### 9.3 joblib (Parallel Processing)

```bash
pip install joblib==1.3.1
```

```python
from joblib import Parallel, delayed

def evaluate_individual(individual, tasks):
    """Evaluate single coordination pattern (Librex.Evo)"""
    total_score = 0
    for task in tasks:
        score = execute_task(task, individual)
        total_score += score
    return total_score / len(tasks)

# Parallel evaluation of population
population = [random_individual() for _ in range(100)]
tasks = load_benchmark_tasks()

# Evaluate in parallel (uses all CPU cores)
fitness_scores = Parallel(n_jobs=-1)(
    delayed(evaluate_individual)(ind, tasks)
    for ind in population
)
```

---

## 10. Docker Integration

### 10.1 Dockerfile

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Install Libria Suite
RUN pip install -e .

# Expose ports (if running API)
EXPOSE 8000

# Run solver
CMD ["python", "-m", "libria_core.main"]
```

### 10.2 Docker Compose

```yaml
version: '3.8'

services:
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes

  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: libria
      POSTGRES_USER: libria
      POSTGRES_PASSWORD: libria_password
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

  libria-core:
    build: .
    depends_on:
      - redis
      - postgres
    environment:
      REDIS_URL: redis://redis:6379/0
      DATABASE_URL: postgresql://libria:libria_password@postgres:5432/libria
    volumes:
      - ./packages:/app/packages

volumes:
  redis_data:
  postgres_data:
```

### 10.3 Running with Docker

```bash
# Build and start all services
docker-compose up --build

# Run specific solver
docker-compose run libria-core python -m Librex.QAP.solver

# Stop all services
docker-compose down

# View logs
docker-compose logs -f libria-core
```

---

## Summary Checklist

### Installation Checklist

- [ ] Python 3.10+ installed
- [ ] Virtual environment created
- [ ] Core dependencies installed (NumPy, SciPy, pandas)
- [ ] OR-Tools installed
- [ ] NetworkX installed
- [ ] PyTorch installed (with CUDA if using GPU)
- [ ] scikit-learn installed
- [ ] Redis installed and running
- [ ] PostgreSQL installed and running
- [ ] Python clients installed (redis, psycopg2, sqlalchemy)

### Integration Checklist

- [ ] Libria-Core base classes implemented
- [ ] Redis Blackboard working (write/read operations)
- [ ] PostgreSQL schema created (execution_logs, solver_performance)
- [ ] OR-Tools integration tested (linear assignment)
- [ ] NetworkX integration tested (graph creation, algorithms)
- [ ] PyTorch integration tested (neural networks, GPU)
- [ ] scikit-learn integration tested (regression models)
- [ ] Docker setup complete (Dockerfile, docker-compose.yml)

---

*Guide Version: 1.0.0*
*Created: November 14, 2025*
*Next Update: After implementation of first solver (Librex.Meta)*
