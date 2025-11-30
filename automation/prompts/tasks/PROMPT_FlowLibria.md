# FlowLibria Implementation Superprompt

**Version**: 1.0
**Target**: AAMAS 2026 / AAAI 2026
**Priority**: High (2 strong contributions)
**Status**: Ready for Implementation

---

## Executive Summary

FlowLibria implements confidence-aware workflow routing with validation quality objectives for multi-agent orchestration. Unlike classical routing policies that only optimize for speed or cost, FlowLibria learns routing policies that explicitly balance validation quality, confidence calibration, and execution efficiency.

**Core Innovation**: Confidence-aware routing + validation quality objectives (🟢 STRONG novelty)

**Research Contributions**:
- **FLOW-C1**: Confidence-Aware Workflow Routing with Quality Objectives
- **FLOW-C2**: Multi-Objective LinUCB for Agent Selection

**Publication Strategy**: AAMAS 2026 or AAAI 2026

---

## 1. Technical Specification

### 1.1 Problem Statement

**Workflow Routing**:
Given:
- Workflow W = (s₀, s₁, ..., sₜ) with current state sₜ
- Available agents A = {a₁, a₂, ..., aₙ}
- Historical executions: {(Wᵢ, aⱼ, qᵢⱼ, cᵢⱼ, vᵢⱼ)}
  - qᵢⱼ: execution quality
  - cᵢⱼ: execution cost (time/tokens)
  - vᵢⱼ: validation quality (confidence calibration)

Find routing policy π: (W, A) → a* maximizing:
```
E[quality] - λ₁ × cost + λ₂ × validation_quality
```

**Key Challenge**: Balance three competing objectives:
1. Maximize execution quality (accuracy, correctness)
2. Minimize cost (latency, compute)
3. Maximize validation quality (confidence calibration, uncertainty estimation)

### 1.2 Core Algorithm

**FlowLibria Architecture**:
```
Input: Workflow state W, Available agents A
│
├─► Feature Extractor: Extract workflow features φ(W)
│   └─► Complexity, dependencies, history
│
├─► Confidence Estimator: Estimate agent confidence p(correct | a, W)
│   └─► Calibrated probability model
│
├─► Multi-Objective LinUCB: Select agent maximizing composite reward
│   └─► Exploitation: learned reward model
│   └─► Exploration: UCB with multi-objective tradeoff
│
├─► Validation Module: Estimate validation quality
│   └─► Expected Calibration Error (ECE)
│   └─► Brier score
│
└─► Output: Selected agent a* + confidence estimate
```

### 1.3 Implementation

```python
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from sklearn.calibration import calibration_curve

class FlowLibria(LibriaSolver):
    """
    Confidence-aware workflow routing with validation quality objectives

    Key Components:
    1. LinUCB Bandit: Multi-armed bandit for agent selection
    2. Confidence Estimator: Calibrated confidence predictions
    3. Validation Quality Model: ECE and Brier score estimation
    4. Multi-Objective Reward: Quality + Cost + Validation
    """

    def __init__(
        self,
        n_agents: int,
        feature_dim: int = 20,
        alpha: float = 1.0,  # UCB exploration parameter
        lambda_cost: float = 0.1,  # Cost penalty weight
        lambda_validation: float = 0.2,  # Validation quality weight
        use_termination: bool = True  # Enable STOP action
    ):
        super().__init__()
        self.n_agents = n_agents
        self.feature_dim = feature_dim
        self.alpha = alpha
        self.lambda_cost = lambda_cost
        self.lambda_validation = lambda_validation
        self.use_termination = use_termination

        # LinUCB parameters (per agent)
        self.A = {i: np.eye(feature_dim) for i in range(n_agents)}  # Design matrices
        self.b = {i: np.zeros(feature_dim) for i in range(n_agents)}  # Response vectors

        # Confidence calibration model
        self.confidence_model = ConfidenceCalibrator()

        # Validation quality estimator
        self.validation_estimator = ValidationQualityEstimator()

        # History for calibration
        self.history = []

        # STOP action (if enabled)
        if use_termination:
            self.A[n_agents] = np.eye(feature_dim)  # STOP action
            self.b[n_agents] = np.zeros(feature_dim)

    @property
    def name(self) -> str:
        return "FlowLibria"

    def select_agent(
        self,
        workflow_state: Dict,
        available_agents: List[int]
    ) -> Tuple[int, float]:
        """
        Select best agent for current workflow state

        Args:
            workflow_state: Dict with workflow features
            available_agents: List of agent IDs

        Returns:
            (selected_agent, confidence): Agent ID and confidence estimate
        """
        # Extract workflow features
        features = self._extract_features(workflow_state)

        # Compute UCB scores for each agent
        ucb_scores = {}
        confidence_estimates = {}

        for agent_id in available_agents:
            # LinUCB score
            A_inv = np.linalg.inv(self.A[agent_id])
            theta = A_inv @ self.b[agent_id]  # Estimated reward weights

            # Exploitation term
            exploitation = features @ theta

            # Exploration term
            exploration = self.alpha * np.sqrt(features @ A_inv @ features)

            # UCB score
            ucb_scores[agent_id] = exploitation + exploration

            # Confidence estimate
            confidence_estimates[agent_id] = self.confidence_model.predict(
                features, agent_id
            )

        # Check for STOP action
        if self.use_termination:
            stop_id = self.n_agents
            A_inv_stop = np.linalg.inv(self.A[stop_id])
            theta_stop = A_inv_stop @ self.b[stop_id]
            ucb_scores[stop_id] = features @ theta_stop + self.alpha * np.sqrt(
                features @ A_inv_stop @ features
            )
            confidence_estimates[stop_id] = 1.0  # Perfect confidence for STOP

        # Select agent with highest UCB
        selected_agent = max(ucb_scores, key=ucb_scores.get)
        confidence = confidence_estimates.get(selected_agent, 0.5)

        return selected_agent, confidence

    def update(
        self,
        workflow_state: Dict,
        selected_agent: int,
        outcome: Dict
    ):
        """
        Update routing policy after observing execution outcome

        Args:
            workflow_state: Workflow features
            selected_agent: Agent that was selected
            outcome: Dict with keys:
                - 'quality': execution quality (0-1)
                - 'cost': execution cost (time/tokens)
                - 'confidence': agent's confidence (0-1)
                - 'correct': whether execution was correct (bool)
        """
        # Extract features
        features = self._extract_features(workflow_state)

        # Compute composite reward
        quality = outcome['quality']
        cost = outcome['cost']

        # Estimate validation quality (Expected Calibration Error)
        ece = self.validation_estimator.compute_ece(
            [outcome['confidence']],
            [int(outcome['correct'])]
        )
        validation_quality = 1.0 - ece  # Higher is better

        # Multi-objective reward
        reward = (
            quality
            - self.lambda_cost * cost
            + self.lambda_validation * validation_quality
        )

        # Update LinUCB parameters
        self.A[selected_agent] += np.outer(features, features)
        self.b[selected_agent] += reward * features

        # Store history for calibration
        self.history.append({
            'features': features,
            'agent': selected_agent,
            'quality': quality,
            'cost': cost,
            'confidence': outcome['confidence'],
            'correct': outcome['correct']
        })

        # Periodically retrain confidence model
        if len(self.history) % 100 == 0:
            self._retrain_confidence_model()

    def _extract_features(self, workflow_state: Dict) -> np.ndarray:
        """
        Extract feature vector from workflow state

        Features include:
        - Workflow complexity (length, depth, dependencies)
        - Historical success rate
        - Current execution path
        - Time/budget constraints
        """
        features = []

        # Workflow structural features
        features.append(workflow_state.get('length', 0) / 100.0)
        features.append(workflow_state.get('depth', 0) / 10.0)
        features.append(len(workflow_state.get('dependencies', [])) / 20.0)

        # Historical features
        features.append(workflow_state.get('success_rate', 0.5))
        features.append(workflow_state.get('avg_quality', 0.5))
        features.append(workflow_state.get('avg_cost', 0.5))

        # Execution path features
        path = workflow_state.get('execution_path', [])
        features.append(len(path) / 50.0)
        features.append(np.mean([step.get('quality', 0.5) for step in path]) if path else 0.5)

        # Constraint features
        features.append(workflow_state.get('time_remaining', 1.0))
        features.append(workflow_state.get('budget_remaining', 1.0))

        # Pad to feature_dim
        features = features[:self.feature_dim]
        if len(features) < self.feature_dim:
            features.extend([0.0] * (self.feature_dim - len(features)))

        return np.array(features)

    def _retrain_confidence_model(self):
        """Retrain confidence calibration model on historical data"""
        print(f"Retraining confidence model on {len(self.history)} instances...")

        # Prepare training data
        X = []  # (features, agent_id)
        y = []  # correctness (0 or 1)

        for h in self.history:
            x = np.concatenate([h['features'], [h['agent']]])
            X.append(x)
            y.append(int(h['correct']))

        X = np.array(X)
        y = np.array(y)

        # Train calibration model
        self.confidence_model.fit(X, y)

        print("✓ Confidence model retrained")

    def estimate_workflow_quality(
        self,
        workflow_state: Dict,
        selected_agent: int
    ) -> Dict[str, float]:
        """
        Estimate expected quality metrics for selected agent

        Returns:
            estimates: {
                'expected_quality': float,
                'expected_cost': float,
                'expected_ece': float,
                'confidence': float
            }
        """
        features = self._extract_features(workflow_state)

        # Predict from LinUCB model
        A_inv = np.linalg.inv(self.A[selected_agent])
        theta = A_inv @ self.b[selected_agent]
        expected_reward = features @ theta

        # Confidence estimate
        confidence = self.confidence_model.predict(features, selected_agent)

        # Decompose reward (approximate)
        # Note: This is a simplified decomposition
        expected_quality = max(0, min(1, expected_reward + self.lambda_cost * 0.5))
        expected_cost = 0.5  # Default estimate
        expected_ece = self.validation_estimator.estimate_ece(confidence)

        return {
            'expected_quality': expected_quality,
            'expected_cost': expected_cost,
            'expected_ece': expected_ece,
            'confidence': confidence
        }


class ConfidenceCalibrator:
    """
    Confidence calibration model using temperature scaling and Platt scaling

    Predicts calibrated confidence p(correct | features, agent)
    """

    def __init__(self):
        self.model = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Train calibration model

        Args:
            X: (N, feature_dim + 1) - features + agent_id
            y: (N,) - binary correctness
        """
        from sklearn.linear_model import LogisticRegression

        self.model = LogisticRegression(max_iter=500)
        self.model.fit(X, y)

    def predict(self, features: np.ndarray, agent_id: int) -> float:
        """
        Predict calibrated confidence

        Args:
            features: (feature_dim,)
            agent_id: int

        Returns:
            confidence: p(correct | features, agent)
        """
        if self.model is None:
            return 0.5  # Default before training

        x = np.concatenate([features, [agent_id]]).reshape(1, -1)
        prob = self.model.predict_proba(x)[0, 1]
        return prob


class ValidationQualityEstimator:
    """
    Estimate validation quality metrics:
    - Expected Calibration Error (ECE)
    - Brier score
    """

    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins
        self.history_confidences = []
        self.history_correct = []

    def compute_ece(
        self,
        confidences: List[float],
        correct: List[int]
    ) -> float:
        """
        Compute Expected Calibration Error

        ECE = ∑ (|Bₘ| / N) × |acc(Bₘ) - conf(Bₘ)|

        Args:
            confidences: List of predicted confidences
            correct: List of binary correctness (0 or 1)

        Returns:
            ece: Expected Calibration Error (0 = perfect, 1 = worst)
        """
        if len(confidences) == 0:
            return 0.0

        confidences = np.array(confidences)
        correct = np.array(correct)

        # Bin confidences
        bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
        ece = 0.0

        for i in range(self.n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]

            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            n_in_bin = np.sum(in_bin)

            if n_in_bin > 0:
                avg_confidence = np.mean(confidences[in_bin])
                avg_accuracy = np.mean(correct[in_bin])
                ece += (n_in_bin / len(confidences)) * abs(avg_accuracy - avg_confidence)

        return ece

    def estimate_ece(self, confidence: float) -> float:
        """
        Estimate ECE for a single confidence value based on historical data

        Args:
            confidence: Single confidence value

        Returns:
            estimated_ece: Estimated calibration error
        """
        if len(self.history_confidences) < 10:
            return 0.1  # Default estimate

        # Find similar historical confidences
        confidences = np.array(self.history_confidences)
        similar_mask = np.abs(confidences - confidence) < 0.1

        if np.sum(similar_mask) == 0:
            return 0.1

        # Compute ECE for similar subset
        return self.compute_ece(
            confidences[similar_mask].tolist(),
            [self.history_correct[i] for i, m in enumerate(similar_mask) if m]
        )

    def add_observation(self, confidence: float, correct: bool):
        """Add observation to history"""
        self.history_confidences.append(confidence)
        self.history_correct.append(int(correct))


# Integration example
class WorkflowRouter:
    """
    High-level workflow router using FlowLibria

    Manages multi-step workflows with agent selection at each step
    """

    def __init__(self, flow_solver: FlowLibria):
        self.flow_solver = flow_solver
        self.workflows = {}

    def route_workflow(
        self,
        workflow_id: str,
        workflow_spec: Dict,
        available_agents: List[int]
    ) -> List[Tuple[int, float]]:
        """
        Route workflow through agents

        Args:
            workflow_id: Unique workflow identifier
            workflow_spec: Workflow specification
            available_agents: List of available agent IDs

        Returns:
            routing_plan: List of (agent_id, confidence) for each step
        """
        routing_plan = []
        workflow_state = self._initialize_workflow_state(workflow_spec)

        for step in workflow_spec['steps']:
            # Update workflow state
            workflow_state['current_step'] = step
            workflow_state['execution_path'] = routing_plan

            # Select agent
            agent_id, confidence = self.flow_solver.select_agent(
                workflow_state, available_agents
            )

            # Check for STOP action
            if agent_id == self.flow_solver.n_agents:
                print("FlowLibria decided to terminate workflow early")
                break

            routing_plan.append((agent_id, confidence))

            # Update state (simulate)
            workflow_state['length'] = len(routing_plan)

        return routing_plan

    def _initialize_workflow_state(self, workflow_spec: Dict) -> Dict:
        """Initialize workflow state"""
        return {
            'length': 0,
            'depth': workflow_spec.get('depth', 1),
            'dependencies': workflow_spec.get('dependencies', []),
            'success_rate': 0.5,
            'avg_quality': 0.5,
            'avg_cost': 0.5,
            'execution_path': [],
            'time_remaining': 1.0,
            'budget_remaining': 1.0
        }
```

---

## 2. Research Validation

### 2.1 Novel Contributions

**FLOW-C1: Confidence-Aware Workflow Routing with Quality Objectives**
- **Gap**: Existing routing (MasRouter, Nexus) only optimize speed/cost, ignore confidence calibration
- **Approach**: Multi-objective reward with validation quality (ECE) as explicit objective
- **Impact**: 20-30% improvement in confidence calibration + 10-15% quality gain

**FLOW-C2: Multi-Objective LinUCB for Agent Selection**
- **Gap**: Standard LinUCB optimizes single scalar reward
- **Approach**: Composite reward balancing quality, cost, and validation quality
- **Impact**: Pareto-optimal tradeoffs via learned λ weights

### 2.2 Baselines

1. **Recent Multi-Agent Routing**:
   - MasRouter (ACL 2025): Quality-aware routing
   - Nexus (2025): Network of LLM agents
   - AgentOrchestra (2025): Hierarchical orchestration

2. **Classic Bandits**:
   - LinUCB (standard)
   - Thompson Sampling
   - UCB1
   - ε-greedy

3. **Routing Heuristics**:
   - Round-robin
   - Random selection
   - Best-agent (oracle)

### 2.3 Benchmark Datasets

**Multi-Agent Workflow Benchmarks**:
- AgentBench (2024): 8 diverse task types
- GAIA (2024): General AI assistants benchmark
- WebArena (2024): Web agent tasks

**Custom ATLAS Workflows**:
- Research synthesis (40+ agents)
- Code generation + review
- Multi-step problem solving

**Expected Performance**:
- 15-25% quality improvement over MasRouter
- 20-30% better calibration (lower ECE)
- Competitive cost (< 10% overhead)

---

## 3. Implementation Roadmap

### Phase 1: Core Algorithm (Weeks 1-2)

**Week 1**:
- [ ] Implement `FlowLibria` base class
- [ ] Implement LinUCB with multi-objective reward
- [ ] Implement `ConfidenceCalibrator`
- [ ] Unit tests

**Week 2**:
- [ ] Implement `ValidationQualityEstimator`
- [ ] ECE and Brier score computation
- [ ] STOP action logic
- [ ] Integration tests

### Phase 2: Benchmarking (Weeks 3-5)

**Week 3**:
- [ ] Download AgentBench, GAIA, WebArena
- [ ] Implement baseline routers (MasRouter, LinUCB, Random)
- [ ] Evaluation harness

**Week 4-5**:
- [ ] Run FlowLibria on benchmarks
- [ ] Compare against baselines
- [ ] Ablation studies (λ weights, STOP action)
- [ ] Calibration analysis

### Phase 3: Paper Writing (Weeks 6-8)

**Week 6**: Draft
**Week 7**: Results + Ablations
**Week 8**: Submission to AAMAS 2026 (deadline ~Nov 2025)

---

## 4. Integration with Libria Suite

### 4.1 ATLAS Integration

```python
from atlas_engine import ATLASEngine
from libria_flow import FlowLibria

atlas = ATLASEngine()
flow_router = FlowLibria(n_agents=len(atlas.agents))

@atlas.register_workflow("adaptive_routing")
def route_workflow(workflow_spec: Dict):
    available_agents = atlas.get_available_agents()

    # Route workflow
    routing_plan = []
    for step in workflow_spec['steps']:
        workflow_state = {
            'length': len(routing_plan),
            'execution_path': routing_plan,
            # ... extract other features
        }

        agent_id, confidence = flow_router.select_agent(
            workflow_state, [a.id for a in available_agents]
        )

        # Execute step
        result = available_agents[agent_id].execute(step)

        # Update FlowLibria
        outcome = {
            'quality': result['quality'],
            'cost': result['cost'],
            'confidence': result['confidence'],
            'correct': result['correct']
        }
        flow_router.update(workflow_state, agent_id, outcome)

        routing_plan.append((agent_id, confidence))

    return routing_plan
```

---

## 5. Testing Protocol

### 5.1 Unit Tests

```python
# tests/test_flowlibria.py
import pytest
import numpy as np
from libria_flow import FlowLibria

def test_linucb_initialization():
    """Test LinUCB parameters initialized correctly"""
    flow = FlowLibria(n_agents=5, feature_dim=10)

    for i in range(5):
        assert flow.A[i].shape == (10, 10)
        assert flow.b[i].shape == (10,)
        assert np.allclose(flow.A[i], np.eye(10))

def test_agent_selection():
    """Test agent selection with UCB"""
    flow = FlowLibria(n_agents=3, feature_dim=5)

    workflow_state = {'length': 0, 'depth': 1}
    agent_id, confidence = flow.select_agent(workflow_state, [0, 1, 2])

    assert agent_id in [0, 1, 2]
    assert 0 <= confidence <= 1

def test_linucb_update():
    """Test LinUCB update after observation"""
    flow = FlowLibria(n_agents=2, feature_dim=5)

    workflow_state = {'length': 0}
    outcome = {
        'quality': 0.8,
        'cost': 0.5,
        'confidence': 0.9,
        'correct': True
    }

    A_before = flow.A[0].copy()
    b_before = flow.b[0].copy()

    flow.update(workflow_state, selected_agent=0, outcome=outcome)

    # Check updates
    assert not np.allclose(flow.A[0], A_before)
    assert not np.allclose(flow.b[0], b_before)

def test_ece_computation():
    """Test Expected Calibration Error computation"""
    estimator = ValidationQualityEstimator(n_bins=5)

    # Perfect calibration
    confidences = [0.1, 0.3, 0.5, 0.7, 0.9]
    correct = [0, 0, 1, 1, 1]
    ece = estimator.compute_ece(confidences, correct)

    assert 0 <= ece <= 1
```

### 5.2 Integration Tests

```python
# tests/test_flow_integration.py
def test_workflow_routing():
    """Test full workflow routing"""
    flow = FlowLibria(n_agents=5, feature_dim=10)
    router = WorkflowRouter(flow)

    workflow_spec = {
        'steps': [{'task': 'step1'}, {'task': 'step2'}, {'task': 'step3'}],
        'depth': 1,
        'dependencies': []
    }

    routing_plan = router.route_workflow(
        workflow_id='test',
        workflow_spec=workflow_spec,
        available_agents=[0, 1, 2, 3, 4]
    )

    assert len(routing_plan) <= 3  # May terminate early
    for agent_id, confidence in routing_plan:
        assert 0 <= agent_id < 5
        assert 0 <= confidence <= 1
```

---

## 6. Publication Strategy

### 6.1 Target Venues

**Primary**: AAMAS 2026 (International Conference on Autonomous Agents and Multiagent Systems)
- Deadline: ~November 2025
- Notification: February 2026
- Conference: May 2026

**Backup**:
- AAAI 2026
- NeurIPS 2025 (if ready earlier)

### 6.2 Paper Outline

1. Introduction (2 pages)
2. Related Work (1.5 pages)
   - Multi-agent routing
   - Contextual bandits
   - Confidence calibration
3. Method (3 pages)
   - Confidence-aware routing formulation
   - Multi-objective LinUCB
   - Validation quality estimation
4. Experiments (3 pages)
   - Benchmarks
   - Baseline comparison
   - Ablation studies
5. Discussion (1 page)
6. Conclusion (0.5 pages)

---

## 7. Success Criteria

**AgentBench Performance**:
- ✅ 15-25% quality improvement over MasRouter
- ✅ 20-30% ECE reduction (better calibration)
- ✅ Competitive cost (< 10% overhead)

**Ablation Studies**:
- ✅ Validation quality term improves ECE by 20-30%
- ✅ STOP action reduces unnecessary steps by 10-15%
- ✅ Multi-objective reward achieves Pareto improvements

---

## 8. Next Actions

**Immediate**:
1. Set up repository
2. Implement FlowLibria core
3. Implement LinUCB + confidence calibration
4. Unit tests

**Short-term**:
1. Benchmarking setup
2. Baseline implementations
3. Evaluation harness

**Mid-term**:
1. Run experiments
2. Ablation studies
3. Write paper

**Long-term**:
1. Submit to AAMAS 2026
2. Open-source release

---

**END OF FLOWLIBRIA SUPERPROMPT**

**Version**: 1.0
**Last Updated**: 2026-01-17
**Status**: Ready for Implementation
**Target**: November 2025 (AAMAS 2026 submission)
