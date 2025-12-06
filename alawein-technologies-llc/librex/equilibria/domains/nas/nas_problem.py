"""NAS problem definition and evaluation framework.

This module defines the NAS optimization problem including:
- Multi-objective formulation
- Architecture evaluation strategies
- Proxy methods for fast evaluation
- Dataset and task specifications
"""

from typing import Dict, List, Optional, Callable, Any, Union, Tuple
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
import time
import warnings
from abc import ABC, abstractmethod

from .architecture import NASCell, MacroArchitecture, Operation


class SearchSpace(Enum):
    """Available search spaces."""
    CELL = 'cell'
    MACRO = 'macro'
    HIERARCHICAL = 'hierarchical'


class EvaluationStrategy(Enum):
    """Architecture evaluation strategies."""
    FULL_TRAINING = 'full'
    EARLY_STOPPING = 'early_stopping'
    WEIGHT_SHARING = 'weight_sharing'
    PROXY = 'proxy'
    ZERO_COST = 'zero_cost'
    PREDICTOR = 'predictor'


@dataclass
class NASObjective:
    """Objective for NAS optimization."""
    name: str
    weight: float = 1.0
    minimize: bool = False
    target_value: Optional[float] = None

    def compute_score(self, value: float) -> float:
        """Compute objective score."""
        if self.minimize:
            score = -value
        else:
            score = value

        if self.target_value is not None:
            # Distance to target
            score = -abs(value - self.target_value)

        return self.weight * score


@dataclass
class NASConstraint:
    """Constraint for NAS optimization."""
    metric: str
    operator: str  # 'le', 'ge', 'eq'
    value: float

    def is_satisfied(self, metric_value: float) -> bool:
        """Check if constraint is satisfied."""
        if self.operator == 'le':
            return metric_value <= self.value
        elif self.operator == 'ge':
            return metric_value >= self.value
        elif self.operator == 'eq':
            return abs(metric_value - self.value) < 1e-6
        else:
            raise ValueError(f"Unknown operator: {self.operator}")


class ModelEvaluator(ABC):
    """Abstract base class for architecture evaluation."""

    @abstractmethod
    def evaluate(self, architecture: Union[NASCell, MacroArchitecture]) -> Dict[str, float]:
        """Evaluate an architecture and return metrics."""
        pass


class SimulatedEvaluator(ModelEvaluator):
    """Simulated evaluator for testing (no actual training)."""

    def __init__(self, noise_level: float = 0.1):
        self.noise_level = noise_level
        self.eval_cache = {}

    def evaluate(self, architecture: Union[NASCell, MacroArchitecture]) -> Dict[str, float]:
        """Simulate evaluation with realistic metrics."""
        # Use hash for caching
        if isinstance(architecture, NASCell):
            arch_hash = architecture.get_hash()
            params = architecture.get_params()
            flops = architecture.get_flops()
            depth = architecture.n_nodes
        else:
            arch_hash = str(architecture.to_dict())
            params = architecture.get_params()
            flops = params * 20  # Rough estimate
            depth = architecture.get_depth()

        if arch_hash in self.eval_cache:
            return self.eval_cache[arch_hash]

        # Simulate accuracy based on architecture properties
        base_accuracy = 0.7

        # More parameters generally help (with diminishing returns)
        param_bonus = min(0.2, np.log(params / 1e5 + 1) * 0.1)

        # Depth bonus (but too deep hurts)
        depth_bonus = 0.05 * min(depth / 10, 1.5)
        if depth > 50:
            depth_bonus -= 0.1 * (depth - 50) / 50

        # Add noise
        noise = np.random.normal(0, self.noise_level)

        accuracy = np.clip(base_accuracy + param_bonus + depth_bonus + noise, 0, 1)

        # Simulate latency (milliseconds)
        latency = 10 + flops / 1e8 + np.random.normal(0, 2)

        # Energy consumption (millijoules)
        energy = params / 1e5 + flops / 1e9 + np.random.normal(0, 0.5)

        metrics = {
            'accuracy': accuracy,
            'params': params,
            'flops': flops,
            'latency_ms': latency,
            'energy_mj': energy,
            'depth': depth
        }

        self.eval_cache[arch_hash] = metrics
        return metrics


class ProxyEvaluator(ModelEvaluator):
    """Fast proxy evaluation using predictive methods."""

    def __init__(self, proxy_type: str = 'grad_norm', fidelity: float = 0.1):
        """
        Initialize proxy evaluator.

        Args:
            proxy_type: Type of proxy ('grad_norm', 'jacob_cov', 'ntk', 'zen')
            fidelity: Training fidelity (0-1, fraction of full training)
        """
        self.proxy_type = proxy_type
        self.fidelity = fidelity
        self.eval_cache = {}

    def evaluate(self, architecture: Union[NASCell, MacroArchitecture]) -> Dict[str, float]:
        """Evaluate using proxy method."""
        # For demonstration, we'll use simulated proxy scores
        # In practice, this would call actual zero-cost proxy implementations

        if isinstance(architecture, NASCell):
            arch_hash = architecture.get_hash()
            params = architecture.get_params()
        else:
            arch_hash = str(architecture.to_dict())
            params = architecture.get_params()

        if arch_hash in self.eval_cache:
            return self.eval_cache[arch_hash]

        # Simulate proxy score based on architecture
        if self.proxy_type == 'grad_norm':
            # Gradient norm correlates with trainability
            proxy_score = np.random.uniform(0.5, 1.5) * (1 + np.log(params / 1e5 + 1) * 0.1)
        elif self.proxy_type == 'jacob_cov':
            # Jacobian covariance measures expressivity
            proxy_score = np.random.uniform(1, 10) * (1 + params / 1e6)
        elif self.proxy_type == 'ntk':
            # Neural Tangent Kernel condition number
            proxy_score = np.random.uniform(0.1, 2) * np.sqrt(params / 1e5 + 1)
        else:
            proxy_score = np.random.uniform(0.5, 1.5)

        # Convert proxy score to estimated accuracy
        # This mapping would be learned from actual data
        estimated_accuracy = 0.6 + 0.3 * np.tanh(proxy_score / 2)

        metrics = {
            'accuracy': estimated_accuracy,
            'params': params,
            'proxy_score': proxy_score,
            'evaluation_time': self.fidelity * 100  # Simulated time in seconds
        }

        self.eval_cache[arch_hash] = metrics
        return metrics


class NASProblem:
    """
    Neural Architecture Search Problem definition.

    This class encapsulates the complete NAS problem including:
    - Search space definition
    - Dataset and task specification
    - Objectives and constraints
    - Evaluation strategy
    """

    def __init__(self,
                 dataset: str,
                 search_space: Union[str, SearchSpace] = 'cell',
                 objectives: Optional[List[Union[str, NASObjective]]] = None,
                 constraints: Optional[Dict[str, float]] = None,
                 evaluation_strategy: Union[str, EvaluationStrategy] = 'proxy',
                 evaluator: Optional[ModelEvaluator] = None,
                 max_evaluations: int = 1000,
                 time_budget: Optional[float] = None):
        """
        Initialize NAS problem.

        Args:
            dataset: Dataset name (e.g., 'cifar10', 'imagenet', 'custom')
            search_space: Type of search space ('cell', 'macro', 'hierarchical')
            objectives: List of objectives (strings or NASObjective objects)
            constraints: Dict of constraints (e.g., {'max_params': 1e6})
            evaluation_strategy: How to evaluate architectures
            evaluator: Custom evaluator instance
            max_evaluations: Maximum number of architecture evaluations
            time_budget: Time budget in seconds
        """
        self.dataset = dataset
        self.search_space = SearchSpace(search_space) if isinstance(search_space, str) else search_space

        # Set up objectives
        if objectives is None:
            objectives = ['accuracy']
        self.objectives = []
        for obj in objectives:
            if isinstance(obj, str):
                # Convert string to objective
                if obj == 'accuracy':
                    self.objectives.append(NASObjective(obj, weight=1.0, minimize=False))
                elif obj in ['params', 'flops', 'latency_ms', 'energy_mj']:
                    self.objectives.append(NASObjective(obj, weight=0.3, minimize=True))
                else:
                    self.objectives.append(NASObjective(obj, weight=1.0, minimize=False))
            else:
                self.objectives.append(obj)

        # Set up constraints
        self.constraints = []
        if constraints:
            for metric, value in constraints.items():
                if metric.startswith('max_'):
                    actual_metric = metric[4:]  # Remove 'max_' prefix
                    self.constraints.append(NASConstraint(actual_metric, 'le', value))
                elif metric.startswith('min_'):
                    actual_metric = metric[4:]  # Remove 'min_' prefix
                    self.constraints.append(NASConstraint(actual_metric, 'ge', value))
                else:
                    self.constraints.append(NASConstraint(metric, 'eq', value))

        self.evaluation_strategy = EvaluationStrategy(evaluation_strategy) if isinstance(
            evaluation_strategy, str) else evaluation_strategy

        # Set up evaluator
        if evaluator is None:
            if self.evaluation_strategy == EvaluationStrategy.PROXY:
                self.evaluator = ProxyEvaluator()
            else:
                # Default to simulated for demonstration
                self.evaluator = SimulatedEvaluator()
        else:
            self.evaluator = evaluator

        self.max_evaluations = max_evaluations
        self.time_budget = time_budget
        self.evaluation_count = 0
        self.start_time = None
        self.history: List[Dict[str, Any]] = []

        # Search space specific parameters
        if self.search_space == SearchSpace.CELL:
            self.cell_config = {
                'n_nodes': 4,
                'n_inputs': 2
            }
        elif self.search_space == SearchSpace.MACRO:
            self.macro_config = {
                'max_layers': 20,
                'input_channels': 3,
                'num_classes': self._get_num_classes()
            }

    def _get_num_classes(self) -> int:
        """Get number of classes for the dataset."""
        dataset_classes = {
            'cifar10': 10,
            'cifar100': 100,
            'imagenet': 1000,
            'mnist': 10,
            'fashion_mnist': 10,
            'svhn': 10,
            'custom': 10  # Default
        }
        return dataset_classes.get(self.dataset.lower(), 10)

    def create_architecture(self) -> Union[NASCell, MacroArchitecture]:
        """Create a new architecture instance for this problem."""
        if self.search_space == SearchSpace.CELL:
            return NASCell(**self.cell_config)
        elif self.search_space == SearchSpace.MACRO:
            return MacroArchitecture(**self.macro_config)
        else:
            raise ValueError(f"Unsupported search space: {self.search_space}")

    def evaluate_architecture(self,
                            architecture: Union[NASCell, MacroArchitecture],
                            return_all_metrics: bool = False) -> Union[float, Dict[str, float]]:
        """
        Evaluate an architecture.

        Args:
            architecture: Architecture to evaluate
            return_all_metrics: If True, return all metrics; else return objective value

        Returns:
            Single objective value or dict of all metrics
        """
        if self.start_time is None:
            self.start_time = time.time()

        # Check budget constraints
        if self.evaluation_count >= self.max_evaluations:
            warnings.warn(f"Evaluation budget exceeded ({self.max_evaluations})")
            if return_all_metrics:
                return {'error': 'budget_exceeded'}
            return float('-inf')

        if self.time_budget and (time.time() - self.start_time) > self.time_budget:
            warnings.warn(f"Time budget exceeded ({self.time_budget}s)")
            if return_all_metrics:
                return {'error': 'time_exceeded'}
            return float('-inf')

        # Evaluate architecture
        metrics = self.evaluator.evaluate(architecture)
        self.evaluation_count += 1

        # Check constraints
        constraint_penalty = 0
        for constraint in self.constraints:
            if constraint.metric in metrics:
                if not constraint.is_satisfied(metrics[constraint.metric]):
                    constraint_penalty += 1000  # Large penalty for violation

        # Compute objective value
        objective_value = 0
        for obj in self.objectives:
            if obj.name in metrics:
                objective_value += obj.compute_score(metrics[obj.name])

        objective_value -= constraint_penalty

        # Store in history
        self.history.append({
            'architecture': architecture,
            'metrics': metrics,
            'objective': objective_value,
            'evaluation': self.evaluation_count,
            'time': time.time() - self.start_time if self.start_time else 0
        })

        if return_all_metrics:
            metrics['objective'] = objective_value
            return metrics
        return objective_value

    def get_dimension(self) -> int:
        """Get the dimension of the optimization problem."""
        if self.search_space == SearchSpace.CELL:
            # Each possible edge has an operation choice and channels
            # Count total possible edges
            total_edges = 0
            for i in range(self.cell_config['n_inputs'],
                          self.cell_config['n_inputs'] + self.cell_config['n_nodes']):
                total_edges += i  # Each node can connect to all previous nodes
            return total_edges * 2  # Operation + channels for each edge

        elif self.search_space == SearchSpace.MACRO:
            # Depth + layer types + channels + kernel sizes + skip connections
            max_layers = self.macro_config['max_layers']
            return 1 + max_layers * 3 + max_layers * max_layers

        else:
            raise ValueError(f"Unsupported search space: {self.search_space}")

    def get_bounds(self) -> List[Tuple[float, float]]:
        """Get bounds for each dimension of the problem."""
        bounds = []

        if self.search_space == SearchSpace.CELL:
            # Count total possible edges
            total_edges = 0
            for i in range(self.cell_config['n_inputs'],
                          self.cell_config['n_inputs'] + self.cell_config['n_nodes']):
                total_edges += i
            for _ in range(total_edges):
                bounds.append((0, 16))  # Operation index (17 operations)
                bounds.append((16, 512))  # Channel size

        elif self.search_space == SearchSpace.MACRO:
            bounds.append((1, self.macro_config['max_layers']))  # Depth
            for _ in range(self.macro_config['max_layers']):
                bounds.append((0, 5))  # Layer type (6 types)
                bounds.append((16, 512))  # Channels
                bounds.append((0, 7))  # Kernel size (0=none, 1, 3, 5, 7)
            # Skip connections (binary)
            for _ in range(self.macro_config['max_layers'] ** 2):
                bounds.append((0, 1))

        return bounds

    def is_multi_objective(self) -> bool:
        """Check if this is a multi-objective problem."""
        return len(self.objectives) > 1

    def get_pareto_front(self) -> List[Dict[str, Any]]:
        """Get Pareto front from evaluation history."""
        if not self.is_multi_objective():
            # Single objective - return best
            if self.history:
                best = max(self.history, key=lambda x: x['objective'])
                return [best]
            return []

        # Multi-objective - compute Pareto front
        pareto_front = []
        for candidate in self.history:
            dominated = False
            for other in self.history:
                if candidate == other:
                    continue

                # Check if other dominates candidate
                all_worse_or_equal = True
                at_least_one_worse = False
                for obj in self.objectives:
                    if obj.name in candidate['metrics'] and obj.name in other['metrics']:
                        cand_val = candidate['metrics'][obj.name]
                        other_val = other['metrics'][obj.name]

                        if obj.minimize:
                            if cand_val < other_val:
                                all_worse_or_equal = False
                            elif cand_val > other_val:
                                at_least_one_worse = True
                        else:
                            if cand_val > other_val:
                                all_worse_or_equal = False
                            elif cand_val < other_val:
                                at_least_one_worse = True

                if all_worse_or_equal and at_least_one_worse:
                    dominated = True
                    break

            if not dominated:
                pareto_front.append(candidate)

        return pareto_front

    def reset(self):
        """Reset the problem state."""
        self.evaluation_count = 0
        self.start_time = None
        self.history = []
        if hasattr(self.evaluator, 'eval_cache'):
            self.evaluator.eval_cache = {}