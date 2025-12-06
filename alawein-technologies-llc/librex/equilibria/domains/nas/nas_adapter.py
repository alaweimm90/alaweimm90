"""NAS adapter for Librex optimization framework.

This module provides the adapter that allows NAS problems to be solved
using Librex's optimization algorithms.
"""

from typing import Dict, Any, Optional, List, Callable, Union, Tuple
import numpy as np
import warnings

from .nas_problem import NASProblem, SearchSpace
from .architecture import NASCell, MacroArchitecture


class StandardizedProblem:
    """Standardized optimization problem format for Librex."""

    def __init__(self,
                 dimension: int,
                 objective_function: Callable[[np.ndarray], float],
                 bounds: Optional[List[Tuple[float, float]]] = None,
                 encoding_type: str = 'continuous',
                 constraint_function: Optional[Callable[[np.ndarray], float]] = None,
                 is_multi_objective: bool = False):
        """
        Initialize standardized problem.

        Args:
            dimension: Problem dimension
            objective_function: Function to optimize
            bounds: Variable bounds
            encoding_type: Type of encoding ('continuous', 'categorical', 'mixed')
            constraint_function: Constraint function (returns penalty)
            is_multi_objective: Whether problem has multiple objectives
        """
        self.dimension = dimension
        self.objective_function = objective_function
        self.bounds = bounds
        self.encoding_type = encoding_type
        self.constraint_function = constraint_function
        self.is_multi_objective = is_multi_objective


class DomainAdapter:
    """Base class for domain adapters."""

    def encode_problem(self, problem: Any) -> StandardizedProblem:
        """Encode domain-specific problem to standardized format."""
        raise NotImplementedError

    def decode_solution(self, solution: np.ndarray, problem: Any) -> Any:
        """Decode optimization solution back to domain format."""
        raise NotImplementedError


class NASAdapter(DomainAdapter):
    """
    Adapter for Neural Architecture Search problems.

    This adapter converts NAS problems into a format that can be optimized
    by Librex's algorithms, handling:
    - Architecture encoding/decoding
    - Objective function wrapping
    - Constraint handling
    - Multi-objective scalarization
    """

    def __init__(self,
                 scalarization_method: str = 'weighted_sum',
                 repair_invalid: bool = True,
                 cache_evaluations: bool = True):
        """
        Initialize NAS adapter.

        Args:
            scalarization_method: Method for multi-objective scalarization
                ('weighted_sum', 'chebyshev', 'pbi')
            repair_invalid: Whether to repair invalid architectures
            cache_evaluations: Whether to cache architecture evaluations
        """
        self.scalarization_method = scalarization_method
        self.repair_invalid = repair_invalid
        self.cache_evaluations = cache_evaluations
        self.eval_cache: Dict[str, float] = {}

    def encode_problem(self, problem: NASProblem) -> StandardizedProblem:
        """
        Encode NAS problem as standardized optimization problem.

        Args:
            problem: NAS problem instance

        Returns:
            StandardizedProblem instance
        """
        dimension = problem.get_dimension()
        bounds = problem.get_bounds()

        # Determine encoding type based on search space
        if problem.search_space == SearchSpace.CELL:
            encoding_type = 'mixed'  # Mix of categorical (operations) and continuous (channels)
        elif problem.search_space == SearchSpace.MACRO:
            encoding_type = 'mixed'
        else:
            encoding_type = 'continuous'

        # Create objective function wrapper
        def objective_function(encoding: np.ndarray) -> float:
            """Wrapped objective function."""
            # Check cache
            cache_key = str(encoding.tolist())
            if self.cache_evaluations and cache_key in self.eval_cache:
                return self.eval_cache[cache_key]

            # Decode architecture
            architecture = self._decode_architecture(encoding, problem)

            # Repair if needed
            if self.repair_invalid:
                architecture = self._repair_architecture(architecture, problem)

            # Evaluate
            if problem.is_multi_objective():
                metrics = problem.evaluate_architecture(architecture, return_all_metrics=True)
                obj_value = self._scalarize_objectives(metrics, problem)
            else:
                obj_value = problem.evaluate_architecture(architecture, return_all_metrics=False)

            # Cache result
            if self.cache_evaluations:
                self.eval_cache[cache_key] = obj_value

            return obj_value

        # Create constraint function if constraints exist
        constraint_function = None
        if problem.constraints:
            def constraint_function(encoding: np.ndarray) -> float:
                """Compute constraint penalty."""
                architecture = self._decode_architecture(encoding, problem)
                metrics = problem.evaluator.evaluate(architecture)

                penalty = 0
                for constraint in problem.constraints:
                    if constraint.metric in metrics:
                        if not constraint.is_satisfied(metrics[constraint.metric]):
                            # Compute violation magnitude
                            value = metrics[constraint.metric]
                            if constraint.operator == 'le':
                                violation = max(0, value - constraint.value)
                            elif constraint.operator == 'ge':
                                violation = max(0, constraint.value - value)
                            else:
                                violation = abs(value - constraint.value)

                            penalty += violation * 1000  # Scale penalty

                return penalty

        return StandardizedProblem(
            dimension=dimension,
            objective_function=objective_function,
            bounds=bounds,
            encoding_type=encoding_type,
            constraint_function=constraint_function,
            is_multi_objective=problem.is_multi_objective()
        )

    def decode_solution(self,
                       solution: np.ndarray,
                       problem: NASProblem) -> Union[NASCell, MacroArchitecture]:
        """
        Decode optimization solution to neural architecture.

        Args:
            solution: Solution vector from optimizer
            problem: Original NAS problem

        Returns:
            Decoded architecture (NASCell or MacroArchitecture)
        """
        architecture = self._decode_architecture(solution, problem)

        if self.repair_invalid:
            architecture = self._repair_architecture(architecture, problem)

        return architecture

    def _decode_architecture(self,
                            encoding: np.ndarray,
                            problem: NASProblem) -> Union[NASCell, MacroArchitecture]:
        """
        Internal method to decode architecture from encoding.

        Args:
            encoding: Encoded architecture vector
            problem: NAS problem instance

        Returns:
            Decoded architecture
        """
        if problem.search_space == SearchSpace.CELL:
            architecture = NASCell(**problem.cell_config)
            architecture.from_encoding(encoding)

        elif problem.search_space == SearchSpace.MACRO:
            architecture = MacroArchitecture(**problem.macro_config)
            architecture.from_flat_encoding(encoding)

        else:
            raise ValueError(f"Unsupported search space: {problem.search_space}")

        return architecture

    def _repair_architecture(self,
                           architecture: Union[NASCell, MacroArchitecture],
                           problem: NASProblem) -> Union[NASCell, MacroArchitecture]:
        """
        Repair invalid architectures.

        Args:
            architecture: Architecture to repair
            problem: NAS problem instance

        Returns:
            Repaired architecture
        """
        if isinstance(architecture, NASCell):
            # Ensure DAG property
            edges_to_remove = []
            for edge in architecture.edges:
                if edge.from_node >= edge.to_node:
                    edges_to_remove.append(edge)

            for edge in edges_to_remove:
                architecture.edges.remove(edge)

            # Ensure each node has at least one input
            for node_id in range(architecture.n_inputs,
                                architecture.n_inputs + architecture.n_nodes):
                has_input = any(e.to_node == node_id for e in architecture.edges)
                if not has_input and node_id > architecture.n_inputs:
                    # Add connection from previous node
                    from .architecture import Operation, OperationType
                    op = Operation(op_type=OperationType.SKIP_CONNECT)
                    architecture.add_edge(node_id - 1, node_id, op)

        elif isinstance(architecture, MacroArchitecture):
            # Ensure valid layer configuration
            if len(architecture.layers) == 0:
                # Add minimal architecture
                from .architecture import Layer
                architecture.layers = [
                    Layer('conv', 32, kernel_size=3),
                    Layer('fc', problem.macro_config['num_classes'])
                ]

            # Ensure final layer matches num_classes
            if architecture.layers:
                last_layer = architecture.layers[-1]
                if last_layer.layer_type == 'fc':
                    last_layer.channels = problem.macro_config['num_classes']
                else:
                    # Add final FC layer
                    from .architecture import Layer
                    architecture.layers.append(
                        Layer('fc', problem.macro_config['num_classes'])
                    )

            # Remove invalid skip connections
            valid_skips = []
            for from_idx, to_idx in architecture.skip_connections:
                if 0 <= from_idx < to_idx < len(architecture.layers):
                    valid_skips.append((from_idx, to_idx))
            architecture.skip_connections = valid_skips

        return architecture

    def _scalarize_objectives(self,
                            metrics: Dict[str, float],
                            problem: NASProblem) -> float:
        """
        Scalarize multiple objectives into single value.

        Args:
            metrics: Dictionary of metric values
            problem: NAS problem with objectives

        Returns:
            Scalarized objective value
        """
        if self.scalarization_method == 'weighted_sum':
            # Weighted sum of objectives
            value = 0
            for obj in problem.objectives:
                if obj.name in metrics:
                    value += obj.compute_score(metrics[obj.name])
            return value

        elif self.scalarization_method == 'chebyshev':
            # Chebyshev scalarization (min-max)
            values = []
            for obj in problem.objectives:
                if obj.name in metrics:
                    score = obj.compute_score(metrics[obj.name])
                    values.append(score)
            return min(values) if values else 0

        elif self.scalarization_method == 'pbi':
            # Penalty-based boundary intersection
            # Simplified version
            values = []
            for obj in problem.objectives:
                if obj.name in metrics:
                    score = obj.compute_score(metrics[obj.name])
                    values.append(score)

            if values:
                # Distance to ideal point (all objectives at max)
                ideal_distance = np.linalg.norm(np.array(values) - 1.0)
                return -ideal_distance
            return 0

        else:
            warnings.warn(f"Unknown scalarization method: {self.scalarization_method}, "
                        "using weighted_sum")
            return self._scalarize_objectives(metrics, problem)

    def get_encoding_hints(self, problem: NASProblem) -> Dict[str, Any]:
        """
        Get hints for optimizers about the encoding.

        Args:
            problem: NAS problem instance

        Returns:
            Dictionary with encoding hints
        """
        hints = {
            'dimension': problem.get_dimension(),
            'bounds': problem.get_bounds(),
            'encoding_type': 'mixed',
            'discrete_indices': [],
            'continuous_indices': []
        }

        if problem.search_space == SearchSpace.CELL:
            # Operations are discrete, channels are continuous
            # Count total possible edges
            total_edges = 0
            for i in range(problem.cell_config['n_inputs'],
                          problem.cell_config['n_inputs'] + problem.cell_config['n_nodes']):
                total_edges += i
            for i in range(total_edges):
                hints['discrete_indices'].append(i * 2)  # Operation index
                hints['continuous_indices'].append(i * 2 + 1)  # Channel size

        elif problem.search_space == SearchSpace.MACRO:
            # Mixed encoding
            hints['discrete_indices'].append(0)  # Depth
            offset = 1
            for i in range(problem.macro_config['max_layers']):
                hints['discrete_indices'].append(offset + i * 3)  # Layer type
                hints['continuous_indices'].append(offset + i * 3 + 1)  # Channels
                hints['discrete_indices'].append(offset + i * 3 + 2)  # Kernel size

        return hints

    def validate_solution(self,
                         solution: np.ndarray,
                         problem: NASProblem) -> Tuple[bool, List[str]]:
        """
        Validate a solution.

        Args:
            solution: Solution vector
            problem: NAS problem instance

        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        is_valid = True

        # Check dimension
        expected_dim = problem.get_dimension()
        if len(solution) != expected_dim:
            issues.append(f"Dimension mismatch: expected {expected_dim}, got {len(solution)}")
            is_valid = False

        # Check bounds
        bounds = problem.get_bounds()
        for i, (val, (low, high)) in enumerate(zip(solution, bounds)):
            if val < low or val > high:
                issues.append(f"Value at index {i} out of bounds: {val} not in [{low}, {high}]")
                is_valid = False

        # Try decoding
        try:
            architecture = self._decode_architecture(solution, problem)

            # Check architecture validity
            if isinstance(architecture, NASCell):
                # Check DAG property
                for edge in architecture.edges:
                    if edge.from_node >= edge.to_node:
                        issues.append(f"Invalid edge: {edge.from_node} -> {edge.to_node} "
                                    "(violates DAG property)")
                        is_valid = False

            elif isinstance(architecture, MacroArchitecture):
                # Check layer count
                if len(architecture.layers) == 0:
                    issues.append("Architecture has no layers")
                    is_valid = False

        except Exception as e:
            issues.append(f"Failed to decode architecture: {str(e)}")
            is_valid = False

        return is_valid, issues

    def reset_cache(self):
        """Clear the evaluation cache."""
        self.eval_cache = {}