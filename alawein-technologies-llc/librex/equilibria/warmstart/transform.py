"""
Solution Transformation for Warm-Starting

This module provides methods to adapt solutions from one problem to another,
enabling effective warm-starting even when problems differ in size or structure.

Transformation Methods:
1. Direct mapping - For same-size problems
2. Interpolation - For continuous problems with different sizes
3. Projection - Mapping to feasible region
4. Permutation scaling - For discrete/combinatorial problems
"""

from enum import Enum
from typing import List, Optional, Tuple

import numpy as np
from scipy.interpolate import interp1d, interp2d
from scipy.optimize import linear_sum_assignment


class TransformMethod(Enum):
    """Available transformation methods."""
    DIRECT = "direct"
    INTERPOLATE = "interpolate"
    PROJECT = "project"
    PERMUTATION_SCALE = "permutation_scale"
    FEATURE_MAP = "feature_map"
    ENSEMBLE = "ensemble"


class SolutionTransformer:
    """
    Transform solutions between different problem instances.

    This class provides various methods to adapt solutions for warm-starting,
    handling different problem sizes, types, and constraint structures.
    """

    def __init__(self):
        """Initialize solution transformer."""
        self.method_map = {
            TransformMethod.DIRECT: self._transform_direct,
            TransformMethod.INTERPOLATE: self._transform_interpolate,
            TransformMethod.PROJECT: self._transform_project,
            TransformMethod.PERMUTATION_SCALE: self._transform_permutation_scale,
            TransformMethod.FEATURE_MAP: self._transform_feature_map,
            TransformMethod.ENSEMBLE: self._transform_ensemble
        }

    def transform(self,
                  solution: np.ndarray,
                  source_problem: dict,
                  target_problem: dict,
                  method: TransformMethod = TransformMethod.DIRECT) -> np.ndarray:
        """
        Transform solution from source to target problem.

        Args:
            solution: Solution for source problem
            source_problem: Source problem definition
            target_problem: Target problem definition
            method: Transformation method to use

        Returns:
            Transformed solution for target problem
        """
        # Get transformation function
        transform_func = self.method_map.get(method, self._transform_direct)

        # Apply transformation
        transformed = transform_func(solution, source_problem, target_problem)

        # Ensure solution is valid for target problem
        transformed = self._ensure_validity(transformed, target_problem)

        return transformed

    def _transform_direct(self,
                         solution: np.ndarray,
                         source_problem: dict,
                         target_problem: dict) -> np.ndarray:
        """
        Direct mapping for same-size problems.

        This is the simplest transformation - use solution as-is.
        """
        source_size = self._get_problem_size(source_problem)
        target_size = self._get_problem_size(target_problem)

        if source_size != target_size:
            # Fall back to interpolation if sizes differ
            return self._transform_interpolate(solution, source_problem, target_problem)

        return solution.copy()

    def _transform_interpolate(self,
                               solution: np.ndarray,
                               source_problem: dict,
                               target_problem: dict) -> np.ndarray:
        """
        Interpolate solution for different problem sizes.

        This works well for continuous optimization problems.
        """
        source_size = self._get_problem_size(source_problem)
        target_size = self._get_problem_size(target_problem)

        # Handle multi-dimensional solutions
        if solution.ndim == 2:
            return self._interpolate_2d(solution, source_size, target_size)
        else:
            return self._interpolate_1d(solution, source_size, target_size)

    def _interpolate_1d(self,
                        solution: np.ndarray,
                        source_size: int,
                        target_size: int) -> np.ndarray:
        """Interpolate 1D solution vector."""
        if source_size == target_size:
            return solution.copy()

        # Create interpolation function
        source_indices = np.linspace(0, 1, source_size)
        target_indices = np.linspace(0, 1, target_size)

        # Check if solution represents a permutation
        if self._is_permutation(solution):
            # Special handling for permutations
            return self._scale_permutation(solution, target_size)

        # Standard interpolation for continuous values
        interpolator = interp1d(source_indices, solution,
                               kind='linear', fill_value='extrapolate')
        transformed = interpolator(target_indices)

        return transformed

    def _interpolate_2d(self,
                        solution: np.ndarray,
                        source_shape: Tuple,
                        target_shape: Tuple) -> np.ndarray:
        """Interpolate 2D solution matrix."""
        if solution.shape == target_shape:
            return solution.copy()

        # Create coordinate grids
        source_x = np.linspace(0, 1, source_shape[0])
        source_y = np.linspace(0, 1, source_shape[1])
        target_x = np.linspace(0, 1, target_shape[0])
        target_y = np.linspace(0, 1, target_shape[1])

        # 2D interpolation
        interpolator = interp2d(source_y, source_x, solution, kind='linear')
        transformed = interpolator(target_y, target_x)

        return transformed

    def _transform_project(self,
                          solution: np.ndarray,
                          source_problem: dict,
                          target_problem: dict) -> np.ndarray:
        """
        Project solution to feasible region of target problem.

        This handles constraint differences between problems.
        """
        transformed = self._transform_interpolate(solution, source_problem, target_problem)

        # Project to constraint bounds if specified
        if 'bounds' in target_problem:
            bounds = target_problem['bounds']
            if isinstance(bounds, tuple):
                lower, upper = bounds
                transformed = np.clip(transformed, lower, upper)
            elif isinstance(bounds, np.ndarray) and bounds.shape[0] == 2:
                transformed = np.clip(transformed, bounds[0], bounds[1])

        # Handle specific constraint types
        problem_type = target_problem.get('type', '')

        if problem_type == 'qap' or problem_type == 'tsp':
            # Ensure permutation constraint
            if not self._is_permutation(transformed):
                transformed = self._project_to_permutation(transformed)

        elif 'constraint_matrix' in target_problem:
            # Linear constraints: Ax <= b
            A = target_problem['constraint_matrix']
            b = target_problem.get('constraint_bounds', np.zeros(len(A)))
            transformed = self._project_linear_constraints(transformed, A, b)

        return transformed

    def _transform_permutation_scale(self,
                                     solution: np.ndarray,
                                     source_problem: dict,
                                     target_problem: dict) -> np.ndarray:
        """
        Scale permutation solution to different size.

        This is specialized for combinatorial problems like QAP/TSP.
        """
        source_size = self._get_problem_size(source_problem)
        target_size = self._get_problem_size(target_problem)

        if not self._is_permutation(solution):
            # Convert to permutation first
            solution = self._project_to_permutation(solution)

        return self._scale_permutation(solution, target_size)

    def _scale_permutation(self, perm: np.ndarray, new_size: int) -> np.ndarray:
        """
        Scale a permutation to a different size.

        Uses relative positioning to maintain structure.
        """
        old_size = len(perm)

        if old_size == new_size:
            return perm.copy()

        # Compute relative positions in [0, 1]
        relative_positions = perm / (old_size - 1) if old_size > 1 else np.zeros(old_size)

        # Map to new size
        new_positions = relative_positions * (new_size - 1)

        # Create new permutation maintaining relative order
        sorted_indices = np.argsort(new_positions)

        if new_size > old_size:
            # Expanding - interpolate missing positions
            new_perm = np.zeros(new_size, dtype=int)

            # Place original elements
            for i, idx in enumerate(sorted_indices[:old_size]):
                new_perm[idx] = i

            # Fill remaining positions
            used = set(new_perm[:old_size])
            available = [i for i in range(new_size) if i not in used]

            j = 0
            for i in range(new_size):
                if i >= old_size:
                    new_perm[sorted_indices[i]] = available[j]
                    j += 1

        else:
            # Shrinking - select subset maintaining order
            selected_indices = sorted_indices[:new_size]
            new_perm = np.argsort(selected_indices)

        return new_perm

    def _transform_feature_map(self,
                               solution: np.ndarray,
                               source_problem: dict,
                               target_problem: dict) -> np.ndarray:
        """
        Transform using feature-based mapping.

        This uses problem features to guide the transformation.
        """
        # Extract features from both problems
        source_features = self._extract_problem_features(source_problem)
        target_features = self._extract_problem_features(target_problem)

        # Compute feature-based scaling
        if len(source_features) > 0 and len(target_features) > 0:
            # Use feature ratio to scale solution
            feature_ratio = np.mean(target_features) / (np.mean(source_features) + 1e-10)
            transformed = solution * feature_ratio

            # Adjust for size difference
            source_size = self._get_problem_size(source_problem)
            target_size = self._get_problem_size(target_problem)

            if source_size != target_size:
                transformed = self._transform_interpolate(
                    transformed, source_problem, target_problem
                )
        else:
            # Fall back to interpolation
            transformed = self._transform_interpolate(
                solution, source_problem, target_problem
            )

        return transformed

    def _transform_ensemble(self,
                           solution: np.ndarray,
                           source_problem: dict,
                           target_problem: dict) -> np.ndarray:
        """
        Ensemble of multiple transformation methods.

        Combines multiple transformations for robustness.
        """
        methods = [
            TransformMethod.INTERPOLATE,
            TransformMethod.PROJECT,
            TransformMethod.FEATURE_MAP
        ]

        transformed_solutions = []
        weights = []

        for method in methods:
            try:
                transformed = self.method_map[method](
                    solution, source_problem, target_problem
                )
                transformed_solutions.append(transformed)

                # Weight based on method reliability
                if method == TransformMethod.PROJECT:
                    weights.append(2.0)  # Higher weight for projection
                else:
                    weights.append(1.0)
            except:
                continue

        if not transformed_solutions:
            # Fall back to direct
            return self._transform_direct(solution, source_problem, target_problem)

        # Weighted average for continuous problems
        weights = np.array(weights)
        weights /= weights.sum()

        if self._is_continuous_problem(target_problem):
            result = np.zeros_like(transformed_solutions[0])
            for sol, w in zip(transformed_solutions, weights):
                result += w * sol
            return result
        else:
            # For discrete problems, use best transformation
            # (one with highest weight)
            best_idx = np.argmax(weights)
            return transformed_solutions[best_idx]

    def _is_permutation(self, solution: np.ndarray) -> bool:
        """Check if solution is a permutation."""
        if solution.ndim != 1:
            return False

        n = len(solution)

        # Check if all integers in range [0, n)
        if not np.all(solution == solution.astype(int)):
            return False

        sorted_sol = np.sort(solution.astype(int))
        return np.array_equal(sorted_sol, np.arange(n))

    def _project_to_permutation(self, solution: np.ndarray) -> np.ndarray:
        """Project solution to nearest permutation."""
        n = len(solution)

        if self._is_permutation(solution):
            return solution.astype(int)

        # Interpret as assignment probabilities
        if solution.ndim == 1:
            # Create cost matrix from solution values
            cost_matrix = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    cost_matrix[i, j] = abs(solution[i] - j)
        else:
            # Assume solution is assignment matrix
            cost_matrix = -solution  # Maximize becomes minimize

        # Solve assignment problem
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        return col_ind

    def _project_linear_constraints(self,
                                    solution: np.ndarray,
                                    A: np.ndarray,
                                    b: np.ndarray) -> np.ndarray:
        """
        Project to satisfy linear constraints Ax <= b.

        Uses simple projection - more sophisticated methods
        could use quadratic programming.
        """
        # Check constraint violations
        violations = A @ solution - b

        if np.all(violations <= 0):
            return solution

        # Simple projection: scale down violated components
        projected = solution.copy()

        for i, violation in enumerate(violations):
            if violation > 0:
                # Find the constraint normal
                normal = A[i]
                # Project solution onto constraint boundary
                adjustment = violation / (np.linalg.norm(normal)**2 + 1e-10)
                projected -= adjustment * normal

        return projected

    def _get_problem_size(self, problem: dict) -> int:
        """Extract problem size."""
        if 'dimension' in problem:
            return problem['dimension']

        for key in ['flow_matrix', 'distance_matrix', 'cost_matrix']:
            if key in problem and isinstance(problem[key], np.ndarray):
                return len(problem[key])

        if 'n' in problem:
            return problem['n']

        return 0

    def _extract_problem_features(self, problem: dict) -> np.ndarray:
        """Extract numerical features from problem."""
        features = []

        # Add size
        features.append(float(self._get_problem_size(problem)))

        # Add matrix properties
        for key in ['flow_matrix', 'distance_matrix', 'cost_matrix']:
            if key in problem and isinstance(problem[key], np.ndarray):
                matrix = problem[key]
                features.append(np.mean(np.abs(matrix)))
                features.append(np.std(matrix))
                features.append(np.max(np.abs(matrix)))

        return np.array(features)

    def _is_continuous_problem(self, problem: dict) -> bool:
        """Check if problem is continuous optimization."""
        problem_type = problem.get('type', '')

        discrete_types = ['qap', 'tsp', 'knapsack', 'sat', 'coloring']
        return problem_type not in discrete_types

    def _ensure_validity(self, solution: np.ndarray, problem: dict) -> np.ndarray:
        """Ensure solution satisfies basic validity requirements."""
        problem_type = problem.get('type', '')

        if problem_type in ['qap', 'tsp']:
            # Ensure permutation
            if not self._is_permutation(solution):
                solution = self._project_to_permutation(solution)

        elif 'bounds' in problem:
            # Ensure bounds
            bounds = problem['bounds']
            if isinstance(bounds, tuple):
                lower, upper = bounds
                solution = np.clip(solution, lower, upper)

        return solution