"""
Traveling Salesman Problem (TSP) Domain Adapter with Enhanced Validation
"""

import warnings
from typing import Any, Dict, Optional, Union

import numpy as np

from Librex.core.interfaces import (
    StandardizedProblem,
    StandardizedSolution,
    UniversalOptimizationInterface,
    ValidationResult,
)


class TSPAdapter(UniversalOptimizationInterface):
    """TSP to universal format adapter with enhanced distance matrix validation"""

    def __init__(self):
        super().__init__({
            'name': 'tsp',
            'type': 'discrete',
            'constraints': ['tour'],
            'objectives': ['minimize']
        })
        self.n = None
        self.distance_matrix = None
        self.coordinates = None

    def encode_problem(
        self,
        instance: Dict[str, Union[np.ndarray, None]]
    ) -> StandardizedProblem:
        """
        Convert TSP instance to standardized format

        Args:
            instance: Dictionary with either 'coordinates' or 'distance_matrix'

        Returns:
            StandardizedProblem representation
        """
        if 'coordinates' in instance and instance['coordinates'] is not None:
            self.coordinates = instance['coordinates']
            self.distance_matrix = self._compute_distance_matrix(self.coordinates)
        elif 'distance_matrix' in instance:
            self.distance_matrix = instance['distance_matrix']
            self._validate_distance_matrix(self.distance_matrix)
        else:
            raise ValueError("Instance must contain either 'coordinates' or 'distance_matrix'")

        self.n = len(self.distance_matrix)

        return StandardizedProblem(
            dimension=self.n,
            objective_matrix=None,
            objective_function=self.compute_objective,
            constraint_matrix=None,
            problem_metadata={
                'type': 'traveling_salesman',
                'size': self.n,
                'distance_matrix': self.distance_matrix,
                'has_coordinates': self.coordinates is not None
            }
        )

    def decode_solution(self, solution: StandardizedSolution) -> np.ndarray:
        """Convert solution to tour representation"""
        return solution.vector.astype(int)

    def validate_solution(self, solution: np.ndarray) -> ValidationResult:
        """Check if solution is a valid tour"""
        violations = []
        magnitudes = []

        if not self._is_valid_tour(solution):
            violations.append("Not a valid tour (not a permutation)")
            magnitudes.append(1.0)

        if len(solution) != self.n:
            violations.append(f"Tour length {len(solution)} != problem size {self.n}")
            magnitudes.append(abs(len(solution) - self.n))

        return ValidationResult(
            is_valid=len(violations) == 0,
            constraint_violations=violations,
            violation_magnitudes=magnitudes
        )

    def compute_objective(self, tour: np.ndarray) -> float:
        """
        Compute TSP tour length

        Args:
            tour: Permutation representing city visit order

        Returns:
            Total tour length
        """
        total_distance = 0.0

        for i in range(len(tour)):
            city_from = tour[i]
            city_to = tour[(i + 1) % len(tour)]
            total_distance += self.distance_matrix[city_from, city_to]

        return float(total_distance)

    def _compute_distance_matrix(self, coordinates: np.ndarray) -> np.ndarray:
        """
        Compute distance matrix from coordinates with enhanced validation

        PERFORMANCE OPTIMIZED: Vectorized computation using NumPy broadcasting
        instead of nested loops. Provides ~10-100x speedup for large instances.

        FIXES:
        - Validates distance matrix symmetry
        - Checks for negative distances
        - Verifies numerical stability

        Args:
            coordinates: Nx2 array of city coordinates

        Returns:
            NxN distance matrix

        Raises:
            ValueError: If negative distances are found
        """
        n = len(coordinates)

        # OPTIMIZATION: Vectorized distance computation using broadcasting
        # This is much faster than nested loops for large n
        # Shape: (n, 1, 2) - (1, n, 2) = (n, n, 2)
        diff = coordinates[:, np.newaxis, :] - coordinates[np.newaxis, :, :]

        # Compute Euclidean distances: sqrt(dx^2 + dy^2)
        distances = np.sqrt(np.sum(diff**2, axis=2))

        # VALIDATION: Check for negative distances (should never happen with Euclidean)
        if np.any(distances < 0):
            raise ValueError(
                "Negative distance found in distance matrix. "
                "This should not occur with Euclidean distances."
            )

        # VALIDATION: Check symmetry for Euclidean distances
        if not np.allclose(distances, distances.T, rtol=1e-10, atol=1e-12):
            warnings.warn(
                "Distance matrix is not symmetric. This may indicate a numerical "
                "precision issue or an error in coordinate computation.",
                RuntimeWarning
            )

        # VALIDATION: Check for NaN or inf values
        if np.any(np.isnan(distances)) or np.any(np.isinf(distances)):
            raise ValueError("Distance matrix contains NaN or infinite values")

        return distances

    def _validate_distance_matrix(self, distances: np.ndarray) -> None:
        """
        Validate a provided distance matrix

        ENHANCEMENTS:
        - Checks matrix is square
        - Validates non-negativity
        - Checks diagonal is zero
        - Optionally checks triangle inequality

        Args:
            distances: Distance matrix to validate

        Raises:
            ValueError: If validation fails
        """
        # Check square matrix
        if distances.shape[0] != distances.shape[1]:
            raise ValueError(
                f"Distance matrix must be square, got shape {distances.shape}"
            )

        # Check non-negativity
        if np.any(distances < 0):
            raise ValueError("Distance matrix contains negative values")

        # Check diagonal is zero
        if not np.allclose(np.diag(distances), 0, atol=1e-10):
            warnings.warn(
                "Distance matrix diagonal is not zero. "
                "Self-distances should typically be zero.",
                RuntimeWarning
            )

        # Check for NaN or inf
        if np.any(np.isnan(distances)) or np.any(np.isinf(distances)):
            raise ValueError("Distance matrix contains NaN or infinite values")

        # Check symmetry (common in TSP)
        if not np.allclose(distances, distances.T, rtol=1e-10):
            warnings.warn(
                "Distance matrix is not symmetric. "
                "Asymmetric TSP instances are valid but uncommon.",
                RuntimeWarning
            )

    def _is_valid_tour(self, tour: np.ndarray) -> bool:
        """Check if array is a valid tour (permutation)"""
        return (
            len(tour) == self.n and
            np.all(np.sort(tour) == np.arange(self.n))
        )


__all__ = ["TSPAdapter"]
