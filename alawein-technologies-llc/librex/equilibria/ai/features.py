"""
Problem feature extraction for method selection

This module analyzes optimization problems to extract features that
help determine which optimization method will perform best.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from Librex.core.interfaces import StandardizedProblem, UniversalOptimizationInterface

logger = logging.getLogger(__name__)


@dataclass
class ProblemFeatures:
    """Features extracted from an optimization problem"""

    # Size features
    dimension: int
    total_elements: int

    # Matrix structure features (if applicable)
    has_matrix: bool
    matrix_sparsity: float  # Fraction of zero elements
    matrix_symmetry: float  # Degree of symmetry (0=asymmetric, 1=symmetric)
    diagonal_dominance: float  # How dominant the diagonal is

    # Value distribution features
    value_range: float  # Max - min value
    value_mean: float
    value_std: float
    value_skewness: float
    value_kurtosis: float

    # Constraint features
    has_constraints: bool
    constraint_density: float  # Fraction of non-zero constraint coefficients
    constraint_count: int

    # Problem type indicators
    is_permutation: bool  # Whether problem involves permutations
    is_binary: bool  # Whether problem involves binary variables
    is_continuous: bool  # Whether problem has continuous variables

    # Connectivity features (for graph-like problems)
    connectivity: float  # Average connectivity
    clustering_coefficient: float  # Local clustering

    # Additional metadata
    problem_type: str  # E.g., 'QAP', 'TSP', 'generic'
    estimated_difficulty: str  # 'easy', 'medium', 'hard', 'very_hard'


class ProblemFeatureExtractor:
    """Extract features from optimization problems"""

    def __init__(self):
        """Initialize the feature extractor"""
        self.feature_cache: Dict[int, ProblemFeatures] = {}

    def extract_features(
        self,
        problem: Any,
        adapter: Optional[UniversalOptimizationInterface] = None,
        standardized_problem: Optional[StandardizedProblem] = None
    ) -> ProblemFeatures:
        """
        Extract features from an optimization problem

        Args:
            problem: Original problem (domain-specific)
            adapter: Domain adapter (required if standardized_problem not provided)
            standardized_problem: Already standardized problem (optional)

        Returns:
            ProblemFeatures: Extracted features
        """
        # Standardize problem if needed
        if standardized_problem is None:
            if adapter is None:
                raise ValueError("Either adapter or standardized_problem must be provided")
            standardized_problem = adapter.encode_problem(problem)

        # Check cache
        cache_key = id(standardized_problem)
        if cache_key in self.feature_cache:
            logger.debug("Using cached features")
            return self.feature_cache[cache_key]

        # Extract basic features
        dimension = standardized_problem.dimension

        # Initialize features
        features = ProblemFeatures(
            dimension=dimension,
            total_elements=dimension * dimension if standardized_problem.objective_matrix is not None else dimension,
            has_matrix=standardized_problem.objective_matrix is not None,
            matrix_sparsity=0.0,
            matrix_symmetry=0.0,
            diagonal_dominance=0.0,
            value_range=0.0,
            value_mean=0.0,
            value_std=0.0,
            value_skewness=0.0,
            value_kurtosis=0.0,
            has_constraints=standardized_problem.constraint_matrix is not None,
            constraint_density=0.0,
            constraint_count=0,
            is_permutation=False,
            is_binary=False,
            is_continuous=False,
            connectivity=0.0,
            clustering_coefficient=0.0,
            problem_type='generic',
            estimated_difficulty='medium'
        )

        # Extract matrix features
        if standardized_problem.objective_matrix is not None:
            features = self._extract_matrix_features(
                standardized_problem.objective_matrix, features
            )

        # Extract constraint features
        if standardized_problem.constraint_matrix is not None:
            features = self._extract_constraint_features(
                standardized_problem.constraint_matrix, features
            )

        # Determine problem type
        features = self._determine_problem_type(problem, adapter, features)

        # Estimate difficulty
        features = self._estimate_difficulty(features)

        # Cache the features
        self.feature_cache[cache_key] = features

        logger.info(f"Extracted features for {features.dimension}D problem")
        return features

    def _extract_matrix_features(
        self, matrix: np.ndarray, features: ProblemFeatures
    ) -> ProblemFeatures:
        """Extract features from the objective matrix"""
        if matrix.size == 0:
            return features

        # Flatten matrix for statistics
        values = matrix.flatten()
        non_zero_values = values[values != 0]

        # Sparsity
        features.matrix_sparsity = np.sum(values == 0) / len(values) if len(values) > 0 else 0.0

        # Symmetry (for square matrices)
        if matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1]:
            symmetry_diff = np.abs(matrix - matrix.T)
            max_diff = np.abs(matrix).max()
            if max_diff > 0:
                features.matrix_symmetry = 1.0 - (symmetry_diff.sum() / (2 * matrix.sum()))
            else:
                features.matrix_symmetry = 1.0

            # Diagonal dominance
            if matrix.shape[0] > 1:
                diagonal = np.abs(np.diag(matrix))
                off_diagonal_sum = np.abs(matrix).sum(axis=1) - diagonal
                if off_diagonal_sum.sum() > 0:
                    features.diagonal_dominance = diagonal.sum() / (diagonal.sum() + off_diagonal_sum.sum())

        # Value distribution
        if len(non_zero_values) > 0:
            features.value_range = non_zero_values.max() - non_zero_values.min()
            features.value_mean = non_zero_values.mean()
            features.value_std = non_zero_values.std()

            # Skewness and kurtosis (requires at least 3 and 4 values respectively)
            if len(non_zero_values) >= 3:
                from scipy import stats
                features.value_skewness = float(stats.skew(non_zero_values))
                if len(non_zero_values) >= 4:
                    features.value_kurtosis = float(stats.kurtosis(non_zero_values))

        # Connectivity (treat as adjacency matrix)
        if matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1]:
            adjacency = (matrix != 0).astype(float)
            degree = adjacency.sum(axis=1)
            features.connectivity = degree.mean() / (matrix.shape[0] - 1) if matrix.shape[0] > 1 else 0.0

            # Clustering coefficient
            if matrix.shape[0] >= 3:
                features.clustering_coefficient = self._compute_clustering_coefficient(adjacency)

        return features

    def _extract_constraint_features(
        self, constraint_matrix: np.ndarray, features: ProblemFeatures
    ) -> ProblemFeatures:
        """Extract features from constraint matrix"""
        if constraint_matrix.size == 0:
            return features

        features.constraint_count = constraint_matrix.shape[0] if constraint_matrix.ndim > 1 else 1
        non_zeros = np.sum(constraint_matrix != 0)
        total_elements = constraint_matrix.size
        features.constraint_density = non_zeros / total_elements if total_elements > 0 else 0.0

        return features

    def _determine_problem_type(
        self, problem: Any, adapter: Optional[UniversalOptimizationInterface],
        features: ProblemFeatures
    ) -> ProblemFeatures:
        """Determine the type of optimization problem"""
        # Check adapter type
        if adapter is not None:
            adapter_name = adapter.__class__.__name__.lower()
            if 'qap' in adapter_name:
                features.problem_type = 'QAP'
                features.is_permutation = True
            elif 'tsp' in adapter_name:
                features.problem_type = 'TSP'
                features.is_permutation = True
            elif 'knapsack' in adapter_name:
                features.problem_type = 'knapsack'
                features.is_binary = True
            elif 'linear' in adapter_name:
                features.problem_type = 'linear'
                features.is_continuous = True

        # Check problem structure
        if isinstance(problem, dict):
            if 'flow_matrix' in problem and 'distance_matrix' in problem:
                features.problem_type = 'QAP'
                features.is_permutation = True
            elif 'distances' in problem or 'distance_matrix' in problem:
                features.problem_type = 'TSP'
                features.is_permutation = True

        return features

    def _estimate_difficulty(self, features: ProblemFeatures) -> ProblemFeatures:
        """Estimate problem difficulty based on features"""
        score = 0.0

        # Size-based difficulty
        if features.dimension < 10:
            score += 0.0
        elif features.dimension < 50:
            score += 1.0
        elif features.dimension < 100:
            score += 2.0
        elif features.dimension < 500:
            score += 3.0
        else:
            score += 4.0

        # Constraint-based difficulty
        if features.has_constraints:
            score += 1.0
            score += features.constraint_density * 2.0

        # Structure-based difficulty
        if features.matrix_sparsity < 0.3:
            score += 1.0  # Dense problems are harder

        if abs(features.matrix_symmetry - 0.5) > 0.3:
            score += 0.5  # Very symmetric or asymmetric is easier

        # High connectivity makes problems harder
        score += features.connectivity * 2.0

        # Normalize and categorize
        max_score = 10.0
        normalized_score = min(score / max_score, 1.0)

        if normalized_score < 0.25:
            features.estimated_difficulty = 'easy'
        elif normalized_score < 0.5:
            features.estimated_difficulty = 'medium'
        elif normalized_score < 0.75:
            features.estimated_difficulty = 'hard'
        else:
            features.estimated_difficulty = 'very_hard'

        return features

    def _compute_clustering_coefficient(self, adjacency: np.ndarray) -> float:
        """Compute the average clustering coefficient"""
        n = adjacency.shape[0]
        clustering_coeffs = []

        for i in range(n):
            neighbors = np.where(adjacency[i] != 0)[0]
            k = len(neighbors)

            if k < 2:
                clustering_coeffs.append(0.0)
                continue

            # Count edges between neighbors
            edges = 0
            for j in range(k):
                for l in range(j + 1, k):
                    if adjacency[neighbors[j], neighbors[l]] != 0:
                        edges += 1

            # Clustering coefficient for node i
            max_edges = k * (k - 1) / 2
            clustering_coeffs.append(edges / max_edges if max_edges > 0 else 0.0)

        return np.mean(clustering_coeffs) if clustering_coeffs else 0.0

    def get_feature_vector(self, features: ProblemFeatures) -> np.ndarray:
        """
        Convert ProblemFeatures to a numerical feature vector for ML models

        Args:
            features: Problem features

        Returns:
            np.ndarray: Feature vector
        """
        vector = np.array([
            features.dimension,
            np.log1p(features.total_elements),  # Log scale for large values
            float(features.has_matrix),
            features.matrix_sparsity,
            features.matrix_symmetry,
            features.diagonal_dominance,
            np.log1p(features.value_range),  # Log scale for ranges
            features.value_mean,
            np.log1p(features.value_std),  # Log scale for std
            features.value_skewness,
            features.value_kurtosis,
            float(features.has_constraints),
            features.constraint_density,
            features.constraint_count,
            float(features.is_permutation),
            float(features.is_binary),
            float(features.is_continuous),
            features.connectivity,
            features.clustering_coefficient,
            # Encode difficulty as numerical
            {'easy': 0.25, 'medium': 0.5, 'hard': 0.75, 'very_hard': 1.0}.get(
                features.estimated_difficulty, 0.5
            )
        ])

        return vector