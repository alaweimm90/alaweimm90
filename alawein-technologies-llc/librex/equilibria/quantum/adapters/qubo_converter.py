"""
QUBO (Quadratic Unconstrained Binary Optimization) Converter

Converts classical combinatorial optimization problems to QUBO formulation
suitable for quantum annealing and other quantum optimization algorithms.

Mathematical Background:
QUBO problems have the form:
    min x^T Q x
where x is a binary vector and Q is a symmetric matrix.

Author: Meshal Alawein
Date: 2025-11-18
"""

import logging
from typing import Dict, Any, Tuple, Optional
import numpy as np
from dataclasses import dataclass

from Librex.core.interfaces import StandardizedProblem

logger = logging.getLogger(__name__)


@dataclass
class QUBOProblem:
    """
    QUBO problem representation.

    Attributes:
        Q: Symmetric QUBO matrix
        offset: Constant offset for the objective function
        num_variables: Number of binary variables
        metadata: Additional problem metadata
    """
    Q: np.ndarray
    offset: float = 0.0
    num_variables: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.num_variables is None:
            self.num_variables = self.Q.shape[0]
        # Ensure Q is symmetric
        self.Q = (self.Q + self.Q.T) / 2

    def evaluate(self, x: np.ndarray) -> float:
        """Evaluate QUBO objective for a binary vector x."""
        return float(x.T @ self.Q @ x + self.offset)


class QUBOConverter:
    """
    Converts optimization problems to QUBO formulation.

    This converter supports:
    - Quadratic Assignment Problem (QAP)
    - Traveling Salesman Problem (TSP)
    - Max-Cut Problem
    - Graph Coloring
    - Binary Quadratic Programming

    The conversion often requires penalty terms to handle constraints.
    """

    def __init__(self, penalty_weight: float = 1000.0):
        """
        Initialize QUBO converter.

        Args:
            penalty_weight: Weight for penalty terms in constraint handling
        """
        self.penalty_weight = penalty_weight
        logger.info(f"Initialized QUBO converter with penalty weight: {penalty_weight}")

    def convert_qap_to_qubo(
        self,
        flow_matrix: np.ndarray,
        distance_matrix: np.ndarray
    ) -> QUBOProblem:
        """
        Convert Quadratic Assignment Problem to QUBO.

        The QAP seeks to assign n facilities to n locations minimizing:
            sum_i,j flow[i,j] * distance[π(i),π(j)]

        We use binary variables x[i,j] = 1 if facility i is assigned to location j.

        Constraints (enforced via penalties):
        - Each facility assigned to exactly one location: sum_j x[i,j] = 1
        - Each location gets exactly one facility: sum_i x[i,j] = 1

        Args:
            flow_matrix: n×n matrix of flows between facilities
            distance_matrix: n×n matrix of distances between locations

        Returns:
            QUBOProblem: QUBO formulation of the QAP

        TODO: Implement advanced penalty tuning based on problem structure
        TODO: Add support for partial assignment problems
        """
        n = flow_matrix.shape[0]
        num_vars = n * n

        # Initialize QUBO matrix
        Q = np.zeros((num_vars, num_vars))

        # Objective function terms
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    for l in range(n):
                        # Variable indices
                        var1 = i * n + k  # x[i,k]
                        var2 = j * n + l  # x[j,l]
                        # Add flow * distance term
                        Q[var1, var2] += flow_matrix[i, j] * distance_matrix[k, l]

        # Penalty terms for row constraints (each facility to one location)
        for i in range(n):
            # Linear penalty term: -2 * penalty * x[i,j]
            for j in range(n):
                var = i * n + j
                Q[var, var] -= 2 * self.penalty_weight

            # Quadratic penalty terms: +penalty * x[i,j] * x[i,k] for j != k
            for j in range(n):
                for k in range(j + 1, n):
                    var1 = i * n + j
                    var2 = i * n + k
                    Q[var1, var2] += 2 * self.penalty_weight
                    Q[var2, var1] += 2 * self.penalty_weight

        # Penalty terms for column constraints (each location gets one facility)
        for j in range(n):
            # Linear penalty term
            for i in range(n):
                var = i * n + j
                Q[var, var] -= 2 * self.penalty_weight

            # Quadratic penalty terms
            for i in range(n):
                for k in range(i + 1, n):
                    var1 = i * n + j
                    var2 = k * n + j
                    Q[var1, var2] += 2 * self.penalty_weight
                    Q[var2, var1] += 2 * self.penalty_weight

        # Make Q symmetric (should already be, but ensure numerical stability)
        Q = (Q + Q.T) / 2

        # Constant offset from penalty terms (n facilities + n locations)
        offset = 2 * n * self.penalty_weight

        metadata = {
            'problem_type': 'QAP',
            'n_facilities': n,
            'n_locations': n,
            'penalty_weight': self.penalty_weight,
            'n_binary_vars': num_vars,
        }

        logger.info(f"Converted QAP (n={n}) to QUBO with {num_vars} binary variables")

        return QUBOProblem(Q=Q, offset=offset, metadata=metadata)

    def convert_tsp_to_qubo(
        self,
        distance_matrix: np.ndarray
    ) -> QUBOProblem:
        """
        Convert Traveling Salesman Problem to QUBO.

        Uses binary variables x[i,t] = 1 if city i is visited at time t.

        Constraints:
        - Each city visited exactly once: sum_t x[i,t] = 1
        - Each time slot has exactly one city: sum_i x[i,t] = 1

        Args:
            distance_matrix: n×n matrix of distances between cities

        Returns:
            QUBOProblem: QUBO formulation of the TSP

        TODO: Implement subtour elimination constraints
        TODO: Add support for asymmetric TSP
        """
        n = distance_matrix.shape[0]
        num_vars = n * n

        # Initialize QUBO matrix
        Q = np.zeros((num_vars, num_vars))

        # Objective: minimize total distance
        for t in range(n):
            t_next = (t + 1) % n  # Circular tour
            for i in range(n):
                for j in range(n):
                    var1 = i * n + t       # x[i,t]
                    var2 = j * n + t_next  # x[j,t+1]
                    Q[var1, var2] += distance_matrix[i, j]

        # Penalty for visiting each city exactly once
        for i in range(n):
            # Linear penalty
            for t in range(n):
                var = i * n + t
                Q[var, var] -= 2 * self.penalty_weight

            # Quadratic penalty
            for t1 in range(n):
                for t2 in range(t1 + 1, n):
                    var1 = i * n + t1
                    var2 = i * n + t2
                    Q[var1, var2] += 2 * self.penalty_weight
                    Q[var2, var1] += 2 * self.penalty_weight

        # Penalty for each time slot having exactly one city
        for t in range(n):
            # Linear penalty
            for i in range(n):
                var = i * n + t
                Q[var, var] -= 2 * self.penalty_weight

            # Quadratic penalty
            for i1 in range(n):
                for i2 in range(i1 + 1, n):
                    var1 = i1 * n + t
                    var2 = i2 * n + t
                    Q[var1, var2] += 2 * self.penalty_weight
                    Q[var2, var1] += 2 * self.penalty_weight

        # Make symmetric
        Q = (Q + Q.T) / 2

        offset = 2 * n * self.penalty_weight

        metadata = {
            'problem_type': 'TSP',
            'n_cities': n,
            'penalty_weight': self.penalty_weight,
            'n_binary_vars': num_vars,
        }

        logger.info(f"Converted TSP (n={n}) to QUBO with {num_vars} binary variables")

        return QUBOProblem(Q=Q, offset=offset, metadata=metadata)

    def convert_maxcut_to_qubo(
        self,
        adjacency_matrix: np.ndarray
    ) -> QUBOProblem:
        """
        Convert Max-Cut problem to QUBO.

        Max-Cut seeks to partition vertices into two sets maximizing edges between sets.
        Uses binary variables x[i] = 0 or 1 for vertex partition assignment.

        The QUBO formulation: maximize sum of w[i,j] * (x[i] + x[j] - 2*x[i]*x[j])

        Args:
            adjacency_matrix: Weighted adjacency matrix of the graph

        Returns:
            QUBOProblem: QUBO formulation of Max-Cut

        TODO: Add support for constrained versions (e.g., balanced cut)
        """
        n = adjacency_matrix.shape[0]
        Q = np.zeros((n, n))

        # Convert Max-Cut to minimization QUBO
        # For edge (i,j) with weight w[i,j], the cut contribution is:
        # w[i,j] if x[i] != x[j], 0 otherwise
        # This equals w[i,j] * (x[i] + x[j] - 2*x[i]*x[j])
        # To maximize, we minimize the negative

        for i in range(n):
            for j in range(i + 1, n):
                weight = adjacency_matrix[i, j]
                if weight != 0:
                    # Linear terms
                    Q[i, i] -= weight
                    Q[j, j] -= weight
                    # Quadratic term
                    Q[i, j] += 2 * weight
                    Q[j, i] += 2 * weight

        metadata = {
            'problem_type': 'Max-Cut',
            'n_vertices': n,
            'n_edges': np.count_nonzero(adjacency_matrix) // 2,
            'n_binary_vars': n,
        }

        logger.info(f"Converted Max-Cut (n={n}) to QUBO")

        return QUBOProblem(Q=Q, offset=0.0, metadata=metadata)

    def convert_from_standardized(
        self,
        problem: StandardizedProblem
    ) -> QUBOProblem:
        """
        Convert a StandardizedProblem to QUBO format.

        Attempts to detect the problem type and apply appropriate conversion.

        Args:
            problem: Standardized optimization problem

        Returns:
            QUBOProblem: QUBO formulation

        TODO: Implement automatic problem type detection
        TODO: Add support for more problem types
        """
        metadata = problem.problem_metadata or {}
        problem_type = metadata.get('problem_type', 'unknown')

        if problem_type == 'QAP' and 'flow_matrix' in metadata and 'distance_matrix' in metadata:
            return self.convert_qap_to_qubo(
                metadata['flow_matrix'],
                metadata['distance_matrix']
            )
        elif problem_type == 'TSP' and problem.objective_matrix is not None:
            return self.convert_tsp_to_qubo(problem.objective_matrix)
        elif problem_type == 'Max-Cut' and problem.objective_matrix is not None:
            return self.convert_maxcut_to_qubo(problem.objective_matrix)
        else:
            # Generic quadratic problem conversion
            logger.warning(f"Unknown problem type '{problem_type}', attempting generic conversion")
            return self._generic_quadratic_conversion(problem)

    def _generic_quadratic_conversion(
        self,
        problem: StandardizedProblem
    ) -> QUBOProblem:
        """
        Generic conversion for quadratic problems.

        Args:
            problem: Standardized problem with objective_matrix

        Returns:
            QUBOProblem: Generic QUBO formulation

        TODO: Implement constraint handling for generic problems
        """
        if problem.objective_matrix is None:
            raise ValueError("Cannot convert problem without objective_matrix to QUBO")

        Q = problem.objective_matrix.copy()
        Q = (Q + Q.T) / 2  # Ensure symmetric

        metadata = {
            'problem_type': 'generic',
            'dimension': problem.dimension,
            'has_constraints': problem.constraint_matrix is not None,
        }

        if problem.constraint_matrix is not None:
            logger.warning("Generic QUBO conversion does not handle constraints. "
                        "Consider implementing problem-specific converter.")

        return QUBOProblem(Q=Q, offset=0.0, metadata=metadata)

    def decode_qubo_solution(
        self,
        qubo_solution: np.ndarray,
        qubo_problem: QUBOProblem
    ) -> Dict[str, Any]:
        """
        Decode QUBO binary solution back to problem-specific format.

        Args:
            qubo_solution: Binary solution vector
            qubo_problem: Original QUBO problem

        Returns:
            Dict containing decoded solution

        TODO: Implement decoders for each problem type
        """
        metadata = qubo_problem.metadata or {}
        problem_type = metadata.get('problem_type', 'unknown')

        if problem_type == 'QAP':
            n = metadata['n_facilities']
            assignment = np.zeros(n, dtype=int)
            for i in range(n):
                for j in range(n):
                    if qubo_solution[i * n + j] > 0.5:
                        assignment[i] = j
            return {
                'assignment': assignment,
                'objective': qubo_problem.evaluate(qubo_solution),
                'is_valid': self._validate_qap_assignment(assignment),
            }
        elif problem_type == 'TSP':
            n = metadata['n_cities']
            tour = np.zeros(n, dtype=int)
            for t in range(n):
                for i in range(n):
                    if qubo_solution[i * n + t] > 0.5:
                        tour[t] = i
            return {
                'tour': tour,
                'objective': qubo_problem.evaluate(qubo_solution),
                'is_valid': self._validate_tsp_tour(tour),
            }
        elif problem_type == 'Max-Cut':
            partition = (qubo_solution > 0.5).astype(int)
            return {
                'partition': partition,
                'objective': -qubo_problem.evaluate(qubo_solution),  # Convert back to maximization
                'is_valid': True,
            }
        else:
            return {
                'solution': qubo_solution,
                'objective': qubo_problem.evaluate(qubo_solution),
                'is_valid': True,
            }

    def _validate_qap_assignment(self, assignment: np.ndarray) -> bool:
        """Validate QAP assignment (each facility to unique location)."""
        return len(np.unique(assignment)) == len(assignment)

    def _validate_tsp_tour(self, tour: np.ndarray) -> bool:
        """Validate TSP tour (visits each city exactly once)."""
        return len(np.unique(tour)) == len(tour)


__all__ = ['QUBOConverter', 'QUBOProblem']