"""
Laplacian Matrix Detection and Characterization

This module provides algorithms to detect if a matrix is a graph Laplacian
and characterize its structure for efficient FFT-based operations.

Mathematical Properties of Laplacian Matrices:
1. Symmetric (L = L^T)
2. Zero row sums (L * 1 = 0)
3. Non-positive off-diagonal entries
4. Positive semi-definite (all eigenvalues ≥ 0)
5. Smallest eigenvalue is 0 (with eigenvector 1)

Special Structures Detectable:
- Grid graphs: 2D lattice structure
- Cycle graphs: Circulant matrices
- Path graphs: Tridiagonal matrices
- Tree graphs: Sparse acyclic structure
- Complete graphs: L = nI - J (J = all-ones matrix)
"""

from enum import Enum
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.sparse import issparse
from scipy.sparse.csgraph import connected_components


class GraphType(Enum):
    """Enumeration of detectable graph types"""
    UNKNOWN = "unknown"
    GRID_2D = "grid_2d"
    GRID_3D = "grid_3d"
    CYCLE = "cycle"
    PATH = "path"
    TREE = "tree"
    COMPLETE = "complete"
    REGULAR = "regular"  # k-regular graph
    GENERAL = "general"   # General graph Laplacian


class LaplacianDetector:
    """
    Detector for Laplacian matrix structure and properties.

    This class analyzes matrices to determine if they are graph Laplacians
    and identifies special structures that enable FFT acceleration.
    """

    def __init__(self, tolerance: float = 1e-10):
        """
        Initialize Laplacian detector.

        Args:
            tolerance: Numerical tolerance for property checks
        """
        self.tolerance = tolerance

    def is_laplacian(self, matrix: np.ndarray) -> bool:
        """
        Check if a matrix is a graph Laplacian.

        Args:
            matrix: Matrix to check

        Returns:
            True if matrix satisfies Laplacian properties
        """
        n = len(matrix)

        # Check 1: Square matrix
        if matrix.shape != (n, n):
            return False

        # Check 2: Symmetry
        if not np.allclose(matrix, matrix.T, atol=self.tolerance):
            return False

        # Check 3: Zero row sums
        row_sums = np.sum(matrix, axis=1)
        if not np.allclose(row_sums, 0, atol=self.tolerance):
            return False

        # Check 4: Non-positive off-diagonal entries
        off_diagonal = matrix - np.diag(np.diag(matrix))
        if np.any(off_diagonal > self.tolerance):
            return False

        # Check 5: Diagonal dominance (sum of abs off-diagonal ≤ diagonal)
        for i in range(n):
            off_diag_sum = np.sum(np.abs(matrix[i, :])) - np.abs(matrix[i, i])
            if matrix[i, i] < off_diag_sum - self.tolerance:
                return False

        return True

    def detect_structure(self, matrix: np.ndarray) -> Tuple[GraphType, Dict]:
        """
        Detect the graph structure of a Laplacian matrix.

        Args:
            matrix: Laplacian matrix to analyze

        Returns:
            Tuple of (GraphType, metadata dict)
        """
        if not self.is_laplacian(matrix):
            return GraphType.UNKNOWN, {"is_laplacian": False}

        n = len(matrix)
        metadata = {
            "is_laplacian": True,
            "size": n,
            "density": np.count_nonzero(matrix) / (n * n)
        }

        # Check for complete graph: L = nI - J
        if self._is_complete_graph(matrix):
            metadata["degree"] = n - 1
            return GraphType.COMPLETE, metadata

        # Check for grid graph
        grid_dim = self._detect_grid_structure(matrix)
        if grid_dim is not None:
            metadata["grid_dimension"] = grid_dim
            if len(grid_dim) == 2:
                metadata["grid_size"] = grid_dim
                return GraphType.GRID_2D, metadata
            elif len(grid_dim) == 3:
                metadata["grid_size"] = grid_dim
                return GraphType.GRID_3D, metadata

        # Check for cycle graph
        if self._is_cycle_graph(matrix):
            metadata["cycle_length"] = n
            return GraphType.CYCLE, metadata

        # Check for path graph
        if self._is_path_graph(matrix):
            metadata["path_length"] = n
            return GraphType.PATH, metadata

        # Check for tree
        if self._is_tree(matrix):
            return GraphType.TREE, metadata

        # Check for regular graph
        degree = self._get_regular_degree(matrix)
        if degree is not None:
            metadata["degree"] = degree
            return GraphType.REGULAR, metadata

        return GraphType.GENERAL, metadata

    def _is_complete_graph(self, L: np.ndarray) -> bool:
        """Check if L represents a complete graph."""
        n = len(L)
        # Complete graph: L = nI - J (J = all-ones)
        expected = n * np.eye(n) - np.ones((n, n))
        return np.allclose(L, expected, atol=self.tolerance)

    def _detect_grid_structure(self, L: np.ndarray) -> Optional[Tuple[int, ...]]:
        """
        Detect if L represents a grid graph and return dimensions.

        For 2D grid of size m×n: L = I_m ⊗ T_n + T_m ⊗ I_n
        where T is the 1D path Laplacian.
        """
        n = len(L)

        # Check for 2D grid
        for m in range(2, int(np.sqrt(n)) + 1):
            if n % m == 0:
                k = n // m
                if self._is_grid_laplacian(L, m, k):
                    return (m, k)

        # Check for 3D grid (less common)
        for m in range(2, int(n**(1/3)) + 2):
            if n % (m * m) == 0:
                k = n // (m * m)
                if self._is_3d_grid_laplacian(L, m, m, k):
                    return (m, m, k)

        return None

    def _is_grid_laplacian(self, L: np.ndarray, m: int, n: int) -> bool:
        """Check if L is the Laplacian of an m×n grid."""
        # Build expected grid Laplacian
        I_m = np.eye(m)
        I_n = np.eye(n)
        T_m = self._path_laplacian(m)
        T_n = self._path_laplacian(n)

        # L = I_m ⊗ T_n + T_m ⊗ I_n
        expected = np.kron(I_m, T_n) + np.kron(T_m, I_n)

        return np.allclose(L, expected, atol=self.tolerance)

    def _is_3d_grid_laplacian(self, L: np.ndarray, m: int, n: int, p: int) -> bool:
        """Check if L is the Laplacian of an m×n×p 3D grid."""
        # Similar to 2D but with three dimensions
        I_m = np.eye(m)
        I_n = np.eye(n)
        I_p = np.eye(p)
        T_m = self._path_laplacian(m)
        T_n = self._path_laplacian(n)
        T_p = self._path_laplacian(p)

        # L = I_m ⊗ I_n ⊗ T_p + I_m ⊗ T_n ⊗ I_p + T_m ⊗ I_n ⊗ I_p
        expected = (np.kron(np.kron(I_m, I_n), T_p) +
                   np.kron(np.kron(I_m, T_n), I_p) +
                   np.kron(np.kron(T_m, I_n), I_p))

        return np.allclose(L, expected, atol=self.tolerance)

    def _path_laplacian(self, n: int) -> np.ndarray:
        """Generate the Laplacian matrix for a path graph."""
        L = 2 * np.eye(n)
        L[0, 0] = 1
        L[-1, -1] = 1

        for i in range(n - 1):
            L[i, i + 1] = -1
            L[i + 1, i] = -1

        return L

    def _is_cycle_graph(self, L: np.ndarray) -> bool:
        """Check if L represents a cycle graph (circulant matrix)."""
        n = len(L)

        # Cycle graph has degree 2 everywhere
        degrees = np.diag(L)
        if not np.allclose(degrees, 2, atol=self.tolerance):
            return False

        # Check circulant structure
        first_row = L[0, :]
        for i in range(1, n):
            expected_row = np.roll(first_row, i)
            if not np.allclose(L[i, :], expected_row, atol=self.tolerance):
                return False

        return True

    def _is_path_graph(self, L: np.ndarray) -> bool:
        """Check if L represents a path graph (tridiagonal)."""
        n = len(L)

        # Check if matrix is tridiagonal
        for i in range(n):
            for j in range(n):
                if abs(i - j) > 1 and abs(L[i, j]) > self.tolerance:
                    return False

        # Check degree sequence: endpoints have degree 1, others degree 2
        degrees = np.diag(L)
        expected_degrees = 2 * np.ones(n)
        expected_degrees[0] = 1
        expected_degrees[-1] = 1

        return np.allclose(degrees, expected_degrees, atol=self.tolerance)

    def _is_tree(self, L: np.ndarray) -> bool:
        """Check if L represents a tree (connected acyclic graph)."""
        n = len(L)

        # Build adjacency matrix from Laplacian
        A = np.diag(np.diag(L)) - L

        # Count edges
        num_edges = np.count_nonzero(A) // 2  # Undirected graph

        # Tree has exactly n-1 edges
        if num_edges != n - 1:
            return False

        # Check connectivity
        if issparse(A):
            n_components, _ = connected_components(A)
        else:
            # Simple DFS for connectivity check
            visited = np.zeros(n, dtype=bool)
            stack = [0]
            while stack:
                v = stack.pop()
                if visited[v]:
                    continue
                visited[v] = True
                for u in range(n):
                    if A[v, u] > 0 and not visited[u]:
                        stack.append(u)
            n_components = 1 if np.all(visited) else 2

        return n_components == 1

    def _get_regular_degree(self, L: np.ndarray) -> Optional[int]:
        """Check if graph is k-regular and return degree k."""
        degrees = np.diag(L)

        # Check if all degrees are the same
        if np.allclose(degrees, degrees[0], atol=self.tolerance):
            return int(degrees[0])

        return None

    def get_spectral_properties(self, L: np.ndarray) -> Dict:
        """
        Compute spectral properties useful for preconditioning.

        Args:
            L: Laplacian matrix

        Returns:
            Dictionary with spectral information
        """
        # Compute eigenvalues
        eigenvalues = np.linalg.eigvalsh(L)

        # Sort eigenvalues
        eigenvalues = np.sort(eigenvalues)

        # Fiedler value (second smallest eigenvalue)
        fiedler_value = eigenvalues[1] if len(eigenvalues) > 1 else 0

        # Spectral gap
        spectral_gap = fiedler_value - eigenvalues[0]

        # Condition number (ratio of largest to smallest non-zero eigenvalue)
        non_zero_eigs = eigenvalues[eigenvalues > self.tolerance]
        if len(non_zero_eigs) > 0:
            condition = non_zero_eigs[-1] / non_zero_eigs[0]
        else:
            condition = np.inf

        return {
            "smallest_eigenvalue": eigenvalues[0],
            "fiedler_value": fiedler_value,
            "largest_eigenvalue": eigenvalues[-1],
            "spectral_gap": spectral_gap,
            "condition_number": condition,
            "eigenvalue_spread": eigenvalues[-1] - eigenvalues[0]
        }