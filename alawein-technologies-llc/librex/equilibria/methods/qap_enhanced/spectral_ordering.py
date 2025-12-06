"""
Spectral Ordering for Initial QAP Solutions

This module implements spectral methods for generating high-quality initial
permutations for the QAP using eigenvector analysis of graph Laplacians.

Key Concepts:
1. Fiedler Vector: The eigenvector corresponding to the second smallest
   eigenvalue (Fiedler value) of the Laplacian provides a natural ordering
   that tends to keep strongly connected nodes close together.

2. FFT Acceleration: For grid graphs, we can compute eigenvectors in
   O(n log n) time using FFT instead of O(n³) with standard methods.

3. Multi-level Ordering: Use multiple eigenvectors to create a
   multi-dimensional embedding for more sophisticated orderings.

Mathematical Foundation:
For a grid graph, the eigenvectors have a known analytical form:
v_{i,j}(x,y) = sin(πix/m) * sin(πjy/n) for an m×n grid

This allows direct computation without expensive eigendecomposition.
"""

from typing import List, Optional, Tuple

import numpy as np
from scipy.sparse.linalg import eigsh

from .laplace_detector import GraphType, LaplacianDetector


class SpectralOrdering:
    """
    Generate initial QAP permutations using spectral graph theory.

    This class provides various spectral ordering methods optimized
    for different graph structures, with FFT acceleration where applicable.
    """

    def __init__(self, use_fft: bool = True):
        """
        Initialize spectral ordering engine.

        Args:
            use_fft: Whether to use FFT acceleration for special structures
        """
        self.use_fft = use_fft
        self.detector = LaplacianDetector()

    def generate_ordering(self,
                         flow_matrix: np.ndarray,
                         distance_matrix: np.ndarray,
                         method: str = "auto") -> np.ndarray:
        """
        Generate initial permutation using spectral methods.

        Args:
            flow_matrix: QAP flow matrix (F)
            distance_matrix: QAP distance matrix (D)
            method: Ordering method ("auto", "fiedler", "multi_level", "hybrid")

        Returns:
            Initial permutation vector
        """
        n = len(flow_matrix)

        # Detect structure in both matrices
        flow_type, flow_meta = self.detector.detect_structure(flow_matrix)
        dist_type, dist_meta = self.detector.detect_structure(distance_matrix)

        # Select best method based on structure
        if method == "auto":
            method = self._select_method(flow_type, dist_type)

        if method == "fiedler":
            return self._fiedler_ordering(flow_matrix, distance_matrix,
                                         flow_type, dist_type)
        elif method == "multi_level":
            return self._multi_level_ordering(flow_matrix, distance_matrix)
        elif method == "hybrid":
            return self._hybrid_ordering(flow_matrix, distance_matrix,
                                        flow_type, dist_type)
        else:
            # Fallback to random
            return np.random.permutation(n)

    def _select_method(self, flow_type: GraphType, dist_type: GraphType) -> str:
        """Select best ordering method based on matrix structures."""
        # If either matrix is a grid and FFT is enabled, use Fiedler
        if self.use_fft and (flow_type == GraphType.GRID_2D or
                             dist_type == GraphType.GRID_2D):
            return "fiedler"

        # For general Laplacians, use multi-level
        if (flow_type in [GraphType.GENERAL, GraphType.REGULAR] or
            dist_type in [GraphType.GENERAL, GraphType.REGULAR]):
            return "multi_level"

        # Default to hybrid approach
        return "hybrid"

    def _fiedler_ordering(self,
                         flow_matrix: np.ndarray,
                         distance_matrix: np.ndarray,
                         flow_type: GraphType,
                         dist_type: GraphType) -> np.ndarray:
        """
        Generate ordering using Fiedler vector(s).

        For grid graphs, use FFT-based computation for efficiency.
        """
        n = len(flow_matrix)

        # Compute Fiedler vectors
        flow_fiedler = self._compute_fiedler_vector(flow_matrix, flow_type)
        dist_fiedler = self._compute_fiedler_vector(distance_matrix, dist_type)

        # Combine Fiedler vectors for joint ordering
        if flow_fiedler is not None and dist_fiedler is not None:
            # Use 2D embedding from both Fiedler vectors
            embedding = np.column_stack([flow_fiedler, dist_fiedler])

            # Sort by first principal component
            u, s, vt = np.linalg.svd(embedding, full_matrices=False)
            scores = u[:, 0]
        elif flow_fiedler is not None:
            scores = flow_fiedler
        elif dist_fiedler is not None:
            scores = dist_fiedler
        else:
            # Fallback to random
            return np.random.permutation(n)

        # Generate permutation by sorting scores
        return np.argsort(scores)

    def _compute_fiedler_vector(self,
                                matrix: np.ndarray,
                                graph_type: GraphType) -> Optional[np.ndarray]:
        """
        Compute Fiedler vector (2nd smallest eigenvalue eigenvector).

        Uses FFT for grid graphs, standard methods otherwise.
        """
        if not self.detector.is_laplacian(matrix):
            return None

        n = len(matrix)

        # Use FFT for 2D grid graphs
        if self.use_fft and graph_type == GraphType.GRID_2D:
            return self._fiedler_vector_grid_fft(n)

        # Use FFT for cycle graphs
        if self.use_fft and graph_type == GraphType.CYCLE:
            return self._fiedler_vector_cycle_fft(n)

        # Standard eigendecomposition for general case
        try:
            # Compute 2 smallest eigenvalues/vectors
            eigenvalues, eigenvectors = eigsh(matrix, k=2, which='SM')

            # Return Fiedler vector (2nd eigenvector)
            return eigenvectors[:, 1]
        except:
            return None

    def _fiedler_vector_grid_fft(self, n: int) -> np.ndarray:
        """
        Compute Fiedler vector for n×n grid using FFT.

        For a grid graph, the Fiedler vector corresponds to the
        eigenfunction sin(πx/n) or sin(πy/n), which can be
        computed directly without eigendecomposition.
        """
        # Assume square grid
        grid_size = int(np.sqrt(n))
        if grid_size * grid_size != n:
            # Not a perfect square, try to find factors
            for m in range(2, int(np.sqrt(n)) + 1):
                if n % m == 0:
                    grid_size = m
                    break

        m = grid_size
        k = n // m

        # Generate 2D coordinates
        x = np.arange(m)
        y = np.arange(k)
        xx, yy = np.meshgrid(x, y, indexing='ij')

        # Fiedler vector for grid: sin(πi/m) for smallest non-zero eigenvalue
        # We use the (0,1) mode which gives vertical strips
        fiedler_2d = np.sin(np.pi * xx / m)

        return fiedler_2d.flatten()

    def _fiedler_vector_cycle_fft(self, n: int) -> np.ndarray:
        """
        Compute Fiedler vector for cycle graph using FFT.

        For a cycle, the eigenvectors are Fourier modes.
        The Fiedler vector corresponds to cos(2πk/n) or sin(2πk/n)
        with k=1.
        """
        k = 1  # First non-constant mode
        indices = np.arange(n)

        # Use cosine mode for Fiedler vector
        fiedler = np.cos(2 * np.pi * k * indices / n)

        return fiedler

    def _multi_level_ordering(self,
                              flow_matrix: np.ndarray,
                              distance_matrix: np.ndarray) -> np.ndarray:
        """
        Generate ordering using multiple eigenvectors.

        This creates a multi-dimensional spectral embedding and
        applies clustering or sorting in the embedded space.
        """
        n = len(flow_matrix)

        # Compute combined Laplacian
        L_combined = self._compute_combined_laplacian(flow_matrix, distance_matrix)

        if L_combined is None:
            return np.random.permutation(n)

        try:
            # Compute first k eigenvectors (k = log(n))
            k = min(int(np.log2(n)) + 1, n - 1, 10)
            eigenvalues, eigenvectors = eigsh(L_combined, k=k, which='SM')

            # Create multi-dimensional embedding (exclude constant eigenvector)
            embedding = eigenvectors[:, 1:]

            # Apply recursive bisection based on eigenvectors
            ordering = self._recursive_bisection(embedding)

            return ordering
        except:
            return np.random.permutation(n)

    def _recursive_bisection(self, embedding: np.ndarray) -> np.ndarray:
        """
        Apply recursive bisection in spectral embedding space.

        This creates a hierarchical partitioning that preserves locality.
        """
        n = len(embedding)
        ordering = np.zeros(n, dtype=int)
        self._bisect_recursive(embedding, np.arange(n), ordering, 0)

        # Convert positions to permutation
        return np.argsort(ordering)

    def _bisect_recursive(self,
                         embedding: np.ndarray,
                         indices: np.ndarray,
                         ordering: np.ndarray,
                         offset: int) -> int:
        """Recursive helper for bisection."""
        if len(indices) <= 1:
            if len(indices) == 1:
                ordering[indices[0]] = offset
            return offset + len(indices)

        # Use first principal component for split
        subset_embedding = embedding[indices]

        if subset_embedding.shape[1] > 0:
            # Compute principal direction
            mean = np.mean(subset_embedding, axis=0)
            centered = subset_embedding - mean
            _, _, vt = np.linalg.svd(centered, full_matrices=False)
            scores = centered @ vt[0]
        else:
            # Random split if no embedding
            scores = np.random.randn(len(indices))

        # Split based on median
        median = np.median(scores)
        left_mask = scores <= median
        right_mask = ~left_mask

        left_indices = indices[left_mask]
        right_indices = indices[right_mask]

        # Recurse on left
        offset = self._bisect_recursive(embedding, left_indices, ordering, offset)

        # Recurse on right
        offset = self._bisect_recursive(embedding, right_indices, ordering, offset)

        return offset

    def _hybrid_ordering(self,
                        flow_matrix: np.ndarray,
                        distance_matrix: np.ndarray,
                        flow_type: GraphType,
                        dist_type: GraphType) -> np.ndarray:
        """
        Hybrid ordering combining multiple strategies.

        This method combines Fiedler ordering with local refinement.
        """
        n = len(flow_matrix)

        # Start with Fiedler ordering
        base_ordering = self._fiedler_ordering(flow_matrix, distance_matrix,
                                               flow_type, dist_type)

        # Apply 2-opt local refinement
        improved_ordering = self._local_refinement_2opt(
            base_ordering, flow_matrix, distance_matrix, max_iter=100
        )

        return improved_ordering

    def _local_refinement_2opt(self,
                               permutation: np.ndarray,
                               flow_matrix: np.ndarray,
                               distance_matrix: np.ndarray,
                               max_iter: int = 100) -> np.ndarray:
        """
        Apply 2-opt local search to refine ordering.

        This performs pairwise swaps to improve the QAP objective.
        """
        n = len(permutation)
        current_perm = permutation.copy()
        current_obj = self._compute_qap_objective(current_perm,
                                                  flow_matrix,
                                                  distance_matrix)

        for _ in range(max_iter):
            improved = False

            # Try all pairwise swaps
            for i in range(n - 1):
                for j in range(i + 1, n):
                    # Swap positions i and j
                    new_perm = current_perm.copy()
                    new_perm[i], new_perm[j] = new_perm[j], new_perm[i]

                    # Compute new objective
                    new_obj = self._compute_qap_objective(new_perm,
                                                          flow_matrix,
                                                          distance_matrix)

                    # Accept improvement
                    if new_obj < current_obj:
                        current_perm = new_perm
                        current_obj = new_obj
                        improved = True
                        break
                if improved:
                    break

            if not improved:
                break

        return current_perm

    def _compute_qap_objective(self,
                               permutation: np.ndarray,
                               flow_matrix: np.ndarray,
                               distance_matrix: np.ndarray) -> float:
        """Compute QAP objective value for a permutation."""
        n = len(permutation)
        P = np.zeros((n, n))
        P[np.arange(n), permutation] = 1
        return np.trace(flow_matrix @ P @ distance_matrix @ P.T)

    def _compute_combined_laplacian(self,
                                    flow_matrix: np.ndarray,
                                    distance_matrix: np.ndarray) -> Optional[np.ndarray]:
        """
        Compute a combined Laplacian from flow and distance matrices.

        This creates a weighted combination that captures both structures.
        """
        # Check if matrices are Laplacians
        flow_is_lap = self.detector.is_laplacian(flow_matrix)
        dist_is_lap = self.detector.is_laplacian(distance_matrix)

        if flow_is_lap and dist_is_lap:
            # Weighted combination
            flow_norm = np.linalg.norm(flow_matrix, 'fro')
            dist_norm = np.linalg.norm(distance_matrix, 'fro')

            if flow_norm > 0 and dist_norm > 0:
                w1 = dist_norm / (flow_norm + dist_norm)
                w2 = flow_norm / (flow_norm + dist_norm)
                return w1 * flow_matrix + w2 * distance_matrix
            elif flow_norm > 0:
                return flow_matrix
            elif dist_norm > 0:
                return distance_matrix

        elif flow_is_lap:
            return flow_matrix
        elif dist_is_lap:
            return distance_matrix

        # If neither is Laplacian, try to construct one
        # from the adjacency structure
        A_flow = np.abs(flow_matrix)
        np.fill_diagonal(A_flow, 0)
        D_flow = np.diag(np.sum(A_flow, axis=1))
        L_flow = D_flow - A_flow

        A_dist = np.abs(distance_matrix)
        np.fill_diagonal(A_dist, 0)
        D_dist = np.diag(np.sum(A_dist, axis=1))
        L_dist = D_dist - A_dist

        # Weighted combination
        return 0.5 * L_flow + 0.5 * L_dist