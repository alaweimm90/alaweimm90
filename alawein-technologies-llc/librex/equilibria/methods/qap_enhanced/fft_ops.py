"""
FFT-Accelerated Operations for Grid Graph Laplacians

This module provides Fast Fourier Transform (FFT) based operations
for efficiently working with grid graph Laplacians in QAP optimization.

Mathematical Foundation:
For a 2D grid graph of size m×n, the Laplacian has the form:
L = I_m ⊗ T_n + T_m ⊗ I_n

where T_k is the 1D path/cycle Laplacian and ⊗ is the Kronecker product.

Key Properties:
1. Eigenvalues: λ_{i,j} = λ_i^{(m)} + λ_j^{(n)}
   where λ_k^{(p)} = 2(1 - cos(πk/p)) for path graphs

2. Eigenvectors: v_{i,j} = v_i^{(m)} ⊗ v_j^{(n)}
   where v_k^{(p)} are sine/cosine basis functions

3. Matrix-vector products: Can be computed in O(n log n) using 2D FFT

Complexity Comparison:
- Standard matrix-vector product: O(n²) for n×n grid
- FFT-based product: O(n log n)
- Standard eigendecomposition: O(n³)
- FFT-based eigendecomposition: O(n log n)
"""

from typing import Optional, Tuple

import numpy as np
from scipy.fftpack import dct, idct, dst, idst


class FFTOperations:
    """
    FFT-based operations for grid graph Laplacians.

    This class provides efficient implementations of common linear algebra
    operations for grid-structured matrices using FFT/DCT/DST transforms.
    """

    def __init__(self):
        """Initialize FFT operations handler."""
        self.cached_eigenvalues = {}
        self.cached_transforms = {}

    def fft_laplace_mult(self,
                        laplacian_type: str,
                        vector: np.ndarray,
                        grid_shape: Tuple[int, int]) -> np.ndarray:
        """
        Fast Laplacian matrix-vector multiplication using FFT.

        Args:
            laplacian_type: Type of grid ("path", "cycle", "periodic")
            vector: Input vector (flattened)
            grid_shape: Shape of the 2D grid (m, n)

        Returns:
            Result of L @ v computed via FFT

        Mathematical Detail:
        For grid Laplacian L with eigendecomposition L = Q Λ Q^T,
        we compute L @ v = Q Λ Q^T @ v using:
        1. w = Q^T @ v (forward transform)
        2. w = Λ @ w (pointwise multiplication)
        3. result = Q @ w (inverse transform)
        """
        m, n = grid_shape
        assert len(vector) == m * n, "Vector size must match grid dimensions"

        # Reshape vector to 2D grid
        V = vector.reshape((m, n))

        if laplacian_type == "path":
            # Use DST for path graph (Dirichlet boundary)
            result = self._path_laplacian_mult_dst(V, m, n)
        elif laplacian_type == "cycle" or laplacian_type == "periodic":
            # Use FFT for periodic boundary
            result = self._cycle_laplacian_mult_fft(V, m, n)
        else:
            # Default to path graph
            result = self._path_laplacian_mult_dst(V, m, n)

        return result.flatten()

    def _path_laplacian_mult_dst(self, V: np.ndarray, m: int, n: int) -> np.ndarray:
        """
        Laplacian multiplication for path graph using DST.

        The path graph Laplacian has eigenvectors that are sine functions,
        which correspond to the Discrete Sine Transform (DST).
        """
        # Forward DST (2D)
        V_transformed = dst(dst(V, type=1, axis=0), type=1, axis=1)

        # Get eigenvalues for path Laplacian
        eigenvalues = self._get_path_laplacian_eigenvalues(m, n)

        # Pointwise multiplication with eigenvalues
        V_transformed *= eigenvalues

        # Inverse DST (2D)
        # Normalize by 1/(4mn) for DST type 1
        result = idst(idst(V_transformed, type=1, axis=0), type=1, axis=1)
        result /= (4 * m * n)

        return result

    def _cycle_laplacian_mult_fft(self, V: np.ndarray, m: int, n: int) -> np.ndarray:
        """
        Laplacian multiplication for cycle/periodic graph using FFT.

        The cycle graph Laplacian has eigenvectors that are complex exponentials,
        which correspond to the standard FFT.
        """
        # Forward 2D FFT
        V_transformed = np.fft.fft2(V)

        # Get eigenvalues for cycle Laplacian
        eigenvalues = self._get_cycle_laplacian_eigenvalues(m, n)

        # Pointwise multiplication with eigenvalues
        V_transformed *= eigenvalues

        # Inverse 2D FFT
        result = np.fft.ifft2(V_transformed).real

        return result

    def _get_path_laplacian_eigenvalues(self, m: int, n: int) -> np.ndarray:
        """
        Compute eigenvalues for 2D path graph Laplacian.

        For an m×n grid with path structure:
        λ_{i,j} = 2(1 - cos(πi/m)) + 2(1 - cos(πj/n))
        """
        cache_key = ("path", m, n)
        if cache_key in self.cached_eigenvalues:
            return self.cached_eigenvalues[cache_key]

        # Create 1D eigenvalues
        lambda_m = 2 * (1 - np.cos(np.pi * np.arange(1, m + 1) / (m + 1)))
        lambda_n = 2 * (1 - np.cos(np.pi * np.arange(1, n + 1) / (n + 1)))

        # Create 2D eigenvalue grid (sum of 1D eigenvalues)
        eigenvalues = lambda_m[:, np.newaxis] + lambda_n[np.newaxis, :]

        self.cached_eigenvalues[cache_key] = eigenvalues
        return eigenvalues

    def _get_cycle_laplacian_eigenvalues(self, m: int, n: int) -> np.ndarray:
        """
        Compute eigenvalues for 2D cycle/periodic graph Laplacian.

        For an m×n grid with periodic boundary:
        λ_{k,l} = 2(1 - cos(2πk/m)) + 2(1 - cos(2πl/n))
        """
        cache_key = ("cycle", m, n)
        if cache_key in self.cached_eigenvalues:
            return self.cached_eigenvalues[cache_key]

        # Create frequency grids
        k = np.fft.fftfreq(m, 1/m)
        l = np.fft.fftfreq(n, 1/n)

        # Create 2D eigenvalue grid
        kk, ll = np.meshgrid(k, l, indexing='ij')
        eigenvalues = 2 * (2 - np.cos(2 * np.pi * kk / m) - np.cos(2 * np.pi * ll / n))

        self.cached_eigenvalues[cache_key] = eigenvalues
        return eigenvalues

    def fiedler_vector_fft(self, grid_shape: Tuple[int, int],
                           boundary: str = "path") -> np.ndarray:
        """
        Compute Fiedler vector for grid graph using FFT.

        The Fiedler vector is the eigenvector corresponding to the
        second smallest eigenvalue of the Laplacian.

        Args:
            grid_shape: Shape of the 2D grid (m, n)
            boundary: Boundary condition ("path" or "cycle")

        Returns:
            Fiedler vector (flattened)
        """
        m, n = grid_shape

        if boundary == "path":
            # For path graph, Fiedler vector is sin(π/m) ⊗ 1
            # or 1 ⊗ sin(π/n) depending on which gives smaller eigenvalue

            # Compute the two candidate eigenvalues
            lambda_10 = 2 * (1 - np.cos(np.pi / (m + 1)))
            lambda_01 = 2 * (1 - np.cos(np.pi / (n + 1)))

            if lambda_10 <= lambda_01:
                # Use (1,0) mode
                i_vals = np.arange(m)
                j_vals = np.arange(n)
                ii, jj = np.meshgrid(i_vals, j_vals, indexing='ij')
                fiedler_2d = np.sin(np.pi * (ii + 1) / (m + 1))
            else:
                # Use (0,1) mode
                i_vals = np.arange(m)
                j_vals = np.arange(n)
                ii, jj = np.meshgrid(i_vals, j_vals, indexing='ij')
                fiedler_2d = np.sin(np.pi * (jj + 1) / (n + 1))

        elif boundary == "cycle":
            # For cycle graph, Fiedler vector corresponds to cos(2π/m) or cos(2π/n)
            lambda_10 = 2 * (1 - np.cos(2 * np.pi / m))
            lambda_01 = 2 * (1 - np.cos(2 * np.pi / n))

            if lambda_10 <= lambda_01:
                # Use (1,0) mode
                i_vals = np.arange(m)
                j_vals = np.arange(n)
                ii, jj = np.meshgrid(i_vals, j_vals, indexing='ij')
                fiedler_2d = np.cos(2 * np.pi * ii / m)
            else:
                # Use (0,1) mode
                i_vals = np.arange(m)
                j_vals = np.arange(n)
                ii, jj = np.meshgrid(i_vals, j_vals, indexing='ij')
                fiedler_2d = np.cos(2 * np.pi * jj / n)

        else:
            # Default to path
            return self.fiedler_vector_fft(grid_shape, "path")

        # Normalize
        fiedler_2d /= np.linalg.norm(fiedler_2d)

        return fiedler_2d.flatten()

    def fast_qap_objective_grid(self,
                                permutation: np.ndarray,
                                flow_laplacian: np.ndarray,
                                distance_laplacian: np.ndarray,
                                grid_shape: Tuple[int, int]) -> float:
        """
        Fast QAP objective computation for grid-structured problems.

        When both F and D are grid Laplacians, we can accelerate
        the computation of trace(F @ P @ D @ P^T) using FFT.

        Args:
            permutation: Permutation vector
            flow_laplacian: Flow matrix (assumed grid Laplacian)
            distance_laplacian: Distance matrix (assumed grid Laplacian)
            grid_shape: Shape of the grid

        Returns:
            QAP objective value

        Note: This assumes both matrices are grid Laplacians.
        For mixed cases, standard computation is used.
        """
        n = len(permutation)
        m, k = grid_shape
        assert m * k == n, "Grid shape must match problem size"

        # Convert permutation to matrix
        P = np.zeros((n, n))
        P[np.arange(n), permutation] = 1

        # For grid Laplacians, we can use the fact that they have
        # tensor product structure to accelerate computation

        # Standard computation for now (can be optimized further)
        return np.trace(flow_laplacian @ P @ distance_laplacian @ P.T)

    def solve_laplacian_system_fft(self,
                                   laplacian_type: str,
                                   b: np.ndarray,
                                   grid_shape: Tuple[int, int],
                                   regularization: float = 1e-10) -> np.ndarray:
        """
        Solve linear system L @ x = b for grid Laplacian L using FFT.

        This is useful for preconditioning and Newton-type methods.

        Args:
            laplacian_type: Type of grid ("path" or "cycle")
            b: Right-hand side vector
            grid_shape: Shape of the grid
            regularization: Regularization parameter (since L is singular)

        Returns:
            Solution vector x

        Mathematical Note:
        Since Laplacians are singular (smallest eigenvalue = 0),
        we solve the regularized system (L + εI) @ x = b.
        """
        m, n = grid_shape
        B = b.reshape((m, n))

        if laplacian_type == "path":
            # Forward DST
            B_transformed = dst(dst(B, type=1, axis=0), type=1, axis=1)

            # Get eigenvalues
            eigenvalues = self._get_path_laplacian_eigenvalues(m, n)

            # Divide by eigenvalues (with regularization)
            B_transformed /= (eigenvalues + regularization)

            # Inverse DST
            X = idst(idst(B_transformed, type=1, axis=0), type=1, axis=1)
            X /= (4 * m * n)

        elif laplacian_type == "cycle":
            # Forward FFT
            B_transformed = np.fft.fft2(B)

            # Get eigenvalues
            eigenvalues = self._get_cycle_laplacian_eigenvalues(m, n)

            # Divide by eigenvalues (with regularization)
            # Handle zero eigenvalue specially
            mask = np.abs(eigenvalues) > 1e-10
            B_transformed[mask] /= eigenvalues[mask]
            B_transformed[~mask] = 0

            # Inverse FFT
            X = np.fft.ifft2(B_transformed).real

        else:
            # Default to path
            return self.solve_laplacian_system_fft("path", b, grid_shape, regularization)

        return X.flatten()

    def compute_eigendecomposition_fft(self,
                                       grid_shape: Tuple[int, int],
                                       boundary: str = "path",
                                       k: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute eigendecomposition of grid Laplacian using FFT.

        Args:
            grid_shape: Shape of the grid (m, n)
            boundary: Boundary condition ("path" or "cycle")
            k: Number of smallest eigenvalues/vectors to return (None = all)

        Returns:
            Tuple of (eigenvalues, eigenvectors)

        This is much faster than standard eigendecomposition:
        O(n log n) vs O(n³) complexity.
        """
        m, n = grid_shape
        total_size = m * n

        if boundary == "path":
            # Eigenvalues for path Laplacian
            lambda_m = 2 * (1 - np.cos(np.pi * np.arange(1, m + 1) / (m + 1)))
            lambda_n = 2 * (1 - np.cos(np.pi * np.arange(1, n + 1) / (n + 1)))

            # All eigenvalues (tensor product)
            eigenvalues = []
            eigenvector_indices = []

            for i in range(m):
                for j in range(n):
                    eigenvalues.append(lambda_m[i] + lambda_n[j])
                    eigenvector_indices.append((i, j))

            eigenvalues = np.array(eigenvalues)

            # Sort by eigenvalue
            sorted_idx = np.argsort(eigenvalues)
            eigenvalues = eigenvalues[sorted_idx]
            eigenvector_indices = [eigenvector_indices[i] for i in sorted_idx]

            # Select k smallest if specified
            if k is not None:
                eigenvalues = eigenvalues[:k]
                eigenvector_indices = eigenvector_indices[:k]

            # Generate eigenvectors using sine basis
            num_vecs = len(eigenvalues)
            eigenvectors = np.zeros((total_size, num_vecs))

            for idx, (i, j) in enumerate(eigenvector_indices):
                # Eigenvector is sin((i+1)πx/(m+1)) ⊗ sin((j+1)πy/(n+1))
                vec_2d = np.outer(
                    np.sin(np.pi * (i + 1) * np.arange(1, m + 1) / (m + 1)),
                    np.sin(np.pi * (j + 1) * np.arange(1, n + 1) / (n + 1))
                )
                vec_2d /= np.linalg.norm(vec_2d)
                eigenvectors[:, idx] = vec_2d.flatten()

        elif boundary == "cycle":
            # Similar for cycle, but using complex exponentials/FFT basis
            eigenvalues = []
            eigenvector_indices = []

            for i in range(m):
                for j in range(n):
                    eig_val = 2 * (2 - np.cos(2 * np.pi * i / m) - np.cos(2 * np.pi * j / n))
                    eigenvalues.append(eig_val)
                    eigenvector_indices.append((i, j))

            eigenvalues = np.array(eigenvalues)

            # Sort by eigenvalue
            sorted_idx = np.argsort(eigenvalues)
            eigenvalues = eigenvalues[sorted_idx]
            eigenvector_indices = [eigenvector_indices[i] for i in sorted_idx]

            # Select k smallest if specified
            if k is not None:
                eigenvalues = eigenvalues[:k]
                eigenvector_indices = eigenvector_indices[:k]

            # Generate eigenvectors using Fourier basis
            num_vecs = len(eigenvalues)
            eigenvectors = np.zeros((total_size, num_vecs), dtype=complex)

            for idx, (i, j) in enumerate(eigenvector_indices):
                vec_2d = np.outer(
                    np.exp(2j * np.pi * i * np.arange(m) / m),
                    np.exp(2j * np.pi * j * np.arange(n) / n)
                )
                vec_2d /= np.linalg.norm(vec_2d)
                eigenvectors[:, idx] = vec_2d.flatten()

            # Convert to real (eigenvectors come in conjugate pairs)
            eigenvectors = eigenvectors.real

        else:
            # Default to path
            return self.compute_eigendecomposition_fft(grid_shape, "path", k)

        return eigenvalues, eigenvectors