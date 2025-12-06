#!/usr/bin/env python3
"""
FFT-Laplace Preconditioning Example for QAP

This example demonstrates the performance improvements achieved by
using FFT-based Laplace preconditioning for structured QAP instances.
"""

import time
import numpy as np
from Librex.methods.qap_enhanced import PreconditionedQAPSolver
from Librex.methods.qap_enhanced.laplace_detector import LaplacianDetector


def create_grid_qap(grid_size: int) -> tuple:
    """
    Create a QAP instance based on grid graph structure.

    This represents facility layout on a grid where:
    - Flow matrix represents communication between facilities
    - Distance matrix represents grid distances
    """
    n = grid_size * grid_size

    # Create grid Laplacian for distance matrix (grid structure)
    D = np.zeros((n, n))
    for i in range(grid_size):
        for j in range(grid_size):
            idx = i * grid_size + j

            # Diagonal (degree)
            degree = 0

            # Connect to neighbors
            if i > 0:  # Up
                neighbor = (i-1) * grid_size + j
                D[idx, neighbor] = -1
                D[neighbor, idx] = -1
                degree += 1

            if i < grid_size - 1:  # Down
                neighbor = (i+1) * grid_size + j
                D[idx, neighbor] = -1
                D[neighbor, idx] = -1
                degree += 1

            if j > 0:  # Left
                neighbor = i * grid_size + (j-1)
                D[idx, neighbor] = -1
                D[neighbor, idx] = -1
                degree += 1

            if j < grid_size - 1:  # Right
                neighbor = i * grid_size + (j+1)
                D[idx, neighbor] = -1
                D[neighbor, idx] = -1
                degree += 1

            D[idx, idx] = degree

    # Create flow matrix with community structure
    F = np.random.exponential(scale=1.0, size=(n, n))
    F = (F + F.T) / 2  # Make symmetric

    # Add stronger flows within communities (2x2 blocks)
    for i in range(0, grid_size, 2):
        for j in range(0, grid_size, 2):
            # Indices of 2x2 block
            block_indices = []
            for di in range(min(2, grid_size - i)):
                for dj in range(min(2, grid_size - j)):
                    block_indices.append((i + di) * grid_size + (j + dj))

            # Strengthen flows within block
            for idx1 in block_indices:
                for idx2 in block_indices:
                    if idx1 != idx2:
                        F[idx1, idx2] *= 3

    return F, D


def benchmark_solvers(flow_matrix, distance_matrix):
    """Compare standard vs preconditioned solver performance."""

    print("\n" + "="*60)
    print("BENCHMARKING QAP SOLVERS")
    print("="*60)

    # Analyze structure
    detector = LaplacianDetector()
    flow_type, flow_meta = detector.detect_structure(flow_matrix)
    dist_type, dist_meta = detector.detect_structure(distance_matrix)

    print(f"\nProblem Analysis:")
    print(f"  Size: {len(flow_matrix)} facilities")
    print(f"  Flow matrix type: {flow_type.value}")
    print(f"  Distance matrix type: {dist_type.value}")

    if dist_type.value == 'grid_2d':
        grid_dims = dist_meta.get('grid_dimension', 'unknown')
        print(f"  Grid dimensions detected: {grid_dims}")

    # Test 1: Standard solver (no preconditioning)
    print("\n1. Standard Solver (Random Initialization):")
    print("-" * 40)

    solver_standard = PreconditionedQAPSolver(enable_fft=False, verbose=True)

    start_time = time.time()
    result_standard = solver_standard.solve(
        flow_matrix, distance_matrix,
        method='simulated_annealing',
        config={'max_iterations': 5000}
    )
    time_standard = time.time() - start_time

    print(f"  Time: {time_standard:.2f} seconds")
    print(f"  Objective: {result_standard['objective']:.2f}")
    print(f"  Iterations: {result_standard.get('iterations', 'N/A')}")

    # Test 2: Preconditioned solver with FFT
    print("\n2. FFT-Preconditioned Solver:")
    print("-" * 40)

    solver_preconditioned = PreconditionedQAPSolver(enable_fft=True, verbose=True)

    start_time = time.time()
    result_preconditioned = solver_preconditioned.solve(
        flow_matrix, distance_matrix,
        method='simulated_annealing',
        config={'max_iterations': 5000}
    )
    time_preconditioned = time.time() - start_time

    print(f"  Time: {time_preconditioned:.2f} seconds")
    print(f"  Objective: {result_preconditioned['objective']:.2f}")
    print(f"  Iterations: {result_preconditioned.get('iterations', 'N/A')}")
    print(f"  Techniques used: {result_preconditioned['preconditioning']['techniques']}")

    # Calculate improvements
    speedup = time_standard / time_preconditioned if time_preconditioned > 0 else 1.0
    quality_improvement = (
        (result_standard['objective'] - result_preconditioned['objective']) /
        result_standard['objective'] * 100
    ) if result_standard['objective'] > 0 else 0

    print("\n" + "="*60)
    print("PERFORMANCE COMPARISON")
    print("="*60)
    print(f"  Speedup: {speedup:.2f}x faster")
    print(f"  Quality improvement: {quality_improvement:.1f}%")
    print(f"  Estimated speedup: {result_preconditioned['preconditioning']['speedup_estimate']:.1f}x")

    return result_preconditioned


def demonstrate_fft_operations():
    """Demonstrate FFT-based operations on grid Laplacians."""

    print("\n" + "="*60)
    print("FFT OPERATIONS DEMONSTRATION")
    print("="*60)

    from Librex.methods.qap_enhanced.fft_ops import FFTOperations

    fft_ops = FFTOperations()
    grid_shape = (8, 8)
    n = grid_shape[0] * grid_shape[1]

    print(f"\nGrid shape: {grid_shape}")
    print(f"Problem size: {n} nodes")

    # Generate Fiedler vector using FFT
    print("\n1. Computing Fiedler vector using FFT...")
    start = time.time()
    fiedler_fft = fft_ops.fiedler_vector_fft(grid_shape, boundary='path')
    time_fft = time.time() - start
    print(f"   FFT computation time: {time_fft*1000:.2f} ms")

    # Compare with standard eigendecomposition
    print("\n2. Computing Fiedler vector using standard eigendecomposition...")

    # Create grid Laplacian
    L = np.zeros((n, n))
    for i in range(grid_shape[0]):
        for j in range(grid_shape[1]):
            idx = i * grid_shape[1] + j
            degree = 0

            if i > 0:
                neighbor = (i-1) * grid_shape[1] + j
                L[idx, neighbor] = -1
                degree += 1
            if i < grid_shape[0] - 1:
                neighbor = (i+1) * grid_shape[1] + j
                L[idx, neighbor] = -1
                degree += 1
            if j > 0:
                neighbor = i * grid_shape[1] + (j-1)
                L[idx, neighbor] = -1
                degree += 1
            if j < grid_shape[1] - 1:
                neighbor = i * grid_shape[1] + (j+1)
                L[idx, neighbor] = -1
                degree += 1

            L[idx, idx] = degree

    start = time.time()
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    fiedler_standard = eigenvectors[:, 1]
    time_standard = time.time() - start
    print(f"   Standard computation time: {time_standard*1000:.2f} ms")

    print(f"\n   Speedup: {time_standard/time_fft:.1f}x faster with FFT")

    # Verify similarity (up to sign)
    similarity = min(
        np.linalg.norm(fiedler_fft - fiedler_standard),
        np.linalg.norm(fiedler_fft + fiedler_standard)
    )
    print(f"   Vector difference (norm): {similarity:.6f}")

    # Demonstrate fast matrix-vector multiplication
    print("\n3. Fast Laplacian matrix-vector multiplication:")

    test_vector = np.random.randn(n)

    # FFT-based
    start = time.time()
    result_fft = fft_ops.fft_laplace_mult('path', test_vector, grid_shape)
    time_fft_mult = time.time() - start

    # Standard
    start = time.time()
    result_standard = L @ test_vector
    time_standard_mult = time.time() - start

    print(f"   FFT multiplication time: {time_fft_mult*1000:.3f} ms")
    print(f"   Standard multiplication time: {time_standard_mult*1000:.3f} ms")
    print(f"   Speedup: {time_standard_mult/time_fft_mult:.1f}x")
    print(f"   Result difference (norm): {np.linalg.norm(result_fft - result_standard):.6f}")


def main():
    """Main demonstration."""

    print("FFT-LAPLACE PRECONDITIONING FOR QAP")
    print("="*60)

    # Test different grid sizes
    grid_sizes = [4, 6, 8]

    for grid_size in grid_sizes:
        print(f"\n\nTESTING {grid_size}×{grid_size} GRID ({grid_size**2} facilities)")
        print("="*60)

        # Create structured QAP instance
        flow_matrix, distance_matrix = create_grid_qap(grid_size)

        # Benchmark solvers
        result = benchmark_solvers(flow_matrix, distance_matrix)

        # Show solution quality
        print(f"\nBest solution found:")
        print(f"  Permutation: {result['solution'][:10]}..." if len(result['solution']) > 10
              else f"  Permutation: {result['solution']}")

    # Demonstrate FFT operations
    demonstrate_fft_operations()

    print("\n" + "="*60)
    print("DEMONSTRATION COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()