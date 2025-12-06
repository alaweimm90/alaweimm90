"""
Iterative Methods for Large Linear Systems - Advanced Example
This example demonstrates advanced iterative methods for solving large sparse
linear systems, including convergence analysis and preconditioning techniques.
Learning Objectives:
- Understand iterative vs direct methods trade-offs
- Implement and analyze convergence of classical iterative methods
- Apply Krylov subspace methods for large systems
- Design and use preconditioners for acceleration
- Analyze computational complexity and memory requirements
- Handle ill-conditioned and indefinite systems
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
import time
import sys
import os
# Add Linear_Algebra package to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from core.matrix_operations import MatrixOperations, SpecialMatrices
from core.linear_systems import IterativeSolvers, DirectSolvers, LinearSystemUtils
from core.vector_operations import VectorOperations
def main():
    """Run advanced iterative methods example."""
    print("Iterative Methods for Large Linear Systems - Advanced Example")
    print("=" * 65)
    print("This example covers advanced iterative solvers and their applications")
    print("Learning: Jacobi, Gauss-Seidel, CG, GMRES, preconditioning, convergence analysis\n")
    # Set up Berkeley color scheme
    berkeley_blue = '#003262'
    california_gold = '#FDB515'
    # Classical iterative methods comparison
    demonstrate_classical_methods()
    # Krylov subspace methods
    demonstrate_krylov_methods()
    # Preconditioning techniques
    demonstrate_preconditioning()
    # Large sparse system application
    demonstrate_sparse_systems()
    # Convergence analysis and theory
    convergence_analysis()
    # Real-world application: 2D Poisson equation
    poisson_equation_example()
    print("\n" + "=" * 65)
    print("Advanced Iterative Methods Complete!")
    print("Key Learning Points:")
    print("• Classical methods work well for diagonally dominant systems")
    print("• Krylov methods are optimal for large sparse systems")
    print("• Preconditioning dramatically improves convergence")
    print("• Method choice depends on matrix structure and conditioning")
    print("• Memory vs computational trade-offs are crucial for large systems")
def demonstrate_classical_methods():
    """Compare classical iterative methods."""
    print("Classical Iterative Methods")
    print("=" * 30)
    # Create test system with known properties
    n = 10
    np.random.seed(42)
    # Create diagonally dominant matrix (ensures convergence)
    A = np.random.randn(n, n)
    A = A + A.T  # Make symmetric
    A = A + (np.abs(np.sum(A, axis=1)) + 1) * np.eye(n)  # Make diagonally dominant
    # Create right-hand side
    x_true = np.random.randn(n)
    b = A @ x_true
    print(f"System size: {n}x{n}")
    print(f"Matrix condition number: {np.linalg.cond(A):.2e}")
    print(f"True solution: {x_true[:5]}... (showing first 5 elements)")
    # Check diagonal dominance
    diag_vals = np.abs(np.diag(A))
    off_diag_sums = np.sum(np.abs(A), axis=1) - diag_vals
    is_diag_dominant = np.all(diag_vals >= off_diag_sums)
    print(f"Diagonally dominant: {is_diag_dominant}")
    # Test classical methods
    methods = ['jacobi', 'gauss_seidel', 'sor']
    colors = ['blue', 'red', 'green']
    results = {}
    tolerance = 1e-8
    max_iterations = 1000
    print(f"\nSolving with tolerance {tolerance:.0e}:")
    print("Method\t\tIterations\tFinal Error\tTime (ms)")
    print("-" * 55)
    plt.figure(figsize=(12, 8))
    for i, method in enumerate(methods):
        start_time = time.time()
        if method == 'jacobi':
            result = IterativeSolvers.jacobi(A, b, max_iterations=max_iterations, tolerance=tolerance)
        elif method == 'gauss_seidel':
            result = IterativeSolvers.gauss_seidel(A, b, max_iterations=max_iterations, tolerance=tolerance)
        elif method == 'sor':
            # Optimal omega for symmetric positive definite case
            eigenvals = np.linalg.eigvals(A)
            rho_jacobi = (np.max(eigenvals) - np.min(eigenvals)) / (np.max(eigenvals) + np.min(eigenvals))
            omega_opt = 2 / (1 + np.sqrt(1 - rho_jacobi**2))
            result = IterativeSolvers.sor(A, b, omega=omega_opt, max_iterations=max_iterations, tolerance=tolerance)
        elapsed_time = (time.time() - start_time) * 1000
        if result.success:
            final_error = np.linalg.norm(result.solution - x_true)
            print(f"{method.upper():<12}\t{result.iterations:<10}\t{final_error:.2e}\t{elapsed_time:.2f}")
            # Plot convergence
            residuals = result.info.get('residuals', [])
            if residuals:
                errors = [np.linalg.norm(A @ result.solution - b)] if not residuals else residuals
                plt.subplot(2, 2, 1)
                plt.semilogy(errors, label=method.upper(), color=colors[i], linewidth=2)
        else:
            print(f"{method.upper():<12}\tFAILED\t\t-\t\t{elapsed_time:.2f}")
        results[method] = result
    plt.subplot(2, 2, 1)
    plt.xlabel('Iteration')
    plt.ylabel('Residual Norm')
    plt.title('Convergence Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    # Effect of conditioning on convergence
    plt.subplot(2, 2, 2)
    condition_numbers = []
    jacobi_iterations = []
    for scale in np.logspace(-2, 2, 20):
        A_scaled = A + scale * np.eye(n)
        cond_num = np.linalg.cond(A_scaled)
        condition_numbers.append(cond_num)
        b_scaled = A_scaled @ x_true
        result = IterativeSolvers.jacobi(A_scaled, b_scaled, max_iterations=500, tolerance=1e-6)
        jacobi_iterations.append(result.iterations if result.success else 500)
    plt.loglog(condition_numbers, jacobi_iterations, 'o-', color='#003262', markersize=4)
    plt.xlabel('Condition Number')
    plt.ylabel('Jacobi Iterations to Convergence')
    plt.title('Conditioning vs Convergence Rate')
    plt.grid(True, alpha=0.3)
    # SOR parameter optimization
    plt.subplot(2, 2, 3)
    omegas = np.linspace(0.1, 1.9, 50)
    sor_iterations = []
    for omega in omegas:
        result = IterativeSolvers.sor(A, b, omega=omega, max_iterations=200, tolerance=1e-6)
        sor_iterations.append(result.iterations if result.success else 200)
    plt.plot(omegas, sor_iterations, color='#FDB515', linewidth=2)
    # Mark optimal omega
    eigenvals = np.linalg.eigvals(A)
    rho_jacobi = (np.max(eigenvals) - np.min(eigenvals)) / (np.max(eigenvals) + np.min(eigenvals))
    omega_opt = 2 / (1 + np.sqrt(1 - rho_jacobi**2))
    plt.axvline(x=omega_opt, color='red', linestyle='--', label=f'Optimal ω={omega_opt:.3f}')
    plt.xlabel('SOR Parameter ω')
    plt.ylabel('Iterations to Convergence')
    plt.title('SOR Parameter Optimization')
    plt.legend()
    plt.grid(True, alpha=0.3)
    # Method comparison on different matrix types
    plt.subplot(2, 2, 4)
    matrix_types = ['Diagonal\nDominant', 'Well\nConditioned', 'Ill\nConditioned', 'Nearly\nSingular']
    jacobi_perf = []
    gs_perf = []
    test_matrices = [
        A,  # Already diagonally dominant
        np.random.randn(n, n) @ np.random.randn(n, n).T + np.eye(n),  # Well-conditioned
        create_hilbert_matrix(n),  # Ill-conditioned
        create_hilbert_matrix(n) + 1e-10 * np.eye(n)  # Nearly singular
    ]
    for test_A in test_matrices:
        test_b = test_A @ x_true
        # Test Jacobi
        result_j = IterativeSolvers.jacobi(test_A, test_b, max_iterations=500, tolerance=1e-6)
        jacobi_perf.append(result_j.iterations if result_j.success else 500)
        # Test Gauss-Seidel
        result_gs = IterativeSolvers.gauss_seidel(test_A, test_b, max_iterations=500, tolerance=1e-6)
        gs_perf.append(result_gs.iterations if result_gs.success else 500)
    x_pos = np.arange(len(matrix_types))
    width = 0.35
    plt.bar(x_pos - width/2, jacobi_perf, width, label='Jacobi', color='#003262', alpha=0.7)
    plt.bar(x_pos + width/2, gs_perf, width, label='Gauss-Seidel', color='#FDB515', alpha=0.7)
    plt.xlabel('Matrix Type')
    plt.ylabel('Iterations to Convergence')
    plt.title('Method Performance vs Matrix Type')
    plt.xticks(x_pos, matrix_types)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
def demonstrate_krylov_methods():
    """Demonstrate Krylov subspace methods."""
    print("\n\nKrylov Subspace Methods")
    print("=" * 26)
    # Create test problems with different properties
    n = 100
    np.random.seed(42)
    # Symmetric positive definite (good for CG)
    A_spd = np.random.randn(n, n)
    A_spd = A_spd.T @ A_spd + 0.1 * np.eye(n)
    # General nonsymmetric (requires GMRES)
    A_general = np.random.randn(n, n) + 2 * np.eye(n)
    x_true = np.random.randn(n)
    print(f"System size: {n}x{n}")
    # Test problems
    problems = [
        ("Symmetric Positive Definite", A_spd),
        ("General Nonsymmetric", A_general)
    ]
    plt.figure(figsize=(15, 10))
    for prob_idx, (prob_name, A) in enumerate(problems):
        b = A @ x_true
        print(f"\n{prob_name} Matrix:")
        print(f"  Condition number: {np.linalg.cond(A):.2e}")
        print(f"  Symmetry error: {np.linalg.norm(A - A.T, 'fro'):.2e}")
        # Test appropriate methods
        if "Symmetric" in prob_name:
            methods = ["CG", "Direct"]
        else:
            methods = ["GMRES", "Direct"]
        method_results = {}
        for method in methods:
            start_time = time.time()
            if method == "CG":
                result = IterativeSolvers.conjugate_gradient(A, b, tolerance=1e-10, max_iterations=n)
            elif method == "GMRES":
                result = IterativeSolvers.gmres(A, b, tolerance=1e-10, max_iterations=n)
            elif method == "Direct":
                result = DirectSolvers.lu_solve(A, b)
            elapsed_time = time.time() - start_time
            if result.success:
                final_error = np.linalg.norm(result.solution - x_true)
                print(f"  {method}: {result.iterations} iterations, error={final_error:.2e}, time={elapsed_time:.3f}s")
                method_results[method] = (result, elapsed_time)
            else:
                print(f"  {method}: FAILED")
        # Convergence analysis for iterative methods
        plt.subplot(2, 3, prob_idx*3 + 1)
        if "Symmetric" in prob_name and "CG" in method_results:
            # CG convergence analysis
            result, _ = method_results["CG"]
            # Theoretical CG convergence bound
            kappa = np.linalg.cond(A)
            iterations = np.arange(1, result.iterations + 1)
            theoretical_bound = 2 * ((np.sqrt(kappa) - 1) / (np.sqrt(kappa) + 1))**iterations
            # Actual residual (approximate since we don't have full history)
            actual_error = np.linalg.norm(result.solution - x_true)
            plt.semilogy(iterations, theoretical_bound, '--', label='CG Theory', color='red', linewidth=2)
            plt.axhline(y=actual_error, color='#003262', label=f'Final Error', linewidth=2)
            plt.xlabel('Iteration')
            plt.ylabel('Error Bound')
            plt.title(f'{prob_name}\nCG Convergence')
            plt.legend()
            plt.grid(True, alpha=0.3)
        # Residual vs iteration (for GMRES or CG)
        plt.subplot(2, 3, prob_idx*3 + 2)
        # Run method again to get residual history
        residuals = []
        if "Symmetric" in prob_name:
            # Manual CG implementation to track residuals
            x = np.zeros(n)
            r = b.copy()
            p = r.copy()
            rsold = np.dot(r, r)
            for iteration in range(min(50, n)):
                Ap = A @ p
                alpha = rsold / np.dot(p, Ap)
                x = x + alpha * p
                r = r - alpha * Ap
                rsnew = np.dot(r, r)
                residuals.append(np.sqrt(rsnew))
                if np.sqrt(rsnew) < 1e-10:
                    break
                beta = rsnew / rsold
                p = r + beta * p
                rsold = rsnew
        else:
            # Use scipy GMRES with callback to track residuals
            def callback(residual):
                residuals.append(residual)
            spla.gmres(A, b, callback=callback, tol=1e-10, maxiter=50)
        if residuals:
            plt.semilogy(residuals, color='#003262', linewidth=2, marker='o', markersize=3)
            plt.xlabel('Iteration')
            plt.ylabel('Residual Norm')
            plt.title(f'{prob_name}\nResidual History')
            plt.grid(True, alpha=0.3)
        # Computational complexity comparison
        plt.subplot(2, 3, prob_idx*3 + 3)
        sizes = [20, 40, 60, 80, 100]
        iterative_times = []
        direct_times = []
        for size in sizes:
            # Create smaller test problem
            if "Symmetric" in prob_name:
                A_test = np.random.randn(size, size)
                A_test = A_test.T @ A_test + 0.1 * np.eye(size)
            else:
                A_test = np.random.randn(size, size) + 2 * np.eye(size)
            x_test = np.random.randn(size)
            b_test = A_test @ x_test
            # Time iterative method
            start_time = time.time()
            if "Symmetric" in prob_name:
                IterativeSolvers.conjugate_gradient(A_test, b_test, tolerance=1e-8)
            else:
                IterativeSolvers.gmres(A_test, b_test, tolerance=1e-8)
            iterative_times.append(time.time() - start_time)
            # Time direct method
            start_time = time.time()
            DirectSolvers.lu_solve(A_test, b_test)
            direct_times.append(time.time() - start_time)
        plt.loglog(sizes, iterative_times, 'o-', label='Iterative', color='#003262', linewidth=2)
        plt.loglog(sizes, direct_times, 's-', label='Direct', color='#FDB515', linewidth=2)
        # Theoretical complexity lines
        plt.loglog(sizes, np.array(sizes)**2 * iterative_times[0] / sizes[0]**2, '--',
                  alpha=0.5, color='#003262', label='O(n²)')
        plt.loglog(sizes, np.array(sizes)**3 * direct_times[0] / sizes[0]**3, '--',
                  alpha=0.5, color='#FDB515', label='O(n³)')
        plt.xlabel('Matrix Size')
        plt.ylabel('Time (seconds)')
        plt.title(f'{prob_name}\nComplexity Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
def demonstrate_preconditioning():
    """Demonstrate preconditioning techniques."""
    print("\n\nPreconditioning Techniques")
    print("=" * 29)
    # Create ill-conditioned test problem
    n = 50
    # Create matrix with wide range of eigenvalues
    eigenvals = np.logspace(-3, 3, n)  # Condition number ~ 1e6
    Q, _ = np.linalg.qr(np.random.randn(n, n))
    A = Q @ np.diag(eigenvals) @ Q.T
    x_true = np.random.randn(n)
    b = A @ x_true
    print(f"System size: {n}x{n}")
    print(f"Condition number: {np.linalg.cond(A):.2e}")
    print(f"Eigenvalue range: [{np.min(eigenvals):.2e}, {np.max(eigenvals):.2e}]")
    # Test different preconditioners
    preconditioners = {
        'None': None,
        'Jacobi': np.diag(np.diag(A)),
        'SSOR': None,  # Will be computed
        'Incomplete Cholesky': None  # Will be computed
    }
    # SSOR preconditioner
    D = np.diag(np.diag(A))
    L = np.tril(A, -1)
    U = np.triu(A, 1)
    omega = 1.0
    M1 = (D + omega * L) @ np.linalg.inv(D)
    M2 = np.linalg.inv(D + omega * U)
    preconditioners['SSOR'] = M1 @ M2
    # Incomplete Cholesky (simplified - just use diagonal + some off-diagonals)
    A_ic = np.diag(np.diag(A))
    # Add largest off-diagonal elements
    off_diag_mask = np.abs(A) > 0.1 * np.max(np.abs(A))
    A_ic = A_ic + np.tril(A * off_diag_mask, -1) + np.triu(A * off_diag_mask, 1)
    try:
        L_ic = np.linalg.cholesky(A_ic)
        preconditioners['Incomplete Cholesky'] = L_ic @ L_ic.T
    except:
        preconditioners['Incomplete Cholesky'] = np.eye(n)
    print("\nPreconditioner Performance:")
    print("Method\t\t\tIterations\tTime (ms)\tFinal Error")
    print("-" * 65)
    results = {}
    plt.figure(figsize=(15, 10))
    for i, (prec_name, prec_matrix) in enumerate(preconditioners.items()):
        start_time = time.time()
        # Solve with preconditioning
        if prec_matrix is None:
            # No preconditioning
            result = IterativeSolvers.conjugate_gradient(A, b, tolerance=1e-8, max_iterations=n)
        else:
            # With preconditioning - solve M^(-1)Ax = M^(-1)b
            try:
                M_inv = np.linalg.inv(prec_matrix)
                A_prec = M_inv @ A
                b_prec = M_inv @ b
                result = IterativeSolvers.conjugate_gradient(A_prec, b_prec, tolerance=1e-8, max_iterations=n)
            except:
                # Fallback if preconditioning fails
                result = IterativeSolvers.conjugate_gradient(A, b, tolerance=1e-8, max_iterations=n)
        elapsed_time = (time.time() - start_time) * 1000
        if result.success:
            final_error = np.linalg.norm(result.solution - x_true)
            print(f"{prec_name:<20}\t{result.iterations:<10}\t{elapsed_time:.2f}\t\t{final_error:.2e}")
            results[prec_name] = {
                'iterations': result.iterations,
                'time': elapsed_time,
                'error': final_error,
                'condition': np.linalg.cond(A_prec) if prec_matrix is not None else np.linalg.cond(A)
            }
        else:
            print(f"{prec_name:<20}\tFAILED\t\t{elapsed_time:.2f}\t\t-")
    # Visualization
    methods = list(results.keys())
    iterations = [results[m]['iterations'] for m in methods]
    times = [results[m]['time'] for m in methods]
    conditions = [results[m]['condition'] for m in methods]
    # Iterations comparison
    plt.subplot(2, 3, 1)
    bars = plt.bar(methods, iterations, color='#003262', alpha=0.7)
    plt.ylabel('Iterations to Convergence')
    plt.title('Preconditioning Effect on Iterations')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    # Add value labels on bars
    for bar, val in zip(bars, iterations):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                str(val), ha='center', va='bottom')
    # Time comparison
    plt.subplot(2, 3, 2)
    bars = plt.bar(methods, times, color='#FDB515', alpha=0.7)
    plt.ylabel('Time (ms)')
    plt.title('Computational Time')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    # Condition number comparison
    plt.subplot(2, 3, 3)
    bars = plt.bar(methods, np.log10(conditions), color='green', alpha=0.7)
    plt.ylabel('log₁₀(Condition Number)')
    plt.title('Effective Condition Number')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    # Convergence history comparison (theoretical)
    plt.subplot(2, 3, 4)
    for method in methods:
        if method in results:
            kappa = results[method]['condition']
            iterations = np.arange(1, results[method]['iterations'] + 1)
            bound = 2 * ((np.sqrt(kappa) - 1) / (np.sqrt(kappa) + 1))**iterations
            plt.semilogy(iterations, bound, label=method, linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Error Bound')
    plt.title('Theoretical Convergence Bounds')
    plt.legend()
    plt.grid(True, alpha=0.3)
    # Preconditioning effectiveness vs condition number
    plt.subplot(2, 3, 5)
    # Test preconditioning on matrices with different condition numbers
    cond_numbers = np.logspace(1, 6, 20)
    no_prec_iters = []
    jacobi_prec_iters = []
    for cond_target in cond_numbers:
        # Create matrix with specific condition number
        eigs = np.linspace(1, cond_target, n)
        Q, _ = np.linalg.qr(np.random.randn(n, n))
        A_test = Q @ np.diag(eigs) @ Q.T
        b_test = A_test @ x_true
        # No preconditioning
        result1 = IterativeSolvers.conjugate_gradient(A_test, b_test, tolerance=1e-6, max_iterations=100)
        no_prec_iters.append(result1.iterations if result1.success else 100)
        # Jacobi preconditioning
        M_jacobi = np.diag(np.diag(A_test))
        try:
            M_inv = np.linalg.inv(M_jacobi)
            A_prec = M_inv @ A_test
            b_prec = M_inv @ b_test
            result2 = IterativeSolvers.conjugate_gradient(A_prec, b_prec, tolerance=1e-6, max_iterations=100)
            jacobi_prec_iters.append(result2.iterations if result2.success else 100)
        except:
            jacobi_prec_iters.append(100)
    plt.loglog(cond_numbers, no_prec_iters, 'o-', label='No Preconditioning', color='#003262')
    plt.loglog(cond_numbers, jacobi_prec_iters, 's-', label='Jacobi Preconditioning', color='#FDB515')
    plt.xlabel('Condition Number')
    plt.ylabel('Iterations to Convergence')
    plt.title('Preconditioning vs Conditioning')
    plt.legend()
    plt.grid(True, alpha=0.3)
    # Memory usage comparison
    plt.subplot(2, 3, 6)
    # Estimate memory usage (in terms of matrix elements stored)
    memory_usage = {
        'None': n*n,  # Store full matrix
        'Jacobi': n,  # Store diagonal only
        'SSOR': n*n,  # Full matrix (for simplicity)
        'Incomplete Cholesky': n*n*0.1  # Assume 10% sparsity
    }
    methods_mem = list(memory_usage.keys())
    memory_vals = list(memory_usage.values())
    bars = plt.bar(methods_mem, np.array(memory_vals) / (n*n) * 100, color='purple', alpha=0.7)
    plt.ylabel('Memory Usage (% of full matrix)')
    plt.title('Memory Requirements')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
def demonstrate_sparse_systems():
    """Demonstrate methods for large sparse systems."""
    print("\n\nLarge Sparse Systems")
    print("=" * 23)
    # Create large sparse matrix (e.g., from finite differences)
    n = 200
    # 1D Laplacian with Dirichlet boundary conditions
    # -u''(x) = f(x), u(0) = u(1) = 0
    # Discretized: (-u_{i-1} + 2u_i - u_{i+1})/h² = f_i
    h = 1.0 / (n + 1)
    # Create sparse matrix efficiently
    diagonals = [-1*np.ones(n-1), 2*np.ones(n), -1*np.ones(n-1)]
    A_sparse = sp.diags(diagonals, [-1, 0, 1], shape=(n, n), format='csr') / h**2
    # Right-hand side (source term)
    x_points = np.linspace(h, 1-h, n)
    f = np.sin(np.pi * x_points)  # Source function
    b_sparse = f
    print(f"System size: {n}x{n}")
    print(f"Matrix sparsity: {A_sparse.nnz / (n*n) * 100:.1f}% non-zero")
    print(f"Condition number: {np.linalg.cond(A_sparse.toarray()):.2e}")
    # Analytical solution for verification
    x_true_analytical = np.sin(np.pi * x_points) / (np.pi**2)
    # Compare dense vs sparse operations
    print("\nDense vs Sparse Performance:")
    print("Method\t\tSetup (ms)\tSolve (ms)\tMemory (MB)")
    print("-" * 55)
    # Dense matrix approach
    start_time = time.time()
    A_dense = A_sparse.toarray()
    setup_time_dense = (time.time() - start_time) * 1000
    start_time = time.time()
    result_dense = DirectSolvers.lu_solve(A_dense, b_sparse)
    solve_time_dense = (time.time() - start_time) * 1000
    memory_dense = A_dense.nbytes / (1024**2)
    print(f"Dense\t\t{setup_time_dense:.2f}\t\t{solve_time_dense:.2f}\t\t{memory_dense:.2f}")
    # Sparse matrix approach
    start_time = time.time()
    # Matrix already in sparse format
    setup_time_sparse = 0.1  # Minimal setup
    start_time = time.time()
    x_sparse = spla.spsolve(A_sparse, b_sparse)
    solve_time_sparse = (time.time() - start_time) * 1000
    memory_sparse = A_sparse.data.nbytes / (1024**2)
    print(f"Sparse\t\t{setup_time_sparse:.2f}\t\t{solve_time_sparse:.2f}\t\t{memory_sparse:.2f}")
    # Iterative methods for sparse systems
    print("\nIterative Methods for Sparse Systems:")
    print("Method\t\tIterations\tTime (ms)\tError")
    print("-" * 45)
    # Convert to dense for our iterative solvers
    iterative_methods = ['CG', 'GMRES (scipy)']
    # CG using our implementation
    start_time = time.time()
    result_cg = IterativeSolvers.conjugate_gradient(A_dense, b_sparse, tolerance=1e-8)
    time_cg = (time.time() - start_time) * 1000
    error_cg = np.linalg.norm(result_cg.solution - x_true_analytical) if result_cg.success else np.inf
    print(f"CG (ours)\t{result_cg.iterations if result_cg.success else 'FAIL'}\t\t{time_cg:.2f}\t\t{error_cg:.2e}")
    # GMRES using scipy (can work directly with sparse matrices)
    start_time = time.time()
    x_gmres, info = spla.gmres(A_sparse, b_sparse, tol=1e-8)
    time_gmres = (time.time() - start_time) * 1000
    error_gmres = np.linalg.norm(x_gmres - x_true_analytical)
    print(f"GMRES (scipy)\t{info if info == 0 else 'FAIL'}\t\t{time_gmres:.2f}\t\t{error_gmres:.2e}")
    # Visualization
    plt.figure(figsize=(15, 10))
    # Matrix structure
    plt.subplot(2, 3, 1)
    plt.spy(A_sparse, markersize=1, color='#003262')
    plt.title('Sparse Matrix Structure')
    plt.xlabel('Column Index')
    plt.ylabel('Row Index')
    # Solution comparison
    plt.subplot(2, 3, 2)
    plt.plot(x_points, x_true_analytical, 'r-', linewidth=2, label='Analytical')
    plt.plot(x_points, result_dense.solution, 'b--', linewidth=2, label='Dense Solver')
    plt.plot(x_points, x_sparse, 'g:', linewidth=2, label='Sparse Solver')
    plt.plot(x_points, x_gmres, 'm-.', linewidth=2, label='GMRES')
    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.title('Solution Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    # Error analysis
    plt.subplot(2, 3, 3)
    error_dense = np.abs(result_dense.solution - x_true_analytical)
    error_sparse = np.abs(x_sparse - x_true_analytical)
    error_gmres = np.abs(x_gmres - x_true_analytical)
    plt.semilogy(x_points, error_dense, 'b-', label='Dense Error')
    plt.semilogy(x_points, error_sparse, 'g-', label='Sparse Error')
    plt.semilogy(x_points, error_gmres, 'm-', label='GMRES Error')
    plt.xlabel('x')
    plt.ylabel('Absolute Error')
    plt.title('Solution Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    # Performance scaling
    plt.subplot(2, 3, 4)
    sizes = [50, 100, 150, 200]
    dense_times = []
    sparse_times = []
    memory_ratios = []
    for size in sizes:
        h_test = 1.0 / (size + 1)
        diagonals = [-1*np.ones(size-1), 2*np.ones(size), -1*np.ones(size-1)]
        A_test_sparse = sp.diags(diagonals, [-1, 0, 1], shape=(size, size), format='csr') / h_test**2
        A_test_dense = A_test_sparse.toarray()
        b_test = np.random.randn(size)
        # Time dense solve
        start_time = time.time()
        DirectSolvers.lu_solve(A_test_dense, b_test)
        dense_times.append(time.time() - start_time)
        # Time sparse solve
        start_time = time.time()
        spla.spsolve(A_test_sparse, b_test)
        sparse_times.append(time.time() - start_time)
        # Memory ratio
        memory_ratios.append(A_test_sparse.data.nbytes / A_test_dense.nbytes)
    plt.loglog(sizes, dense_times, 'o-', label='Dense', color='#003262', linewidth=2)
    plt.loglog(sizes, sparse_times, 's-', label='Sparse', color='#FDB515', linewidth=2)
    plt.xlabel('Matrix Size')
    plt.ylabel('Solve Time (s)')
    plt.title('Scaling Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    # Memory usage
    plt.subplot(2, 3, 5)
    plt.semilogx(sizes, np.array(memory_ratios) * 100, 'o-', color='green', linewidth=2)
    plt.xlabel('Matrix Size')
    plt.ylabel('Sparse/Dense Memory (%)')
    plt.title('Memory Efficiency')
    plt.grid(True, alpha=0.3)
    # Sparsity pattern for different problems
    plt.subplot(2, 3, 6)
    # Create different sparse patterns
    n_small = 50
    # 2D Laplacian (5-point stencil)
    def create_2d_laplacian(nx, ny):
        n = nx * ny
        row = []
        col = []
        data = []
        for i in range(nx):
            for j in range(ny):
                idx = i * ny + j
                # Center point
                row.append(idx)
                col.append(idx)
                data.append(4.0)
                # Neighbors
                if i > 0:  # Left
                    row.append(idx)
                    col.append((i-1) * ny + j)
                    data.append(-1.0)
                if i < nx-1:  # Right
                    row.append(idx)
                    col.append((i+1) * ny + j)
                    data.append(-1.0)
                if j > 0:  # Down
                    row.append(idx)
                    col.append(i * ny + (j-1))
                    data.append(-1.0)
                if j < ny-1:  # Up
                    row.append(idx)
                    col.append(i * ny + (j+1))
                    data.append(-1.0)
        return sp.csr_matrix((data, (row, col)), shape=(n, n))
    A_2d = create_2d_laplacian(int(np.sqrt(n_small)), int(np.sqrt(n_small)))
    plt.spy(A_2d, markersize=1, color='#003262')
    plt.title('2D Laplacian Sparsity Pattern')
    plt.xlabel('Column')
    plt.ylabel('Row')
    plt.tight_layout()
    plt.show()
def convergence_analysis():
    """Analyze convergence theory and practice."""
    print("\n\nConvergence Analysis")
    print("=" * 23)
    # Create test matrices with known spectral properties
    n = 20
    # Different matrix types
    matrices = {}
    # Well-conditioned matrix
    np.random.seed(42)
    A = np.random.randn(n, n)
    matrices['Well-conditioned'] = A.T @ A + np.eye(n)
    # Ill-conditioned matrix
    matrices['Ill-conditioned'] = create_hilbert_matrix(n)
    # Clustered eigenvalues
    eigs_clustered = np.concatenate([np.ones(n//2), 100*np.ones(n//2)])
    Q, _ = np.linalg.qr(np.random.randn(n, n))
    matrices['Clustered spectrum'] = Q @ np.diag(eigs_clustered) @ Q.T
    # Create analysis
    plt.figure(figsize=(15, 12))
    for idx, (name, A) in enumerate(matrices.items()):
        # Eigenvalue analysis
        eigenvals = np.linalg.eigvals(A)
        eigenvals = np.sort(np.real(eigenvals))
        cond_num = np.max(eigenvals) / np.min(eigenvals)
        print(f"\n{name} Matrix:")
        print(f"  Condition number: {cond_num:.2e}")
        print(f"  Eigenvalue range: [{np.min(eigenvals):.2e}, {np.max(eigenvals):.2e}]")
        # Plot eigenvalue spectrum
        plt.subplot(3, 4, idx*4 + 1)
        plt.semilogy(range(1, len(eigenvals)+1), eigenvals, 'o-', color='#003262', markersize=4)
        plt.xlabel('Index')
        plt.ylabel('Eigenvalue')
        plt.title(f'{name}\nEigenvalue Spectrum')
        plt.grid(True, alpha=0.3)
        # Convergence rate prediction vs actual
        x_true = np.random.randn(n)
        b = A @ x_true
        # Run CG with detailed tracking
        x = np.zeros(n)
        r = b.copy()
        p = r.copy()
        rsold = np.dot(r, r)
        errors_actual = []
        residuals_actual = []
        for iteration in range(min(30, n)):
            error = np.linalg.norm(x - x_true)
            residual = np.linalg.norm(r)
            errors_actual.append(error)
            residuals_actual.append(residual)
            if residual < 1e-12:
                break
            Ap = A @ p
            alpha = rsold / np.dot(p, Ap)
            x = x + alpha * p
            r = r - alpha * Ap
            rsnew = np.dot(r, r)
            beta = rsnew / rsold
            p = r + beta * p
            rsold = rsnew
        # Theoretical convergence rate
        kappa = cond_num
        iterations = np.arange(1, len(errors_actual) + 1)
        theoretical_rate = 2 * ((np.sqrt(kappa) - 1) / (np.sqrt(kappa) + 1))**iterations
        # Plot convergence comparison
        plt.subplot(3, 4, idx*4 + 2)
        plt.semilogy(iterations, errors_actual, 'o-', label='Actual Error', color='#003262', markersize=4)
        plt.semilogy(iterations, theoretical_rate[:len(iterations)], '--', label='CG Theory', color='red', linewidth=2)
        plt.xlabel('Iteration')
        plt.ylabel('Error')
        plt.title(f'{name}\nConvergence Rate')
        plt.legend()
        plt.grid(True, alpha=0.3)
        # Residual vs error
        plt.subplot(3, 4, idx*4 + 3)
        plt.loglog(residuals_actual, errors_actual, 'o-', color='#003262', markersize=4)
        plt.xlabel('Residual Norm')
        plt.ylabel('Error Norm')
        plt.title(f'{name}\nResidual vs Error')
        plt.grid(True, alpha=0.3)
        # Method comparison for this matrix
        plt.subplot(3, 4, idx*4 + 4)
        methods_test = ['jacobi', 'gauss_seidel', 'cg']
        iterations_needed = []
        for method in methods_test:
            if method == 'jacobi':
                result = IterativeSolvers.jacobi(A, b, tolerance=1e-6, max_iterations=100)
            elif method == 'gauss_seidel':
                result = IterativeSolvers.gauss_seidel(A, b, tolerance=1e-6, max_iterations=100)
            elif method == 'cg':
                result = IterativeSolvers.conjugate_gradient(A, b, tolerance=1e-6, max_iterations=100)
            iterations_needed.append(result.iterations if result.success else 100)
        bars = plt.bar(methods_test, iterations_needed, color=['blue', 'red', 'green'], alpha=0.7)
        plt.ylabel('Iterations to Convergence')
        plt.title(f'{name}\nMethod Comparison')
        plt.xticks(rotation=45)
        # Add value labels
        for bar, val in zip(bars, iterations_needed):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    str(val), ha='center', va='bottom')
    plt.tight_layout()
    plt.show()
    # Theoretical analysis summary
    print("\nConvergence Theory Summary:")
    print("=" * 30)
    print("• CG converges in at most n iterations for SPD matrices")
    print("• Convergence rate depends on condition number κ = λ_max/λ_min")
    print("• CG error bound: ||e_k|| ≤ 2||e_0|| * ((√κ-1)/(√κ+1))^k")
    print("• Clustered eigenvalues lead to faster convergence")
    print("• Preconditioning improves effective condition number")
def poisson_equation_example():
    """Solve 2D Poisson equation using iterative methods."""
    print("\n\nReal-World Application: 2D Poisson Equation")
    print("=" * 50)
    print("Solving: -∇²u = f in Ω = [0,1]×[0,1]")
    print("         u = 0 on ∂Ω (boundary)")
    print("where f(x,y) = 2π²sin(πx)sin(πy)")
    print("Analytical solution: u(x,y) = sin(πx)sin(πy)")
    # Grid setup
    nx, ny = 32, 32  # Grid points in each direction
    dx, dy = 1.0/(nx+1), 1.0/(ny+1)
    x = np.linspace(dx, 1-dx, nx)
    y = np.linspace(dy, 1-dy, ny)
    X, Y = np.meshgrid(x, y)
    # Right-hand side
    f = 2 * np.pi**2 * np.sin(np.pi * X) * np.sin(np.pi * Y)
    f_vec = f.flatten()
    # Analytical solution
    u_analytical = np.sin(np.pi * X) * np.sin(np.pi * Y)
    u_analytical_vec = u_analytical.flatten()
    print(f"Grid size: {nx}×{ny} = {nx*ny} unknowns")
    # Create 2D Laplacian matrix using finite differences
    # 5-point stencil: [-1, 4, -1] in x and y directions
    n = nx * ny
    # Build matrix efficiently
    diagonals = []
    offsets = []
    # Main diagonal
    diagonals.append(4 * np.ones(n))
    offsets.append(0)
    # X-direction connections
    diag_x = -np.ones(n-1)
    # Remove connections across y-boundary
    for i in range(ny-1, n-1, ny):
        diag_x[i] = 0
    diagonals.extend([diag_x, diag_x])
    offsets.extend([-1, 1])
    # Y-direction connections
    diagonals.extend([-np.ones(n-ny), -np.ones(n-ny)])
    offsets.extend([-ny, ny])
    A_2d = sp.diags(diagonals, offsets, shape=(n, n), format='csr')
    A_2d = A_2d / dx**2  # Scale by grid spacing
    print(f"Matrix sparsity: {A_2d.nnz/(n*n)*100:.1f}% non-zero")
    print(f"Condition number estimate: {1/(dx**2):.2e}")
    # Solve using different methods
    methods = {
        'Direct (scipy)': lambda A, b: spla.spsolve(A, b),
        'CG (scipy)': lambda A, b: spla.cg(A, b, tol=1e-8)[0],
        'GMRES (scipy)': lambda A, b: spla.gmres(A, b, tol=1e-8)[0]
    }
    print("\nSolver Performance:")
    print("Method\t\tTime (ms)\tError\t\tMemory")
    print("-" * 50)
    solutions = {}
    for method_name, solver in methods.items():
        start_time = time.time()
        try:
            if 'scipy' in method_name:
                u_numerical = solver(A_2d, f_vec)
            else:
                # Our implementation would go here
                u_numerical = solver(A_2d.toarray(), f_vec)
            solve_time = (time.time() - start_time) * 1000
            # Compute error
            error = np.linalg.norm(u_numerical - u_analytical_vec)
            # Memory estimate (rough)
            memory_mb = A_2d.data.nbytes / (1024**2) if sp.issparse(A_2d) else A_2d.nbytes / (1024**2)
            print(f"{method_name:<15}\t{solve_time:.2f}\t\t{error:.2e}\t{memory_mb:.2f} MB")
            solutions[method_name] = u_numerical.reshape(ny, nx)
        except Exception as e:
            print(f"{method_name:<15}\tFAILED\t\t-\t\t-")
    # Visualization
    plt.figure(figsize=(15, 12))
    # Analytical solution
    plt.subplot(2, 3, 1)
    contour = plt.contourf(X, Y, u_analytical, levels=20, cmap='viridis')
    plt.colorbar(contour)
    plt.title('Analytical Solution')
    plt.xlabel('x')
    plt.ylabel('y')
    # Numerical solutions
    plot_idx = 2
    for method_name, u_num in solutions.items():
        if plot_idx <= 3:
            plt.subplot(2, 3, plot_idx)
            contour = plt.contourf(X, Y, u_num, levels=20, cmap='viridis')
            plt.colorbar(contour)
            plt.title(f'{method_name}')
            plt.xlabel('x')
            plt.ylabel('y')
            plot_idx += 1
    # Error visualization
    if solutions:
        first_solution = list(solutions.values())[0]
        error_field = np.abs(first_solution - u_analytical)
        plt.subplot(2, 3, 4)
        contour = plt.contourf(X, Y, error_field, levels=20, cmap='hot')
        plt.colorbar(contour)
        plt.title('Absolute Error')
        plt.xlabel('x')
        plt.ylabel('y')
    # Matrix structure
    plt.subplot(2, 3, 5)
    plt.spy(A_2d[:200, :200], markersize=0.5, color='#003262')  # Show subset for visibility
    plt.title('Matrix Sparsity Pattern\n(200×200 subset)')
    plt.xlabel('Column')
    plt.ylabel('Row')
    # Cross-section comparison
    plt.subplot(2, 3, 6)
    # Take cross-section at y = 0.5
    mid_idx = ny // 2
    x_cross = x
    u_analytical_cross = u_analytical[mid_idx, :]
    plt.plot(x_cross, u_analytical_cross, 'r-', linewidth=2, label='Analytical')
    for method_name, u_num in solutions.items():
        if 'Direct' in method_name:  # Just show one numerical solution
            u_numerical_cross = u_num[mid_idx, :]
            plt.plot(x_cross, u_numerical_cross, 'b--', linewidth=2, label='Numerical')
            break
    plt.xlabel('x')
    plt.ylabel('u(x, 0.5)')
    plt.title('Cross-section at y = 0.5')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    # Convergence study
    print("\nGrid Convergence Study:")
    print("Grid\tError\t\tOrder")
    print("-" * 30)
    grid_sizes = [8, 16, 32]
    errors = []
    for grid_size in grid_sizes:
        # Solve on different grid sizes
        dx_test = 1.0 / (grid_size + 1)
        x_test = np.linspace(dx_test, 1-dx_test, grid_size)
        X_test, Y_test = np.meshgrid(x_test, x_test)
        # Create system
        n_test = grid_size * grid_size
        diagonals_test = []
        offsets_test = []
        diagonals_test.append(4 * np.ones(n_test))
        offsets_test.append(0)
        diag_x_test = -np.ones(n_test-1)
        for i in range(grid_size-1, n_test-1, grid_size):
            diag_x_test[i] = 0
        diagonals_test.extend([diag_x_test, diag_x_test])
        offsets_test.extend([-1, 1])
        diagonals_test.extend([-np.ones(n_test-grid_size), -np.ones(n_test-grid_size)])
        offsets_test.extend([-grid_size, grid_size])
        A_test = sp.diags(diagonals_test, offsets_test, shape=(n_test, n_test), format='csr')
        A_test = A_test / dx_test**2
        # RHS and analytical solution
        f_test = 2 * np.pi**2 * np.sin(np.pi * X_test) * np.sin(np.pi * Y_test)
        u_analytical_test = np.sin(np.pi * X_test) * np.sin(np.pi * Y_test)
        # Solve
        u_numerical_test = spla.spsolve(A_test, f_test.flatten())
        # Compute error
        error = np.linalg.norm(u_numerical_test - u_analytical_test.flatten(), np.inf)
        errors.append(error)
        # Convergence order
        if len(errors) > 1:
            order = np.log(errors[-2] / errors[-1]) / np.log(2)
            print(f"{grid_size}×{grid_size}\t{error:.2e}\t{order:.2f}")
        else:
            print(f"{grid_size}×{grid_size}\t{error:.2e}\t-")
    print("\nExpected convergence order: 2.0 (second-order finite differences)")
def create_hilbert_matrix(n):
    """Create Hilbert matrix (ill-conditioned test case)."""
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            H[i, j] = 1.0 / (i + j + 1)
    return H
if __name__ == "__main__":
    main()