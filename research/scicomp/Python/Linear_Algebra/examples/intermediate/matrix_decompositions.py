"""
Matrix Decompositions - Intermediate Example
This example demonstrates matrix decomposition techniques including LU, QR,
Cholesky, and SVD decompositions with practical applications.
Learning Objectives:
- Understand different matrix decomposition methods
- Apply decompositions to solve linear systems efficiently
- Use SVD for data analysis and dimensionality reduction
- Analyze matrix properties through decompositions
- Implement numerical algorithms using decompositions
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import os
# Add Linear_Algebra package to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from core.matrix_operations import MatrixOperations, MatrixDecompositions, SpecialMatrices
from core.linear_systems import DirectSolvers
from core.vector_operations import VectorOperations
def main():
    """Run matrix decompositions example."""
    print("Matrix Decompositions - Intermediate Example")
    print("=" * 50)
    print("This example covers matrix decomposition techniques and applications")
    print("Learning: LU, QR, Cholesky, SVD decompositions and their uses\n")
    # Set up Berkeley color scheme
    berkeley_blue = '#003262'
    california_gold = '#FDB515'
    # LU decomposition demonstration
    demonstrate_lu_decomposition()
    # QR decomposition and applications
    demonstrate_qr_decomposition()
    # Cholesky decomposition for positive definite matrices
    demonstrate_cholesky_decomposition()
    # SVD and its applications
    demonstrate_svd_applications()
    # Eigenvalue decomposition
    demonstrate_eigenvalue_decomposition()
    # Practical application: Image compression using SVD
    image_compression_example()
    print("\n" + "=" * 50)
    print("Matrix Decompositions Complete!")
    print("Key Learning Points:")
    print("• LU decomposition efficiently solves multiple systems with same matrix")
    print("• QR decomposition is robust for least squares problems")
    print("• Cholesky decomposition is fastest for positive definite systems")
    print("• SVD provides optimal low-rank approximations")
    print("• Eigendecomposition reveals matrix spectral properties")
def demonstrate_lu_decomposition():
    """Demonstrate LU decomposition and its applications."""
    print("LU Decomposition")
    print("=" * 20)
    # Create test matrix
    A = np.array([
        [2, 1, 1],
        [4, 3, 3],
        [8, 7, 9]
    ])
    print("Matrix A:")
    print(A)
    # Perform LU decomposition
    P, L, U = MatrixDecompositions.lu_decomposition(A)
    print("\nLU Decomposition: PA = LU")
    print("Permutation matrix P:")
    print(P)
    print("\nLower triangular L:")
    print(L)
    print("\nUpper triangular U:")
    print(U)
    # Verify decomposition
    PA = P @ A
    LU = L @ U
    print(f"\nVerification: ||PA - LU|| = {np.linalg.norm(PA - LU):.2e}")
    # Application: Solve multiple systems with same A
    print("\n1. Solving Multiple Systems Efficiently")
    print("-" * 38)
    # Multiple right-hand sides
    b1 = np.array([4, 10, 24])
    b2 = np.array([1, 2, 3])
    b3 = np.array([0, 1, 0])
    print("Solving Ax = b for multiple b vectors:")
    print(f"b1 = {b1}")
    print(f"b2 = {b2}")
    print(f"b3 = {b3}")
    # Solve using LU decomposition (efficient for multiple RHS)
    solutions = []
    for i, b in enumerate([b1, b2, b3], 1):
        result = DirectSolvers.lu_solve(A, b)
        if result.success:
            solutions.append(result.solution)
            print(f"Solution x{i} = {result.solution}")
            # Verify
            residual = A @ result.solution - b
            print(f"  Residual norm: {np.linalg.norm(residual):.2e}")
    # Application: Matrix inversion using LU
    print("\n2. Matrix Inversion via LU")
    print("-" * 25)
    # Solve A * X = I to find A^(-1)
    n = A.shape[0]
    I = np.eye(n)
    A_inv = np.zeros_like(A, dtype=float)
    for i in range(n):
        result = DirectSolvers.lu_solve(A, I[:, i])
        if result.success:
            A_inv[:, i] = result.solution
    print("Matrix inverse A^(-1):")
    print(A_inv)
    # Verify inversion
    product = A @ A_inv
    print(f"\nVerification: ||A * A^(-1) - I|| = {np.linalg.norm(product - I):.2e}")
    # Determinant from LU
    print("\n3. Determinant from LU")
    print("-" * 20)
    det_from_lu = np.prod(np.diag(U)) * np.linalg.det(P)
    det_direct = np.linalg.det(A)
    print(f"Determinant from LU: {det_from_lu:.6f}")
    print(f"Direct determinant:  {det_direct:.6f}")
    print(f"Difference: {abs(det_from_lu - det_direct):.2e}")
def demonstrate_qr_decomposition():
    """Demonstrate QR decomposition and least squares."""
    print("\n\nQR Decomposition")
    print("=" * 20)
    # Create overdetermined system (more equations than unknowns)
    np.random.seed(42)
    m, n = 6, 4
    A = np.random.randn(m, n)
    print(f"Matrix A ({m}x{n}):")
    print(A)
    # QR decomposition
    Q, R = MatrixDecompositions.qr_decomposition(A, mode='economic')
    print(f"\nEconomic QR: A = QR")
    print(f"Q matrix ({Q.shape[0]}x{Q.shape[1]}):")
    print(Q)
    print(f"\nR matrix ({R.shape[0]}x{R.shape[1]}):")
    print(R)
    # Verify decomposition
    QR = Q @ R
    print(f"\nVerification: ||A - QR|| = {np.linalg.norm(A - QR):.2e}")
    # Verify Q orthogonality
    QtQ = Q.T @ Q
    I_small = np.eye(Q.shape[1])
    print(f"Q orthogonality: ||Q^T Q - I|| = {np.linalg.norm(QtQ - I_small):.2e}")
    # Application: Least squares problem
    print("\n1. Least Squares Line Fitting")
    print("-" * 28)
    # Generate noisy data for line fitting
    x_data = np.linspace(0, 5, 10)
    true_slope = 2.5
    true_intercept = 1.2
    noise = 0.3 * np.random.randn(len(x_data))
    y_data = true_slope * x_data + true_intercept + noise
    # Set up least squares problem: find line y = mx + c
    # Design matrix: [1, x1; 1, x2; ...; 1, xn]
    A_fit = np.column_stack([np.ones(len(x_data)), x_data])
    print(f"True line: y = {true_slope}x + {true_intercept}")
    print(f"Data points with noise added")
    # Solve using QR
    result = DirectSolvers.qr_solve(A_fit, y_data)
    if result.success:
        intercept_fit, slope_fit = result.solution
        print(f"Fitted line: y = {slope_fit:.3f}x + {intercept_fit:.3f}")
        print(f"Residual norm: {result.residual_norm:.3f}")
        # Plot results
        plt.figure(figsize=(10, 6))
        plt.subplot(1, 2, 1)
        plt.scatter(x_data, y_data, color='#003262', s=50, label='Noisy data', zorder=5)
        x_line = np.linspace(-0.5, 5.5, 100)
        y_true = true_slope * x_line + true_intercept
        y_fit = slope_fit * x_line + intercept_fit
        plt.plot(x_line, y_true, '--', color='red', linewidth=2, label='True line')
        plt.plot(x_line, y_fit, color='#FDB515', linewidth=2, label='Fitted line')
        # Show residuals
        y_predicted = A_fit @ result.solution
        for i in range(len(x_data)):
            plt.plot([x_data[i], x_data[i]], [y_data[i], y_predicted[i]], 'gray', alpha=0.7)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('QR-based Least Squares Fitting')
        plt.legend()
        plt.grid(True, alpha=0.3)
        # Condition number analysis
        plt.subplot(1, 2, 2)
        cond_numbers = []
        ranks = []
        for k in range(2, min(10, len(x_data))):
            A_sub = A_fit[:k, :]
            cond_numbers.append(np.linalg.cond(A_sub))
            ranks.append(np.linalg.matrix_rank(A_sub))
        plt.semilogy(range(2, 2 + len(cond_numbers)), cond_numbers, 'o-', color='#003262')
        plt.xlabel('Number of data points')
        plt.ylabel('Condition number')
        plt.title('Condition Number vs Data Size')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
def demonstrate_cholesky_decomposition():
    """Demonstrate Cholesky decomposition for positive definite matrices."""
    print("\n\nCholesky Decomposition")
    print("=" * 25)
    # Create positive definite matrix
    np.random.seed(42)
    A_base = np.random.randn(4, 4)
    A = A_base.T @ A_base + 0.1 * np.eye(4)  # Ensure positive definite
    print("Positive definite matrix A:")
    print(A)
    # Verify positive definiteness
    eigenvals = np.linalg.eigvals(A)
    print(f"\nEigenvalues: {eigenvals}")
    print(f"All positive: {np.all(eigenvals > 0)}")
    # Cholesky decomposition
    L = MatrixDecompositions.cholesky_decomposition(A, lower=True)
    print("\nCholesky factor L (lower triangular):")
    print(L)
    # Verify decomposition
    LLT = L @ L.T
    print(f"\nVerification: ||A - LL^T|| = {np.linalg.norm(A - LLT):.2e}")
    # Application: Efficient solving for positive definite systems
    print("\n1. Efficient System Solving")
    print("-" * 26)
    b = np.array([1, 2, 3, 4])
    print(f"Right-hand side: {b}")
    # Solve using Cholesky
    result_chol = DirectSolvers.cholesky_solve(A, b)
    # Compare with LU
    result_lu = DirectSolvers.lu_solve(A, b)
    if result_chol.success and result_lu.success:
        print(f"Cholesky solution: {result_chol.solution}")
        print(f"LU solution:       {result_lu.solution}")
        print(f"Difference: {np.linalg.norm(result_chol.solution - result_lu.solution):.2e}")
        print(f"\nCholesky residual: {result_chol.residual_norm:.2e}")
        print(f"LU residual:       {result_lu.residual_norm:.2e}")
    # Application: Generating correlated random variables
    print("\n2. Generating Correlated Random Variables")
    print("-" * 41)
    # Desired covariance matrix
    desired_corr = np.array([
        [1.0, 0.7, 0.3],
        [0.7, 1.0, 0.5],
        [0.3, 0.5, 1.0]
    ])
    print("Desired correlation matrix:")
    print(desired_corr)
    # Cholesky factor of correlation matrix
    L_corr = MatrixDecompositions.cholesky_decomposition(desired_corr, lower=True)
    # Generate uncorrelated samples and transform
    n_samples = 1000
    uncorr_samples = np.random.randn(3, n_samples)
    corr_samples = L_corr @ uncorr_samples
    # Compute sample correlation
    sample_corr = np.corrcoef(corr_samples)
    print(f"\nSample correlation matrix ({n_samples} samples):")
    print(sample_corr)
    print(f"Error: {np.linalg.norm(sample_corr - desired_corr):.3f}")
    # Visualize
    fig = plt.figure(figsize=(12, 4))
    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.hist(corr_samples[i, :], bins=30, alpha=0.7, color='#003262', edgecolor='black')
        plt.title(f'Variable {i+1}')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
def demonstrate_svd_applications():
    """Demonstrate SVD and its applications."""
    print("\n\nSingular Value Decomposition (SVD)")
    print("=" * 35)
    # Create test matrix
    np.random.seed(42)
    A = np.random.randn(5, 3)
    print("Matrix A (5x3):")
    print(A)
    # SVD decomposition
    U, s, Vt = MatrixDecompositions.svd(A, full_matrices=False)
    print(f"\nSVD: A = U @ diag(s) @ V^T")
    print(f"U matrix ({U.shape[0]}x{U.shape[1]}):")
    print(U)
    print(f"\nSingular values: {s}")
    print(f"\nV^T matrix ({Vt.shape[0]}x{Vt.shape[1]}):")
    print(Vt)
    # Verify decomposition
    A_reconstructed = U @ np.diag(s) @ Vt
    print(f"\nVerification: ||A - U*S*V^T|| = {np.linalg.norm(A - A_reconstructed):.2e}")
    # Verify orthogonality
    print(f"U orthogonality: ||U^T*U - I|| = {np.linalg.norm(U.T @ U - np.eye(U.shape[1])):.2e}")
    print(f"V orthogonality: ||V*V^T - I|| = {np.linalg.norm(Vt @ Vt.T - np.eye(Vt.shape[0])):.2e}")
    # Application 1: Low-rank approximation
    print("\n1. Low-rank Approximation")
    print("-" * 24)
    # Create a matrix with known low-rank structure
    rank_2_matrix = np.outer(np.array([1, 2, 3, 4, 5]), np.array([1, 1, 2])) + \
                    np.outer(np.array([2, 1, 1, 2, 1]), np.array([1, 2, 1]))
    # Add small amount of noise
    noisy_matrix = rank_2_matrix + 0.1 * np.random.randn(*rank_2_matrix.shape)
    print("Original rank-2 matrix + noise:")
    print(noisy_matrix)
    # SVD for denoising
    U_noise, s_noise, Vt_noise = MatrixDecompositions.svd(noisy_matrix, full_matrices=False)
    print(f"\nSingular values: {s_noise}")
    # Reconstruct using only largest singular values
    k = 2  # Keep only 2 components
    A_denoised = U_noise[:, :k] @ np.diag(s_noise[:k]) @ Vt_noise[:k, :]
    print(f"\nDenoised matrix (rank {k}):")
    print(A_denoised)
    print(f"Original rank: {np.linalg.matrix_rank(noisy_matrix)}")
    print(f"Denoised rank: {np.linalg.matrix_rank(A_denoised)}")
    print(f"Frobenius error: {np.linalg.norm(rank_2_matrix - A_denoised, 'fro'):.3f}")
    # Application 2: Principal Component Analysis (PCA)
    print("\n2. Principal Component Analysis")
    print("-" * 30)
    # Generate 2D data with correlation
    n_points = 200
    np.random.seed(42)
    # Original data in 2D
    theta = np.pi / 6  # 30 degree rotation
    rotation = np.array([[np.cos(theta), -np.sin(theta)],
                        [np.sin(theta), np.cos(theta)]])
    # Uncorrelated data
    data_uncorr = np.random.randn(2, n_points)
    data_uncorr[0, :] *= 3  # Different variances
    data_uncorr[1, :] *= 1
    # Rotate to create correlation
    data = rotation @ data_uncorr
    # Center the data
    data_centered = data - np.mean(data, axis=1, keepdims=True)
    print(f"Data shape: {data_centered.shape}")
    print(f"Data covariance matrix:")
    cov_matrix = np.cov(data_centered)
    print(cov_matrix)
    # PCA using SVD
    U_pca, s_pca, Vt_pca = MatrixDecompositions.svd(data_centered, full_matrices=False)
    # Principal components are columns of Vt.T
    principal_components = Vt_pca.T
    explained_variance = s_pca**2 / (n_points - 1)
    print(f"\nPrincipal components:")
    print(principal_components)
    print(f"Explained variance: {explained_variance}")
    print(f"Explained variance ratio: {explained_variance / np.sum(explained_variance)}")
    # Visualize PCA
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(data_centered[0, :], data_centered[1, :], alpha=0.6, color='#003262')
    # Plot principal components
    center = np.mean(data_centered, axis=1)
    scale = 2 * np.sqrt(explained_variance)
    for i in range(2):
        direction = principal_components[:, i] * scale[i]
        plt.arrow(center[0], center[1], direction[0], direction[1],
                 head_width=0.2, head_length=0.3, fc='#FDB515', ec='#FDB515', linewidth=2)
        plt.text(center[0] + direction[0]*1.1, center[1] + direction[1]*1.1,
                f'PC{i+1}', fontsize=12, color='#FDB515', weight='bold')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Original Data with Principal Components')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    # Project data onto principal components
    projected_data = Vt_pca @ data_centered
    plt.subplot(1, 2, 2)
    plt.scatter(projected_data[0, :], projected_data[1, :], alpha=0.6, color='#003262')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.title('Data in Principal Component Space')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()
def demonstrate_eigenvalue_decomposition():
    """Demonstrate eigenvalue decomposition and applications."""
    print("\n\nEigenvalue Decomposition")
    print("=" * 27)
    # Create symmetric matrix for real eigenvalues
    np.random.seed(42)
    A_base = np.random.randn(4, 4)
    A = A_base + A_base.T  # Make symmetric
    print("Symmetric matrix A:")
    print(A)
    # Eigenvalue decomposition
    eigenvals, eigenvecs = MatrixDecompositions.symmetric_eigendecomposition(A)
    print(f"\nEigenvalues (ascending order): {eigenvals}")
    print(f"\nEigenvectors:")
    print(eigenvecs)
    # Verify decomposition: A * v = λ * v
    print("\nVerification (A*v = λ*v for each eigenpair):")
    for i in range(len(eigenvals)):
        v = eigenvecs[:, i]
        Av = A @ v
        lambda_v = eigenvals[i] * v
        error = np.linalg.norm(Av - lambda_v)
        print(f"  Eigenpair {i+1}: ||A*v - λ*v|| = {error:.2e}")
    # Verify orthogonality of eigenvectors
    VtV = eigenvecs.T @ eigenvecs
    I = np.eye(4)
    print(f"\nEigenvector orthogonality: ||V^T*V - I|| = {np.linalg.norm(VtV - I):.2e}")
    # Application: Matrix powers using eigendecomposition
    print("\n1. Efficient Matrix Powers")
    print("-" * 26)
    # Compute A^10 using eigendecomposition
    n = 10
    A_power_eigen = eigenvecs @ np.diag(eigenvals**n) @ eigenvecs.T
    A_power_direct = np.linalg.matrix_power(A, n)
    print(f"A^{n} using eigendecomposition:")
    print(A_power_eigen)
    print(f"\nA^{n} using direct computation:")
    print(A_power_direct)
    print(f"Difference: {np.linalg.norm(A_power_eigen - A_power_direct):.2e}")
    # Application: Quadratic forms and definiteness
    print("\n2. Quadratic Forms Analysis")
    print("-" * 26)
    print(f"Matrix eigenvalues: {eigenvals}")
    if np.all(eigenvals > 0):
        definiteness = "positive definite"
    elif np.all(eigenvals >= 0):
        definiteness = "positive semidefinite"
    elif np.all(eigenvals < 0):
        definiteness = "negative definite"
    elif np.all(eigenvals <= 0):
        definiteness = "negative semidefinite"
    else:
        definiteness = "indefinite"
    print(f"Matrix is: {definiteness}")
    # Visualize quadratic form x^T A x = c (for 2D case)
    if A.shape[0] >= 2:
        A_2d = A[:2, :2]  # Take 2x2 submatrix
        eigenvals_2d, eigenvecs_2d = np.linalg.eigh(A_2d)
        print(f"\n2D submatrix eigenvalues: {eigenvals_2d}")
        # Create grid for contour plot
        x = np.linspace(-3, 3, 100)
        y = np.linspace(-3, 3, 100)
        X, Y = np.meshgrid(x, y)
        # Compute quadratic form
        Z = np.zeros_like(X)
        for i in range(len(x)):
            for j in range(len(y)):
                vec = np.array([X[i, j], Y[i, j]])
                Z[i, j] = vec.T @ A_2d @ vec
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        contours = plt.contour(X, Y, Z, levels=20, colors='#003262', alpha=0.7)
        plt.clabel(contours, inline=True, fontsize=8)
        # Plot eigenvector directions
        center = [0, 0]
        for i in range(2):
            direction = eigenvecs_2d[:, i] * 2
            plt.arrow(center[0], center[1], direction[0], direction[1],
                     head_width=0.1, head_length=0.15, fc='#FDB515', ec='#FDB515', linewidth=2)
            plt.text(direction[0]*1.2, direction[1]*1.2, f'λ={eigenvals_2d[i]:.2f}',
                    fontsize=10, color='#FDB515', weight='bold')
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.title('Quadratic Form: x^T A x = c')
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        # Eigenvalue visualization
        plt.subplot(1, 2, 2)
        plt.bar(range(1, len(eigenvals)+1), eigenvals, color='#003262', alpha=0.7)
        plt.axhline(y=0, color='red', linestyle='--', alpha=0.7)
        plt.xlabel('Eigenvalue Index')
        plt.ylabel('Eigenvalue')
        plt.title('Eigenvalue Spectrum')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
def image_compression_example():
    """Demonstrate image compression using SVD."""
    print("\n\nPractical Application: Image Compression with SVD")
    print("=" * 52)
    # Create a simple synthetic "image" (2D pattern)
    x = np.linspace(-2, 2, 64)
    y = np.linspace(-2, 2, 64)
    X, Y = np.meshgrid(x, y)
    # Create interesting pattern
    image = np.exp(-(X**2 + Y**2)/2) * np.cos(4*X) * np.sin(3*Y) + \
            0.3 * np.exp(-((X-1)**2 + (Y+0.5)**2)/0.5)
    print(f"Original image size: {image.shape}")
    print(f"Original rank: {np.linalg.matrix_rank(image)}")
    # SVD compression
    U, s, Vt = MatrixDecompositions.svd(image, full_matrices=False)
    print(f"Number of singular values: {len(s)}")
    print(f"Largest singular values: {s[:10]}")
    # Compress with different ranks
    ranks = [1, 5, 10, 20, len(s)]
    plt.figure(figsize=(15, 10))
    for i, k in enumerate(ranks):
        if k <= len(s):
            # Reconstruct using first k singular values
            image_compressed = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]
            # Compute compression ratio and error
            original_elements = image.size
            compressed_elements = k * (U.shape[0] + Vt.shape[1]) + k  # U[:,:k] + Vt[:k,:] + s[:k]
            compression_ratio = original_elements / compressed_elements
            relative_error = np.linalg.norm(image - image_compressed, 'fro') / np.linalg.norm(image, 'fro')
            plt.subplot(2, 3, i+1)
            plt.imshow(image_compressed, cmap='viridis', extent=[-2, 2, -2, 2])
            plt.colorbar()
            if k == len(s):
                title = f'Original\n(rank {k})'
            else:
                title = f'Rank {k}\nCompression: {compression_ratio:.1f}x\nError: {relative_error:.3f}'
            plt.title(title)
            plt.xlabel('x')
            plt.ylabel('y')
    # Singular value decay
    plt.subplot(2, 3, 6)
    plt.semilogy(range(1, len(s)+1), s, 'o-', color='#003262', markersize=4)
    plt.xlabel('Singular Value Index')
    plt.ylabel('Singular Value')
    plt.title('Singular Value Decay')
    plt.grid(True, alpha=0.3)
    # Mark compression points
    for k in ranks[:-1]:
        if k <= len(s):
            plt.axvline(x=k, color='#FDB515', linestyle='--', alpha=0.7)
            plt.text(k+1, s[k-1], f'{k}', color='#FDB515', fontweight='bold')
    plt.tight_layout()
    plt.show()
    # Compression analysis
    print("\nCompression Analysis:")
    print("Rank\tComp.Ratio\tRel.Error\tStorage%")
    print("-" * 40)
    for k in ranks[:-1]:
        if k <= len(s):
            image_compressed = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]
            original_elements = image.size
            compressed_elements = k * (U.shape[0] + Vt.shape[1]) + k
            compression_ratio = original_elements / compressed_elements
            storage_percent = (compressed_elements / original_elements) * 100
            relative_error = np.linalg.norm(image - image_compressed, 'fro') / np.linalg.norm(image, 'fro')
            print(f"{k:4d}\t{compression_ratio:8.1f}\t{relative_error:8.3f}\t{storage_percent:7.1f}%")
if __name__ == "__main__":
    main()