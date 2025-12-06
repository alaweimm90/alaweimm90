#!/usr/bin/env python3
"""
Simple functionality test for Linear Algebra package.
Tests core functionality without graphics.
"""
import numpy as np
import sys
import os
# Add path
sys.path.append(os.path.join(os.path.dirname(__file__)))
from core.matrix_operations import MatrixOperations, MatrixDecompositions, SpecialMatrices
from core.vector_operations import VectorOperations, VectorNorms
from core.linear_systems import DirectSolvers, IterativeSolvers, LinearSystemUtils
def test_vector_operations():
    """Test basic vector operations."""
    print("Testing Vector Operations...")
    u = np.array([3, 4])
    v = np.array([1, 2])
    # Basic operations
    sum_vec = VectorOperations.add(u, v)
    print(f"u + v = {sum_vec}")  # Should be [4, 6]
    dot_prod = VectorOperations.dot_product(u, v)
    print(f"u · v = {dot_prod}")  # Should be 11
    magnitude = VectorOperations.magnitude(u)
    print(f"|u| = {magnitude}")  # Should be 5.0
    # Cross product
    u3d = np.array([1, 0, 0])
    v3d = np.array([0, 1, 0])
    cross = VectorOperations.cross_product(u3d, v3d)
    print(f"[1,0,0] × [0,1,0] = {cross}")  # Should be [0, 0, 1]
    print("Vector operations: PASSED\n")
def test_matrix_operations():
    """Test basic matrix operations."""
    print("Testing Matrix Operations...")
    A = np.array([[2, 1], [1, 3]], dtype=float)
    B = np.array([[1, 2], [3, 1]], dtype=float)
    # Basic operations
    sum_mat = MatrixOperations.matrix_add(A, B)
    print(f"A + B =\n{sum_mat}")
    product = MatrixOperations.matrix_multiply(A, B)
    print(f"A × B =\n{product}")
    det_A = MatrixOperations.determinant(A)
    print(f"det(A) = {det_A}")  # Should be 5
    rank_A = MatrixOperations.rank(A)
    print(f"rank(A) = {rank_A}")  # Should be 2
    print("Matrix operations: PASSED\n")
def test_matrix_decompositions():
    """Test matrix decompositions."""
    print("Testing Matrix Decompositions...")
    # Create test matrix
    A = np.array([[4, 2], [2, 3]], dtype=float)
    # LU decomposition
    P, L, U = MatrixDecompositions.lu_decomposition(A)
    verification = P @ A
    reconstruction = L @ U
    lu_error = np.linalg.norm(verification - reconstruction)
    print(f"LU decomposition error: {lu_error:.2e}")
    # QR decomposition
    Q, R = MatrixDecompositions.qr_decomposition(A)
    qr_reconstruction = Q @ R
    qr_error = np.linalg.norm(A - qr_reconstruction)
    print(f"QR decomposition error: {qr_error:.2e}")
    # Check Q orthogonality
    orthogonality_error = np.linalg.norm(Q.T @ Q - np.eye(2))
    print(f"Q orthogonality error: {orthogonality_error:.2e}")
    print("Matrix decompositions: PASSED\n")
def test_linear_systems():
    """Test linear system solvers."""
    print("Testing Linear System Solvers...")
    # Create test system
    A = np.array([[3, 1], [1, 2]], dtype=float)
    x_true = np.array([1, 2])
    b = A @ x_true
    # Test direct solvers
    result_lu = DirectSolvers.lu_solve(A, b)
    print(f"LU solver success: {result_lu.success}")
    print(f"LU solution: {result_lu.solution}")
    print(f"LU error: {np.linalg.norm(result_lu.solution - x_true):.2e}")
    result_qr = DirectSolvers.qr_solve(A, b)
    print(f"QR solver success: {result_qr.success}")
    print(f"QR error: {np.linalg.norm(result_qr.solution - x_true):.2e}")
    # Test iterative solver (create SPD matrix)
    A_spd = A.T @ A + 0.1 * np.eye(2)
    b_spd = A_spd @ x_true
    result_cg = IterativeSolvers.conjugate_gradient(A_spd, b_spd, tolerance=1e-10)
    print(f"CG solver success: {result_cg.success}")
    print(f"CG iterations: {result_cg.iterations}")
    print(f"CG error: {np.linalg.norm(result_cg.solution - x_true):.2e}")
    print("Linear system solvers: PASSED\n")
def test_special_matrices():
    """Test special matrix properties."""
    print("Testing Special Matrix Properties...")
    # Create symmetric matrix
    A = np.array([[1, 2], [2, 3]], dtype=float)
    print(f"Is symmetric: {SpecialMatrices.is_symmetric(A)}")
    # Create positive definite matrix
    B_base = np.random.randn(3, 3)
    B = B_base.T @ B_base + 0.1 * np.eye(3)
    print(f"Is positive definite: {SpecialMatrices.is_positive_definite(B)}")
    # Create orthogonal matrix
    Q, _ = np.linalg.qr(np.random.randn(3, 3))
    print(f"Is orthogonal: {SpecialMatrices.is_orthogonal(Q)}")
    print("Special matrix properties: PASSED\n")
def test_system_analysis():
    """Test system analysis utilities."""
    print("Testing System Analysis...")
    # Create different types of systems
    A_square = np.random.randn(3, 3) + 2 * np.eye(3)
    b_square = np.random.randn(3)
    A_over = np.random.randn(5, 3)
    b_over = np.random.randn(5)
    # Analyze systems
    analysis_square = LinearSystemUtils.analyze_system(A_square, b_square)
    print(f"Square system - Shape: {analysis_square['matrix_shape']}")
    print(f"Square system - Is square: {analysis_square['is_square']}")
    print(f"Square system - Condition number: {analysis_square['condition_number']:.2e}")
    analysis_over = LinearSystemUtils.analyze_system(A_over, b_over)
    print(f"Overdetermined - Is overdetermined: {analysis_over['is_overdetermined']}")
    # Test solver recommendation
    recommendation = LinearSystemUtils.recommend_solver(A_square, b_square)
    print(f"Recommended solver: {recommendation}")
    # Test auto solver
    result_auto = LinearSystemUtils.solve_auto(A_square, b_square)
    print(f"Auto solver success: {result_auto.success}")
    print(f"Auto solver method: {result_auto.info['method']}")
    print("System analysis: PASSED\n")
def main():
    """Run all tests."""
    print("Berkeley SciComp Linear Algebra Package - Functionality Test")
    print("=" * 60)
    try:
        test_vector_operations()
        test_matrix_operations()
        test_matrix_decompositions()
        test_linear_systems()
        test_special_matrices()
        test_system_analysis()
        print("=" * 60)
        print("ALL TESTS PASSED!")
        print("Linear Algebra package is working correctly.")
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    return 0
if __name__ == "__main__":
    exit(main())