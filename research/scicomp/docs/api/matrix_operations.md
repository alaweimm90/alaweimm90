# matrix_operations
**Module:** `Python/Linear_Algebra/core/matrix_operations.py`
## Overview
Matrix Operations for Scientific Computing
Comprehensive matrix operations including basic arithmetic, decompositions,
eigenvalue problems, and specialized algorithms for scientific applications.
## Functions
### `create_test_matrices()`
Create a set of test matrices for validation.
**Source:** [Line 603](Python/Linear_Algebra/core/matrix_operations.py#L603)
## Classes
### `MatrixOperations`
Core matrix operations for scientific computing.
Features:
- Basic matrix arithmetic with error checking
- Matrix decompositions (LU, QR, Cholesky, SVD)
- Eigenvalue and eigenvector computation
- Matrix norms and condition numbers
- Specialized algorithms for scientific computing
#### Methods
##### `validate_matrix(matrix, name)`
Validate matrix input and convert to numpy array.
**Source:** [Line 28](Python/Linear_Algebra/core/matrix_operations.py#L28)
##### `validate_vector(vector, name)`
Validate vector input and convert to numpy array.
**Source:** [Line 42](Python/Linear_Algebra/core/matrix_operations.py#L42)
##### `matrix_multiply(A, B, check_compatibility)`
Matrix multiplication with dimension checking.
Parameters:
A: First matrix
B: Second matrix or vector
check_compatibility: Whether to check dimension compatibility
Returns:
Matrix product A @ B
**Source:** [Line 55](Python/Linear_Algebra/core/matrix_operations.py#L55)
##### `matrix_add(A, B)`
Matrix addition with dimension checking.
**Source:** [Line 82](Python/Linear_Algebra/core/matrix_operations.py#L82)
##### `matrix_subtract(A, B)`
Matrix subtraction with dimension checking.
**Source:** [Line 93](Python/Linear_Algebra/core/matrix_operations.py#L93)
##### `matrix_power(A, n)`
Matrix power A^n using repeated squaring.
Parameters:
A: Square matrix
n: Non-negative integer power
Returns:
A^n
**Source:** [Line 104](Python/Linear_Algebra/core/matrix_operations.py#L104)
##### `transpose(A)`
Matrix transpose.
**Source:** [Line 129](Python/Linear_Algebra/core/matrix_operations.py#L129)
##### `conjugate_transpose(A)`
Conjugate transpose (Hermitian transpose).
**Source:** [Line 135](Python/Linear_Algebra/core/matrix_operations.py#L135)
##### `trace(A)`
Matrix trace (sum of diagonal elements).
**Source:** [Line 141](Python/Linear_Algebra/core/matrix_operations.py#L141)
##### `determinant(A)`
Matrix determinant.
**Source:** [Line 151](Python/Linear_Algebra/core/matrix_operations.py#L151)
##### `rank(A, tolerance)`
Matrix rank using SVD.
Parameters:
A: Input matrix
tolerance: Tolerance for rank determination
Returns:
Matrix rank
**Source:** [Line 161](Python/Linear_Algebra/core/matrix_operations.py#L161)
##### `condition_number(A, p)`
Matrix condition number.
Parameters:
A: Input matrix
p: Order of the norm (None, 1, -1, 2, -2, inf, -inf, 'fro')
Returns:
Condition number
**Source:** [Line 182](Python/Linear_Algebra/core/matrix_operations.py#L182)
##### `frobenius_norm(A)`
Frobenius norm of matrix.
**Source:** [Line 197](Python/Linear_Algebra/core/matrix_operations.py#L197)
##### `spectral_norm(A)`
Spectral norm (largest singular value).
**Source:** [Line 203](Python/Linear_Algebra/core/matrix_operations.py#L203)
##### `nuclear_norm(A)`
Nuclear norm (sum of singular values).
**Source:** [Line 209](Python/Linear_Algebra/core/matrix_operations.py#L209)
**Class Source:** [Line 15](Python/Linear_Algebra/core/matrix_operations.py#L15)
### `MatrixDecompositions`
Matrix decomposition algorithms.
Features:
- LU decomposition with partial pivoting
- QR decomposition (Householder and Givens)
- Cholesky decomposition
- Singular Value Decomposition (SVD)
- Eigenvalue decomposition
- Schur decomposition
#### Methods
##### `lu_decomposition(A, permute_l)`
LU decomposition with partial pivoting.
Parameters:
A: Input matrix
permute_l: Whether to apply permutations to L
Returns:
Tuple of (P, L, U) where PA = LU
**Source:** [Line 229](Python/Linear_Algebra/core/matrix_operations.py#L229)
##### `qr_decomposition(A, mode)`
QR decomposition using Householder reflections.
Parameters:
A: Input matrix
mode: 'full' or 'economic'
Returns:
Tuple of (Q, R) where A = QR
**Source:** [Line 249](Python/Linear_Algebra/core/matrix_operations.py#L249)
##### `cholesky_decomposition(A, lower)`
Cholesky decomposition for positive definite matrices.
Parameters:
A: Positive definite matrix
lower: Whether to return lower triangular factor
Returns:
Cholesky factor L (if lower=True) or U (if lower=False)
**Source:** [Line 273](Python/Linear_Algebra/core/matrix_operations.py#L273)
##### `svd(A, full_matrices, compute_uv)`
Singular Value Decomposition.
Parameters:
A: Input matrix
full_matrices: Whether to compute full U and Vt matrices
compute_uv: Whether to compute U and Vt
Returns:
If compute_uv=True: (U, s, Vt) where A = U @ diag(s) @ Vt
If compute_uv=False: s (singular values only)
**Source:** [Line 300](Python/Linear_Algebra/core/matrix_operations.py#L300)
##### `eigendecomposition(A, right, left)`
Eigenvalue decomposition for general matrices.
Parameters:
A: Square matrix
right: Whether to compute right eigenvectors
left: Whether to compute left eigenvectors
Returns:
Tuple of (eigenvalues, eigenvectors)
**Source:** [Line 319](Python/Linear_Algebra/core/matrix_operations.py#L319)
##### `symmetric_eigendecomposition(A, subset_by_index, subset_by_value)`
Eigenvalue decomposition for symmetric/Hermitian matrices.
Parameters:
A: Symmetric/Hermitian matrix
subset_by_index: Tuple (il, iu) to compute eigenvalues il through iu
subset_by_value: Tuple (vl, vu) to compute eigenvalues in range [vl, vu]
Returns:
Tuple of (eigenvalues, eigenvectors) sorted in ascending order
**Source:** [Line 352](Python/Linear_Algebra/core/matrix_operations.py#L352)
##### `schur_decomposition(A, output)`
Schur decomposition.
Parameters:
A: Square matrix
output: 'real' or 'complex'
Returns:
Tuple of (T, Z) where A = Z @ T @ Z.H
**Source:** [Line 384](Python/Linear_Algebra/core/matrix_operations.py#L384)
**Class Source:** [Line 215](Python/Linear_Algebra/core/matrix_operations.py#L215)
### `SpecialMatrices`
Special matrix types and properties.
Features:
- Matrix property checking (symmetric, orthogonal, etc.)
- Special matrix generation
- Matrix transformations
#### Methods
##### `is_symmetric(A, tolerance)`
Check if matrix is symmetric.
**Source:** [Line 415](Python/Linear_Algebra/core/matrix_operations.py#L415)
##### `is_hermitian(A, tolerance)`
Check if matrix is Hermitian.
**Source:** [Line 425](Python/Linear_Algebra/core/matrix_operations.py#L425)
##### `is_orthogonal(A, tolerance)`
Check if matrix is orthogonal (A.T @ A = I).
**Source:** [Line 435](Python/Linear_Algebra/core/matrix_operations.py#L435)
##### `is_unitary(A, tolerance)`
Check if matrix is unitary (A.H @ A = I).
**Source:** [Line 448](Python/Linear_Algebra/core/matrix_operations.py#L448)
##### `is_positive_definite(A)`
Check if matrix is positive definite.
**Source:** [Line 461](Python/Linear_Algebra/core/matrix_operations.py#L461)
##### `is_positive_semidefinite(A)`
Check if matrix is positive semidefinite.
**Source:** [Line 478](Python/Linear_Algebra/core/matrix_operations.py#L478)
##### `make_symmetric(A)`
Make matrix symmetric by (A + A.T) / 2.
**Source:** [Line 492](Python/Linear_Algebra/core/matrix_operations.py#L492)
##### `make_hermitian(A)`
Make matrix Hermitian by (A + A.H) / 2.
**Source:** [Line 502](Python/Linear_Algebra/core/matrix_operations.py#L502)
##### `gram_schmidt(A, normalize)`
Gram-Schmidt orthogonalization of matrix columns.
Parameters:
A: Input matrix
normalize: Whether to normalize the orthogonal vectors
Returns:
Matrix with orthogonal (or orthonormal) columns
**Source:** [Line 512](Python/Linear_Algebra/core/matrix_operations.py#L512)
##### `householder_reflector(x)`
Compute Householder reflector matrix.
Parameters:
x: Input vector
Returns:
Householder matrix H such that Hx = ||x||e_1
**Source:** [Line 549](Python/Linear_Algebra/core/matrix_operations.py#L549)
##### `givens_rotation(a, b)`
Compute Givens rotation matrix.
Parameters:
a, b: Values to rotate
Returns:
Tuple of (c, s, G) where G is 2x2 Givens matrix
**Source:** [Line 576](Python/Linear_Algebra/core/matrix_operations.py#L576)
**Class Source:** [Line 404](Python/Linear_Algebra/core/matrix_operations.py#L404)