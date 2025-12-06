# linear_systems
**Module:** `Python/Linear_Algebra/core/linear_systems.py`
## Overview
Linear Systems Solvers for Scientific Computing
Comprehensive algorithms for solving linear systems Ax = b including
direct methods, iterative methods, and specialized solvers.
## Functions
### `create_test_systems()`
Create test linear systems for validation.
**Source:** [Line 795](Python/Linear_Algebra/core/linear_systems.py#L795)
## Classes
### `SolverResult`
Result container for linear system solvers.
**Class Source:** [Line 20](Python/Linear_Algebra/core/linear_systems.py#L20)
### `DirectSolvers`
Direct methods for solving linear systems.
Features:
- LU decomposition with partial pivoting
- Cholesky decomposition for symmetric positive definite systems
- QR decomposition for overdetermined systems
- SVD for rank-deficient systems
- Specialized algorithms for structured matrices
#### Methods
##### `lu_solve(A, b, check_finite)`
Solve Ax = b using LU decomposition with partial pivoting.
Parameters:
A: Coefficient matrix
b: Right-hand side vector
check_finite: Whether to check for finite values
Returns:
SolverResult with solution and metadata
**Source:** [Line 42](Python/Linear_Algebra/core/linear_systems.py#L42)
##### `cholesky_solve(A, b, lower)`
Solve Ax = b using Cholesky decomposition for symmetric positive definite A.
Parameters:
A: Symmetric positive definite matrix
b: Right-hand side vector
lower: Whether to use lower triangular factor
Returns:
SolverResult with solution and metadata
**Source:** [Line 89](Python/Linear_Algebra/core/linear_systems.py#L89)
##### `qr_solve(A, b, mode)`
Solve Ax = b using QR decomposition (supports overdetermined systems).
Parameters:
A: Coefficient matrix (m x n, m >= n)
b: Right-hand side vector
mode: QR decomposition mode ('full' or 'economic')
Returns:
SolverResult with solution and metadata
**Source:** [Line 140](Python/Linear_Algebra/core/linear_systems.py#L140)
##### `svd_solve(A, b, rcond)`
Solve Ax = b using SVD (handles rank-deficient systems).
Parameters:
A: Coefficient matrix
b: Right-hand side vector
rcond: Relative condition number for rank determination
Returns:
SolverResult with solution and metadata
**Source:** [Line 196](Python/Linear_Algebra/core/linear_systems.py#L196)
##### `tridiagonal_solve(diag, upper, lower, b)`
Solve tridiagonal system using Thomas algorithm.
Parameters:
diag: Main diagonal
upper: Upper diagonal
lower: Lower diagonal
b: Right-hand side vector
Returns:
SolverResult with solution and metadata
**Source:** [Line 248](Python/Linear_Algebra/core/linear_systems.py#L248)
**Class Source:** [Line 29](Python/Linear_Algebra/core/linear_systems.py#L29)
### `IterativeSolvers`
Iterative methods for solving linear systems.
Features:
- Jacobi iteration
- Gauss-Seidel iteration
- Successive Over-Relaxation (SOR)
- Conjugate Gradient method
- GMRES method
- BiCGSTAB method
#### Methods
##### `jacobi(A, b, x0, max_iterations, tolerance)`
Solve Ax = b using Jacobi iteration.
Parameters:
A: Coefficient matrix
b: Right-hand side vector
x0: Initial guess
max_iterations: Maximum number of iterations
tolerance: Convergence tolerance
Returns:
SolverResult with solution and metadata
**Source:** [Line 339](Python/Linear_Algebra/core/linear_systems.py#L339)
##### `gauss_seidel(A, b, x0, max_iterations, tolerance)`
Solve Ax = b using Gauss-Seidel iteration.
Parameters:
A: Coefficient matrix
b: Right-hand side vector
x0: Initial guess
max_iterations: Maximum number of iterations
tolerance: Convergence tolerance
Returns:
SolverResult with solution and metadata
**Source:** [Line 410](Python/Linear_Algebra/core/linear_systems.py#L410)
##### `sor(A, b, omega, x0, max_iterations, tolerance)`
Solve Ax = b using Successive Over-Relaxation (SOR).
Parameters:
A: Coefficient matrix
b: Right-hand side vector
omega: Relaxation parameter (0 < omega < 2)
x0: Initial guess
max_iterations: Maximum number of iterations
tolerance: Convergence tolerance
Returns:
SolverResult with solution and metadata
**Source:** [Line 478](Python/Linear_Algebra/core/linear_systems.py#L478)
##### `conjugate_gradient(A, b, x0, max_iterations, tolerance, preconditioner)`
Solve Ax = b using Conjugate Gradient method (for symmetric positive definite A).
Parameters:
A: Symmetric positive definite matrix
b: Right-hand side vector
x0: Initial guess
max_iterations: Maximum number of iterations
tolerance: Convergence tolerance
preconditioner: Preconditioner matrix
Returns:
SolverResult with solution and metadata
**Source:** [Line 551](Python/Linear_Algebra/core/linear_systems.py#L551)
##### `gmres(A, b, x0, max_iterations, tolerance, restart)`
Solve Ax = b using GMRES method.
Parameters:
A: Coefficient matrix
b: Right-hand side vector
x0: Initial guess
max_iterations: Maximum number of iterations
tolerance: Convergence tolerance
restart: GMRES restart parameter
Returns:
SolverResult with solution and metadata
**Source:** [Line 622](Python/Linear_Algebra/core/linear_systems.py#L622)
**Class Source:** [Line 325](Python/Linear_Algebra/core/linear_systems.py#L325)
### `LinearSystemUtils`
Utility functions for linear systems.
Features:
- System conditioning analysis
- Residual analysis
- Solver selection heuristics
#### Methods
##### `analyze_system(A, b)`
Analyze linear system properties to guide solver selection.
Parameters:
A: Coefficient matrix
b: Right-hand side vector
Returns:
Dictionary with system properties
**Source:** [Line 691](Python/Linear_Algebra/core/linear_systems.py#L691)
##### `recommend_solver(A, b)`
Recommend appropriate solver based on system properties.
Parameters:
A: Coefficient matrix
b: Right-hand side vector
Returns:
Recommended solver name
**Source:** [Line 736](Python/Linear_Algebra/core/linear_systems.py#L736)
##### `solve_auto(A, b)`
Automatically select and apply appropriate solver.
Parameters:
A: Coefficient matrix
b: Right-hand side vector
**kwargs: Additional parameters passed to solver
Returns:
SolverResult from chosen solver
**Source:** [Line 767](Python/Linear_Algebra/core/linear_systems.py#L767)
**Class Source:** [Line 680](Python/Linear_Algebra/core/linear_systems.py#L680)