"""
Input validation framework for Librex.QAP.

Provides comprehensive validation functions to ensure data integrity
and provide helpful error messages for debugging.
"""

from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray


def validate_matrix(
    matrix: NDArray,
    name: str = "matrix",
    shape: Optional[Tuple[int, ...]] = None,
    square: bool = False,
    positive: bool = False,
    symmetric: bool = False,
) -> None:
    """
    Validate matrix properties and raise informative errors.

    Parameters
    ----------
    matrix : np.ndarray
        Matrix to validate
    name : str, default="matrix"
        Name of matrix (for error messages)
    shape : tuple of int, optional
        Expected shape (e.g., (n, n))
    square : bool, default=False
        Require square matrix
    positive : bool, default=False
        Require all elements >= 0
    symmetric : bool, default=False
        Require symmetric matrix

    Raises
    ------
    TypeError
        If matrix is not ndarray
    ValueError
        If any validation fails
    """
    if not isinstance(matrix, np.ndarray):
        raise TypeError(f"{name} must be np.ndarray, got {type(matrix).__name__}")

    if matrix.ndim != 2:
        raise ValueError(f"{name} must be 2D, got {matrix.ndim}D with shape {matrix.shape}")

    if square and matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} must be square, got shape {matrix.shape}")

    if shape is not None and matrix.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {matrix.shape}")

    if positive and np.any(matrix < 0):
        raise ValueError(f"{name} must be non-negative, found minimum value: {np.min(matrix)}")

    if symmetric and not np.allclose(matrix, matrix.T):
        max_diff = np.max(np.abs(matrix - matrix.T))
        raise ValueError(f"{name} must be symmetric (max difference: {max_diff})")


def validate_doubly_stochastic(
    matrix: NDArray,
    name: str = "matrix",
    tol: float = 1e-9,
) -> None:
    """
    Validate doubly-stochastic property.

    Parameters
    ----------
    matrix : np.ndarray
        Matrix to validate (n x n)
    name : str, default="matrix"
        Matrix name (for error messages)
    tol : float, default=1e-9
        Tolerance for sum violations

    Raises
    ------
    ValueError
        If not doubly-stochastic
    """
    validate_matrix(matrix, name=name, square=True, positive=True)

    n = matrix.shape[0]
    ones = np.ones(n)

    # Check row sums
    row_sums = matrix @ ones
    if not np.allclose(row_sums, 1.0, atol=tol):
        worst_row = np.argmax(np.abs(row_sums - 1.0))
        raise ValueError(
            f"{name} row {worst_row} sums to {row_sums[worst_row]}, "
            f"expected 1.0 (tolerance: {tol})"
        )

    # Check column sums
    col_sums = matrix.T @ ones
    if not np.allclose(col_sums, 1.0, atol=tol):
        worst_col = np.argmax(np.abs(col_sums - 1.0))
        raise ValueError(
            f"{name} column {worst_col} sums to {col_sums[worst_col]}, "
            f"expected 1.0 (tolerance: {tol})"
        )


def validate_permutation(
    matrix: NDArray,
    name: str = "permutation",
    tol: float = 1e-9,
) -> None:
    """
    Validate permutation matrix (binary, doubly-stochastic).

    Parameters
    ----------
    matrix : np.ndarray
        Matrix to validate (n x n)
    name : str, default="permutation"
        Matrix name (for error messages)
    tol : float, default=1e-9
        Tolerance for comparisons

    Raises
    ------
    ValueError
        If not a valid permutation matrix
    """
    validate_matrix(matrix, name=name, square=True)

    # Check binary property: each element should be 0 or 1
    if not np.allclose(matrix * (1 - matrix), 0, atol=tol):
        bad_indices = np.where((matrix < -tol) | (matrix > 1 + tol))
        raise ValueError(
            f"{name} must be binary (0 or 1), " f"found {len(bad_indices[0])} invalid entries"
        )

    # Check doubly-stochastic (should be exact for integer matrices)
    validate_doubly_stochastic(matrix, name=name, tol=tol)


def validate_qap_problem(
    A: NDArray,
    B: NDArray,
    n: Optional[int] = None,
) -> None:
    """
    Validate QAP problem matrices.

    Parameters
    ----------
    A : np.ndarray
        Flow matrix (n x n)
    B : np.ndarray
        Distance matrix (n x n)
    n : int, optional
        Expected problem size

    Raises
    ------
    ValueError
        If matrices don't form valid QAP
    """
    validate_matrix(A, name="Flow matrix A", square=True)
    validate_matrix(B, name="Distance matrix B", square=True)

    if A.shape != B.shape:
        raise ValueError(f"A and B must have same shape, got A:{A.shape} and B:{B.shape}")

    if n is not None and A.shape[0] != n:
        raise ValueError(f"Expected n={n}, got matrix size {A.shape[0]}")


def validate_permutation_vector(
    perm: NDArray,
    n: Optional[int] = None,
    name: str = "permutation",
) -> None:
    """
    Validate permutation vector.

    Parameters
    ----------
    perm : np.ndarray
        Permutation indices
    n : int, optional
        Expected size
    name : str, default="permutation"
        Vector name (for error messages)

    Raises
    ------
    ValueError
        If not a valid permutation
    """
    if not isinstance(perm, np.ndarray):
        raise TypeError(f"{name} must be np.ndarray, got {type(perm).__name__}")

    if perm.ndim != 1:
        raise ValueError(f"{name} must be 1D, got {perm.ndim}D")

    if len(perm) == 0:
        raise ValueError(f"{name} must not be empty")

    size = len(perm)

    if n is not None and size != n:
        raise ValueError(f"Expected size {n}, got {size}")

    # Check is valid permutation (0 to n-1, each value appears once)
    if not np.array_equal(np.sort(perm), np.arange(size)):
        unique_vals = np.unique(perm)
        if len(unique_vals) != size:
            raise ValueError(f"{name} has duplicate values: {perm}")
        if np.min(perm) != 0 or np.max(perm) != size - 1:
            raise ValueError(
                f"{name} values must be in [0, {size-1}], "
                f"got min={np.min(perm)}, max={np.max(perm)}"
            )


def validate_history_dict(history: dict, name: str = "history") -> None:
    """
    Validate convergence history dictionary.

    Parameters
    ----------
    history : dict
        History dictionary with 'times', 'objectives', etc.
    name : str, default="history"
        Dictionary name (for error messages)

    Raises
    ------
    ValueError
        If required keys missing or malformed
    """
    required_keys = ["times", "objectives"]
    missing = [k for k in required_keys if k not in history]

    if missing:
        raise ValueError(f"{name} missing required keys: {missing}")

    times = history["times"]
    objectives = history["objectives"]

    if not isinstance(times, (list, np.ndarray)):
        raise TypeError(f"{name}['times'] must be list or array")

    if not isinstance(objectives, (list, np.ndarray)):
        raise TypeError(f"{name}['objectives'] must be list or array")

    if len(times) != len(objectives):
        raise ValueError(
            f"{name}['times'] and ['objectives'] must have same length, "
            f"got {len(times)} and {len(objectives)}"
        )

    if len(times) > 0:
        times_arr = np.asarray(times)
        if not np.all(np.diff(times_arr) >= 0):
            raise ValueError(f"{name}['times'] must be non-decreasing")


__all__ = [
    "validate_matrix",
    "validate_doubly_stochastic",
    "validate_permutation",
    "validate_qap_problem",
    "validate_permutation_vector",
    "validate_history_dict",
]
