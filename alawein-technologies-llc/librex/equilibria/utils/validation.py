"""
Validation utilities for optimization problems

Shared validation functions to avoid code duplication across adapters.
"""

import warnings
from typing import Optional

import numpy as np


def validate_permutation(perm: np.ndarray, n: Optional[int] = None) -> bool:
    """
    Check if array is a valid permutation

    Args:
        perm: Array to check
        n: Expected length (if None, inferred from perm)

    Returns:
        bool: True if valid permutation, False otherwise

    Example:
        >>> validate_permutation(np.array([0, 1, 2]))
        True
        >>> validate_permutation(np.array([0, 0, 1]))
        False
    """
    if n is None:
        n = len(perm)

    return (
        len(perm) == n and
        np.all(np.sort(perm) == np.arange(n))
    )


def validate_distance_matrix(
    distances: np.ndarray,
    check_symmetry: bool = True,
    check_triangle_inequality: bool = False,
    tolerance: float = 1e-10
) -> None:
    """
    Validate a distance matrix

    Args:
        distances: Distance matrix to validate
        check_symmetry: Whether to check for symmetry
        check_triangle_inequality: Whether to check triangle inequality
        tolerance: Numerical tolerance for comparisons

    Raises:
        ValueError: If validation fails

    Warnings:
        RuntimeWarning: For non-critical issues

    Example:
        >>> dist = np.array([[0, 1, 2], [1, 0, 3], [2, 3, 0]])
        >>> validate_distance_matrix(dist)  # OK
        >>>
        >>> bad_dist = np.array([[0, -1], [1, 0]])
        >>> validate_distance_matrix(bad_dist)  # Raises ValueError
    """
    # Check square matrix
    if distances.shape[0] != distances.shape[1]:
        raise ValueError(
            f"Distance matrix must be square, got shape {distances.shape}"
        )

    # Check non-negativity
    if np.any(distances < 0):
        raise ValueError("Distance matrix contains negative values")

    # Check diagonal is zero (or close to zero)
    if not np.allclose(np.diag(distances), 0, atol=tolerance):
        warnings.warn(
            "Distance matrix diagonal is not zero. "
            "Self-distances should typically be zero.",
            RuntimeWarning
        )

    # Check for NaN or inf
    if np.any(np.isnan(distances)) or np.any(np.isinf(distances)):
        raise ValueError("Distance matrix contains NaN or infinite values")

    # Check symmetry (common in distance matrices)
    if check_symmetry and not np.allclose(distances, distances.T, rtol=tolerance):
        warnings.warn(
            "Distance matrix is not symmetric. "
            "Asymmetric distance matrices are valid but uncommon.",
            RuntimeWarning
        )

    # Check triangle inequality (computationally expensive)
    if check_triangle_inequality:
        n = len(distances)
        violations = 0
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    if distances[i, j] > distances[i, k] + distances[k, j] + tolerance:
                        violations += 1

        if violations > 0:
            warnings.warn(
                f"Triangle inequality violated {violations} times. "
                f"This may indicate an invalid distance metric.",
                RuntimeWarning
            )


def validate_matrix_properties(
    matrix: np.ndarray,
    symmetric: bool = False,
    non_negative: bool = False,
    square: bool = True,
    tolerance: float = 1e-10
) -> dict:
    """
    Validate general matrix properties

    Args:
        matrix: Matrix to validate
        symmetric: Require symmetry
        non_negative: Require all elements >= 0
        square: Require square matrix
        tolerance: Numerical tolerance

    Returns:
        dict: Dictionary of property validation results

    Example:
        >>> A = np.array([[0, 1], [1, 0]])
        >>> props = validate_matrix_properties(A, symmetric=True)
        >>> props['is_symmetric']
        True
    """
    results = {
        'shape': matrix.shape,
        'is_square': matrix.shape[0] == matrix.shape[1],
        'is_symmetric': np.allclose(matrix, matrix.T, rtol=tolerance),
        'is_non_negative': np.all(matrix >= 0),
        'has_nan': np.any(np.isnan(matrix)),
        'has_inf': np.any(np.isinf(matrix)),
        'diagonal_zero': np.allclose(np.diag(matrix), 0, atol=tolerance) if matrix.shape[0] == matrix.shape[1] else False,
    }

    # Check required properties
    if square and not results['is_square']:
        raise ValueError(f"Matrix must be square, got shape {matrix.shape}")

    if symmetric and not results['is_symmetric']:
        raise ValueError("Matrix must be symmetric")

    if non_negative and not results['is_non_negative']:
        raise ValueError("Matrix contains negative values")

    if results['has_nan']:
        raise ValueError("Matrix contains NaN values")

    if results['has_inf']:
        raise ValueError("Matrix contains infinite values")

    return results


__all__ = [
    'validate_permutation',
    'validate_distance_matrix',
    'validate_matrix_properties',
]
