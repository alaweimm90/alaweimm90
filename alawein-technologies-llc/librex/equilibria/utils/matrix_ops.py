"""
Matrix operation utilities

Common matrix operations for optimization problems.
"""

from typing import Optional

import numpy as np


def permutation_to_matrix(perm: np.ndarray) -> np.ndarray:
    """
    Convert permutation vector to permutation matrix

    Args:
        perm: Permutation vector (e.g., [1, 2, 0])

    Returns:
        np.ndarray: Permutation matrix P where P[i, perm[i]] = 1

    Example:
        >>> perm = np.array([1, 2, 0])
        >>> P = permutation_to_matrix(perm)
        >>> P
        array([[0, 1, 0],
               [0, 0, 1],
               [1, 0, 0]])
    """
    n = len(perm)
    P = np.zeros((n, n))
    P[np.arange(n), perm] = 1
    return P


def matrix_to_permutation(P: np.ndarray) -> np.ndarray:
    """
    Convert permutation matrix to permutation vector

    Args:
        P: Permutation matrix (n x n)

    Returns:
        np.ndarray: Permutation vector

    Example:
        >>> P = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
        >>> perm = matrix_to_permutation(P)
        >>> perm
        array([1, 2, 0])
    """
    if P.ndim == 1:
        # Flattened matrix, reshape first
        n = int(np.sqrt(len(P)))
        P = P.reshape((n, n))

    return np.argmax(P, axis=1)


def create_doubly_stochastic_constraints(n: int) -> np.ndarray:
    """
    Generate doubly stochastic matrix constraints for permutation matrix

    Creates constraint matrix for: row sums = 1, column sums = 1

    Args:
        n: Dimension of permutation matrix

    Returns:
        np.ndarray: Constraint matrix of shape (2n, n²)

    Example:
        >>> constraints = create_doubly_stochastic_constraints(3)
        >>> constraints.shape
        (6, 9)
    """
    constraints = np.zeros((2 * n, n * n))

    # Row sum = 1 constraints
    for i in range(n):
        constraints[i, i*n:(i+1)*n] = 1

    # Column sum = 1 constraints
    for j in range(n):
        constraints[n + j, j::n] = 1

    return constraints


def swap_elements(arr: np.ndarray, i: int, j: int) -> np.ndarray:
    """
    Swap two elements in array (returns copy)

    Args:
        arr: Input array
        i: First index
        j: Second index

    Returns:
        np.ndarray: Array with elements at i and j swapped

    Example:
        >>> arr = np.array([0, 1, 2, 3])
        >>> swap_elements(arr, 1, 3)
        array([0, 3, 2, 1])
    """
    result = arr.copy()
    result[i], result[j] = result[j], result[i]
    return result


def generate_random_permutation(n: int, seed: Optional[int] = None) -> np.ndarray:
    """
    Generate random permutation

    Args:
        n: Size of permutation
        seed: Random seed for reproducibility

    Returns:
        np.ndarray: Random permutation of [0, 1, ..., n-1]

    Example:
        >>> perm = generate_random_permutation(5, seed=42)
        >>> len(perm)
        5
        >>> set(perm) == set(range(5))
        True
    """
    if seed is not None:
        np.random.seed(seed)

    perm = np.arange(n)
    np.random.shuffle(perm)
    return perm


def compute_permutation_distance(perm1: np.ndarray, perm2: np.ndarray) -> int:
    """
    Compute Hamming distance between two permutations

    Args:
        perm1: First permutation
        perm2: Second permutation

    Returns:
        int: Number of positions where permutations differ

    Example:
        >>> p1 = np.array([0, 1, 2, 3])
        >>> p2 = np.array([0, 2, 1, 3])
        >>> compute_permutation_distance(p1, p2)
        2
    """
    return np.sum(perm1 != perm2)


__all__ = [
    'permutation_to_matrix',
    'matrix_to_permutation',
    'create_doubly_stochastic_constraints',
    'swap_elements',
    'generate_random_permutation',
    'compute_permutation_distance',
]
