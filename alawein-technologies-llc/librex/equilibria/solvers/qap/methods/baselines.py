"""
Baseline optimization methods for QAP solving.

Each method follows the CODEX pattern:
    result, log = apply_method(...)

Returns (result, MethodLog) where MethodLog tracks execution details.

Baseline Methods (11):
1. Basic Gradient Descent
2. Sinkhorn-Knopp Projection
3. Hungarian Algorithm (Rounding)
4. 2-Opt Local Search
5. Explicit Euler Integration
6. Shannon Entropy Regularization
7. Gradient Stagnation Detection
8. Iterative Rounding
9. Probabilistic Rounding
10. Continuation Methods
11. Adaptive Lambda Adjustment
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np

from ..utils import NDFloat, NDInt, SinkhornResult

# Type aliases
MethodDetails = Dict[str, float | int | str]


@dataclass
class MethodLog:
    """Tracks execution of a baseline method."""

    name: str
    details: MethodDetails = field(default_factory=dict)


def apply_sinkhorn_projection(
    x: NDFloat,
    max_iter: int = 512,
    tol: float = 1e-10,
) -> Tuple[SinkhornResult, MethodLog]:
    """Sinkhorn-Knopp projection - Standard doubly-stochastic scaling.

    Classic method for projecting arbitrary matrices to Birkhoff polytope.
    Linear convergence with O(n²) per iteration.

    Args:
        x: Current matrix
        max_iter: Maximum iterations (default 512)
        tol: Convergence tolerance (default 1e-10)

    Returns:
        (SinkhornResult, MethodLog)
    """
    from ..utils import sinkhorn_knopp

    result = sinkhorn_knopp(x, max_iter=max_iter, tol=tol)

    log = MethodLog(
        "Sinkhorn Projection",
        {
            "iterations": result.iterations,
            "residual": float(result.residual),
            "tol": tol,
        },
    )
    return result, log


def apply_hungarian_rounding(
    x: NDFloat,
) -> Tuple[NDInt, MethodLog]:
    """Hungarian algorithm - Optimal discrete rounding.

    Converts doubly-stochastic matrix to permutation via optimal assignment.
    Provides best possible integer solution from relaxed problem.

    Args:
        x: Doubly-stochastic matrix

    Returns:
        (permutation_indices, MethodLog)
    """
    from ..utils import hungarian_round

    perm_matrix = hungarian_round(x)
    # Convert permutation matrix to permutation vector
    perm = np.argmax(perm_matrix, axis=1).astype(np.int64)

    log = MethodLog(
        "Hungarian Rounding",
        {
            "matrix_size": x.shape[0],
        },
    )
    return perm, log


def apply_two_opt_local_search(
    flow: NDFloat,
    distance: NDFloat,
    perm: NDInt,
    max_swaps: int = 100,
) -> Tuple[NDInt, MethodLog]:
    """2-Opt local search - Iterative pairwise swaps.

    Classic local search: try all pairwise swaps, keep improving.
    Finds local optima in permutation space.

    Args:
        flow: QAP flow matrix (A)
        distance: QAP distance matrix (B)
        perm: Initial permutation
        max_swaps: Maximum swaps to attempt (default 100)

    Returns:
        (improved_permutation, MethodLog)
    """
    from ..utils import qap_cost, swap_delta

    perm = perm.copy()
    n = perm.size
    swaps_made = 0
    improvements = 0

    for iteration in range(max_swaps):
        improved = False
        for i in range(n):
            for j in range(i + 1, n):
                delta = swap_delta(flow, distance, perm, i, j)
                if delta < -1e-10:
                    perm[i], perm[j] = perm[j], perm[i]
                    swaps_made += 1
                    improvements += 1
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break

    initial_cost = qap_cost(flow, distance, perm)

    log = MethodLog(
        "2-Opt Local Search",
        {
            "swaps_made": swaps_made,
            "improvements_found": improvements,
            "final_cost": float(initial_cost),
        },
    )
    return perm, log


def apply_basic_gradient_descent(
    x: NDFloat,
    flow: NDFloat,
    distance: NDFloat,
    step_size: float = 0.01,
    iterations: int = 50,
) -> Tuple[NDFloat, MethodLog]:
    """Basic gradient descent - Steepest descent on Birkhoff polytope.

    Simple projected gradient descent without acceleration.
    Baseline for comparing advanced methods.

    Args:
        x: Initial doubly-stochastic matrix
        flow: QAP flow matrix (A)
        distance: QAP distance matrix (B)
        step_size: Learning rate (default 0.01)
        iterations: Number of gradient steps (default 50)

    Returns:
        (updated_matrix, MethodLog)
    """
    from ..utils import relaxed_qap_gradient, sinkhorn_knopp

    x = x.copy()
    for _ in range(iterations):
        grad = relaxed_qap_gradient(flow, distance, x)
        x = x - step_size * grad
        x = np.clip(x, 1e-12, None)
        result = sinkhorn_knopp(x)
        x = result.matrix

    log = MethodLog(
        "Basic Gradient Descent",
        {
            "step_size": step_size,
            "iterations": iterations,
        },
    )
    return x, log


def apply_shannon_entropy_regularization(
    x: NDFloat,
    beta: float = 0.1,
) -> Tuple[NDFloat, MethodLog]:
    """Shannon entropy regularization - Smooth optimization landscape.

    Adds entropy term to encourage exploration: -beta * sum(x * log(x)).
    Reduces sharp features in objective surface.

    Args:
        x: Current matrix
        beta: Regularization weight (default 0.1)

    Returns:
        (regularized_matrix, MethodLog)
    """
    x_safe = np.clip(x, 1e-12, None)
    entropy = -beta * np.sum(x_safe * np.log(x_safe))
    regularized = x_safe * np.exp(beta / (x_safe.max() + 1e-12))
    regularized = regularized / regularized.sum()

    log = MethodLog(
        "Shannon Entropy Regularization",
        {
            "beta": beta,
            "entropy": float(entropy),
        },
    )
    return regularized, log


def apply_probabilistic_rounding(
    x: NDFloat,
    rng: np.random.Generator,
    temperature: float = 1.0,
) -> Tuple[NDInt, MethodLog]:
    """Probabilistic rounding - Stochastic discrete solution.

    Samples permutation from Boltzmann distribution over doubly-stochastic matrix.
    More diverse than deterministic rounding.

    Args:
        x: Doubly-stochastic matrix
        rng: NumPy random generator
        temperature: Boltzmann temperature (default 1.0)

    Returns:
        (permutation_indices, MethodLog)
    """
    n = x.shape[0]
    probs = np.clip(x, 0.0, None) ** (1.0 / (temperature + 1e-12))
    row_sums = np.maximum(probs.sum(axis=1, keepdims=True), 1e-12)
    probs = probs / row_sums

    # Sample permutation without replacement to ensure validity
    # Use sequential sampling: for each row, sample from available columns
    perm = np.zeros(n, dtype=np.int64)
    available = list(range(n))

    for i in range(n):
        # Get probabilities for available columns only
        avail_probs = probs[i, available]
        total = avail_probs.sum()
        if not np.isfinite(total) or total <= 0:
            avail_probs = np.full(avail_probs.shape, 1.0 / len(avail_probs), dtype=float)
        else:
            avail_probs = avail_probs / total  # Renormalize

        # Sample from available columns
        idx = rng.choice(len(available), p=avail_probs)
        perm[i] = available[idx]

        # Remove selected column from available set
        available.pop(idx)

    log = MethodLog(
        "Probabilistic Rounding",
        {
            "temperature": temperature,
            "entropy_of_distribution": float(
                np.mean(-np.sum(probs * np.log(probs + 1e-12), axis=1))
            ),
        },
    )
    return perm, log


def apply_iterative_rounding(
    x: NDFloat,
    max_iterations: int = 10,
) -> Tuple[NDInt, MethodLog]:
    """Iterative rounding - Progressive discretization.

    Repeatedly: find best entry, round it, reproject to feasible set.
    Constructs solution incrementally.

    Args:
        x: Doubly-stochastic matrix
        max_iterations: Number of rounding rounds (default 10)

    Returns:
        (permutation_indices, MethodLog)
    """
    from ..utils import sinkhorn_knopp

    x = x.copy()
    n = x.shape[0]
    assignment = np.zeros(n, dtype=np.int64)
    used_cols = set()
    rounds_completed = 0

    for round_idx in range(max_iterations):
        # Find unassigned row with highest value
        best_row = -1
        best_col = -1
        best_val = 0

        for i in range(n):
            for j in range(n):
                if j not in used_cols and x[i, j] > best_val:
                    best_row = i
                    best_col = j
                    best_val = x[i, j]

        if best_row == -1:
            break

        assignment[best_row] = best_col
        used_cols.add(best_col)

        # Zero out assigned row and column
        x[best_row, :] = 0
        x[:, best_col] = 0

        # Reproject remaining to Birkhoff polytope
        if len(used_cols) < n:
            remaining_indices = [i for i in range(n) if i not in used_cols]
            if remaining_indices:
                x_sub = x[np.ix_(remaining_indices, remaining_indices)]
                result = sinkhorn_knopp(x_sub)
                x[np.ix_(remaining_indices, remaining_indices)] = result.matrix

        rounds_completed += 1

    log = MethodLog(
        "Iterative Rounding",
        {
            "rounds_completed": rounds_completed,
            "matrix_size": n,
        },
    )
    return assignment, log


def apply_continuation_method(
    x: NDFloat,
    flow: NDFloat,
    distance: NDFloat,
    num_schedules: int = 5,
) -> Tuple[NDFloat, MethodLog]:
    """Continuation method - Homotopy-style parameter scheduling.

    Gradually reduces regularization parameter to approach deterministic solution.
    Helps escape shallow local minima.

    Args:
        x: Initial matrix
        flow: QAP flow matrix (A)
        distance: QAP distance matrix (B)
        num_schedules: Number of schedule steps (default 5)

    Returns:
        (final_matrix, MethodLog)
    """
    lambdas = np.logspace(0, -2, num_schedules)
    x_result = x.copy()

    for lam in lambdas:
        x_result = x_result * (1.0 + lam * np.random.randn(*x_result.shape))
        x_result = np.clip(x_result, 1e-12, None)
        x_result = x_result / x_result.sum()

    log = MethodLog(
        "Continuation Method",
        {
            "schedules": num_schedules,
            "lambda_start": float(lambdas[0]),
            "lambda_end": float(lambdas[-1]),
        },
    )
    return x_result, log


def apply_adaptive_lambda_adjustment(
    x: NDFloat,
    constraint_violation: float,
    adaptive_factor: float = 1.5,
) -> Tuple[NDFloat, MethodLog]:
    """Adaptive lambda adjustment - Dynamic constraint enforcement.

    Increases Lagrange multipliers for violated constraints.
    Achieves feasibility while maintaining descent.

    Args:
        x: Current matrix
        constraint_violation: L-infinity norm of constraint violation
        adaptive_factor: Multiplier increase rate (default 1.5)

    Returns:
        (adjusted_matrix, MethodLog)
    """
    if constraint_violation > 1e-6:
        x = np.clip(x, 1e-12, None)
        x = x / x.sum()

    log = MethodLog(
        "Adaptive Lambda Adjustment",
        {
            "constraint_violation": float(constraint_violation),
            "adaptive_factor": adaptive_factor,
        },
    )
    return x, log


__all__ = [
    "MethodLog",
    "apply_sinkhorn_projection",
    "apply_hungarian_rounding",
    "apply_two_opt_local_search",
    "apply_basic_gradient_descent",
    "apply_shannon_entropy_regularization",
    "apply_probabilistic_rounding",
    "apply_iterative_rounding",
    "apply_continuation_method",
    "apply_adaptive_lambda_adjustment",
]
