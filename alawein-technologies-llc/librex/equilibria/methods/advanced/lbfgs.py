"""L-BFGS quasi-Newton optimization method."""

import jax.numpy as jnp
from jax import grad, jit
from typing import Callable, Tuple, List


@jit
def lbfgs_two_loop(grad_x: jnp.ndarray, s_hist: List, y_hist: List) -> jnp.ndarray:
    """Two-loop recursion for L-BFGS search direction.
    
    Args:
        grad_x: Current gradient
        s_hist: History of position differences
        y_hist: History of gradient differences
        
    Returns:
        Search direction
    """
    q = grad_x
    alpha = []
    
    # First loop (backward)
    for i in range(len(s_hist) - 1, -1, -1):
        rho = 1.0 / jnp.dot(y_hist[i], s_hist[i])
        a = rho * jnp.dot(s_hist[i], q)
        alpha.append(a)
        q = q - a * y_hist[i]
    
    # Initial Hessian approximation
    if len(s_hist) > 0:
        gamma = jnp.dot(s_hist[-1], y_hist[-1]) / jnp.dot(y_hist[-1], y_hist[-1])
        r = gamma * q
    else:
        r = q
    
    # Second loop (forward)
    alpha.reverse()
    for i in range(len(s_hist)):
        rho = 1.0 / jnp.dot(y_hist[i], s_hist[i])
        beta = rho * jnp.dot(y_hist[i], r)
        r = r + s_hist[i] * (alpha[i] - beta)
    
    return -r


def backtracking_line_search(
    f: Callable,
    x: jnp.ndarray,
    p: jnp.ndarray,
    grad_x: jnp.ndarray,
    c1: float = 1e-4,
    rho: float = 0.5,
    max_iter: int = 50
) -> float:
    """Backtracking line search with Armijo condition.
    
    Args:
        f: Objective function
        x: Current position
        p: Search direction
        grad_x: Current gradient
        c1: Armijo condition parameter
        rho: Backtracking factor
        max_iter: Maximum iterations
        
    Returns:
        Step size alpha
    """
    alpha = 1.0
    fx = f(x)
    descent = jnp.dot(grad_x, p)
    
    for _ in range(max_iter):
        if f(x + alpha * p) <= fx + c1 * alpha * descent:
            break
        alpha *= rho
        if alpha < 1e-10:
            break
    
    return alpha


def optimize_lbfgs(
    f: Callable,
    x0: jnp.ndarray,
    max_iter: int = 100,
    tol: float = 1e-6,
    m: int = 10,
    verbose: bool = False
) -> Tuple[jnp.ndarray, dict]:
    """L-BFGS optimization algorithm.
    
    Args:
        f: Objective function to minimize
        x0: Initial point
        max_iter: Maximum iterations
        tol: Convergence tolerance
        m: History size
        verbose: Print progress
        
    Returns:
        Tuple of (optimal point, info dict)
    """
    x = x0
    s_hist = []
    y_hist = []
    grad_f = grad(f)
    
    info = {
        'iterations': 0,
        'function_evals': 0,
        'gradient_norm': [],
        'converged': False
    }
    
    for i in range(max_iter):
        g = grad_f(x)
        grad_norm = jnp.linalg.norm(g)
        info['gradient_norm'].append(float(grad_norm))
        
        if verbose and i % 10 == 0:
            print(f"Iter {i}: f(x) = {f(x):.6e}, ||g|| = {grad_norm:.6e}")
        
        if grad_norm < tol:
            info['converged'] = True
            break
        
        # Compute search direction
        p = lbfgs_two_loop(g, s_hist, y_hist)
        
        # Line search
        alpha = backtracking_line_search(f, x, p, g)
        info['function_evals'] += 1
        
        # Update position
        x_new = x + alpha * p
        g_new = grad_f(x_new)
        
        # Update history
        s = x_new - x
        y = g_new - g
        
        s_hist.append(s)
        y_hist.append(y)
        
        # Keep only m most recent updates
        if len(s_hist) > m:
            s_hist.pop(0)
            y_hist.pop(0)
        
        x = x_new
        info['iterations'] = i + 1
    
    return x, info


class LBFGS:
    """L-BFGS optimizer class for API compatibility."""
    
    def __init__(self, m: int = 10, tol: float = 1e-6, max_iter: int = 100):
        self.m = m
        self.tol = tol
        self.max_iter = max_iter
    
    def optimize(self, f: Callable, x0: jnp.ndarray, verbose: bool = False) -> jnp.ndarray:
        """Optimize function f starting from x0."""
        result, _ = optimize_lbfgs(f, x0, self.max_iter, self.tol, self.m, verbose)
        return result
