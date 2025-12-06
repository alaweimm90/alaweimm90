"""Tests for L-BFGS optimizer."""

import jax.numpy as jnp
import pytest
from Librex.methods.advanced.lbfgs import optimize_lbfgs, LBFGS


def test_lbfgs_rosenbrock():
    """Test L-BFGS on Rosenbrock function."""
    def rosenbrock(x):
        return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2
    
    x0 = jnp.array([-1.0, 1.0])
    result, info = optimize_lbfgs(rosenbrock, x0, max_iter=200)
    
    assert info['converged'], "L-BFGS should converge on Rosenbrock"
    assert jnp.allclose(result, jnp.array([1.0, 1.0]), atol=1e-3)
    assert rosenbrock(result) < 1e-4


def test_lbfgs_sphere():
    """Test L-BFGS on sphere function."""
    def sphere(x):
        return jnp.sum(x**2)
    
    x0 = jnp.array([5.0, 5.0, 5.0])
    result, info = optimize_lbfgs(sphere, x0)
    
    assert info['converged']
    assert jnp.allclose(result, jnp.zeros(3), atol=1e-4)


def test_lbfgs_quadratic():
    """Test L-BFGS on simple quadratic."""
    def quadratic(x):
        return x[0]**2 + 2*x[1]**2
    
    x0 = jnp.array([10.0, 10.0])
    result, info = optimize_lbfgs(quadratic, x0)
    
    assert info['converged']
    assert jnp.allclose(result, jnp.zeros(2), atol=1e-4)


def test_lbfgs_class_api():
    """Test LBFGS class API."""
    optimizer = LBFGS(m=10, tol=1e-6, max_iter=100)
    
    def f(x):
        return jnp.sum(x**2)
    
    x0 = jnp.array([3.0, 4.0])
    result = optimizer.optimize(f, x0)
    
    assert jnp.allclose(result, jnp.zeros(2), atol=1e-4)


def test_lbfgs_convergence_info():
    """Test that convergence info is returned correctly."""
    def f(x):
        return jnp.sum(x**2)
    
    x0 = jnp.array([1.0, 1.0])
    result, info = optimize_lbfgs(f, x0)
    
    assert 'iterations' in info
    assert 'function_evals' in info
    assert 'gradient_norm' in info
    assert 'converged' in info
    assert info['iterations'] > 0
    assert len(info['gradient_norm']) == info['iterations']


def test_lbfgs_history_size():
    """Test that history size parameter works."""
    def f(x):
        return jnp.sum(x**2)
    
    x0 = jnp.ones(10)
    
    # Small history
    result1, info1 = optimize_lbfgs(f, x0, m=3)
    
    # Large history
    result2, info2 = optimize_lbfgs(f, x0, m=20)
    
    assert jnp.allclose(result1, jnp.zeros(10), atol=1e-3)
    assert jnp.allclose(result2, jnp.zeros(10), atol=1e-3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
