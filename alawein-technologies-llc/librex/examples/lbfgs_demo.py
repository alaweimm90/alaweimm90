"""Demo of L-BFGS optimizer."""

import jax.numpy as jnp
from Librex.methods.advanced.lbfgs import optimize_lbfgs, LBFGS


def rosenbrock(x):
    """Rosenbrock function: f(x,y) = (1-x)^2 + 100(y-x^2)^2"""
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2


def sphere(x):
    """Sphere function: f(x) = sum(x_i^2)"""
    return jnp.sum(x**2)


def rastrigin(x):
    """Rastrigin function (harder to optimize)"""
    n = len(x)
    return 10 * n + jnp.sum(x**2 - 10 * jnp.cos(2 * jnp.pi * x))


def main():
    print("=" * 60)
    print("L-BFGS Optimization Demo")
    print("=" * 60)
    
    # Example 1: Rosenbrock
    print("\n1. Rosenbrock Function")
    print("-" * 40)
    x0 = jnp.array([-1.0, 1.0])
    result, info = optimize_lbfgs(rosenbrock, x0, verbose=True)
    print(f"\nOptimal point: {result}")
    print(f"Optimal value: {rosenbrock(result):.6e}")
    print(f"Iterations: {info['iterations']}")
    print(f"Converged: {info['converged']}")
    
    # Example 2: High-dimensional sphere
    print("\n\n2. 100-Dimensional Sphere Function")
    print("-" * 40)
    x0 = jnp.ones(100) * 5.0
    result, info = optimize_lbfgs(sphere, x0, max_iter=50)
    print(f"Optimal value: {sphere(result):.6e}")
    print(f"Iterations: {info['iterations']}")
    print(f"Final gradient norm: {info['gradient_norm'][-1]:.6e}")
    
    # Example 3: Using class API
    print("\n\n3. Using LBFGS Class API")
    print("-" * 40)
    optimizer = LBFGS(m=10, tol=1e-6, max_iter=100)
    
    def quadratic(x):
        return x[0]**2 + 2*x[1]**2 + 3*x[2]**2
    
    x0 = jnp.array([10.0, 10.0, 10.0])
    result = optimizer.optimize(quadratic, x0, verbose=False)
    print(f"Optimal point: {result}")
    print(f"Optimal value: {quadratic(result):.6e}")
    
    # Example 4: Comparison with different history sizes
    print("\n\n4. Effect of History Size")
    print("-" * 40)
    x0 = jnp.ones(50) * 3.0
    
    for m in [3, 10, 20]:
        result, info = optimize_lbfgs(sphere, x0, m=m, max_iter=50)
        print(f"m={m:2d}: {info['iterations']:3d} iterations, "
              f"final value = {sphere(result):.6e}")
    
    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
