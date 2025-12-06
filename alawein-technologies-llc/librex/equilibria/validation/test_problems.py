"""
Analytical test problems with known solutions for validating optimization algorithms.

This module provides a comprehensive suite of test functions with known analytical
solutions or well-established global optima for validating optimization methods.
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass


@dataclass
class TestProblem:
    """Base class for test problems with analytical solutions."""

    name: str
    dimension: int
    bounds: Tuple[float, float]
    optimal_solution: np.ndarray
    optimal_value: float
    description: str

    def evaluate(self, x: np.ndarray) -> float:
        """Evaluate the objective function at point x."""
        raise NotImplementedError


class ConvexQuadratic(TestProblem):
    """
    Convex quadratic function: f(x) = (1/2) x^T Q x + b^T x + c

    Analytical solution: x* = -Q^{-1} b
    where Q is positive definite.
    """

    def __init__(self, n: int, seed: Optional[int] = None):
        if seed is not None:
            np.random.seed(seed)

        # Generate positive definite Q
        A = np.random.randn(n, n)
        self.Q = A.T @ A + np.eye(n)  # Ensure positive definiteness
        self.b = np.random.randn(n)
        self.c = np.random.randn()

        # Analytical solution
        optimal_sol = -np.linalg.solve(self.Q, self.b)
        optimal_val = self._evaluate_quadratic(optimal_sol)

        super().__init__(
            name=f"ConvexQuadratic_{n}D",
            dimension=n,
            bounds=(-10.0, 10.0),
            optimal_solution=optimal_sol,
            optimal_value=optimal_val,
            description=f"Convex quadratic function in {n} dimensions"
        )

    def _evaluate_quadratic(self, x: np.ndarray) -> float:
        """Evaluate the quadratic function."""
        return 0.5 * x @ self.Q @ x + self.b @ x + self.c

    def evaluate(self, x: np.ndarray) -> float:
        """Evaluate the objective function."""
        return self._evaluate_quadratic(x)


class SphereFunction(TestProblem):
    """
    Sphere function: f(x) = sum(x_i^2)

    Global minimum: x* = (0, 0, ..., 0), f(x*) = 0
    Convex, unimodal, separable.
    """

    def __init__(self, n: int):
        super().__init__(
            name=f"Sphere_{n}D",
            dimension=n,
            bounds=(-5.12, 5.12),
            optimal_solution=np.zeros(n),
            optimal_value=0.0,
            description=f"Sphere function in {n} dimensions"
        )

    def evaluate(self, x: np.ndarray) -> float:
        """Evaluate the sphere function."""
        return np.sum(x**2)


class RosenbrockFunction(TestProblem):
    """
    Rosenbrock function: f(x) = sum_{i=1}^{n-1} [100(x_{i+1} - x_i^2)^2 + (1 - x_i)^2]

    Global minimum: x* = (1, 1, ..., 1), f(x*) = 0
    Non-convex, unimodal, narrow valley.
    """

    def __init__(self, n: int):
        super().__init__(
            name=f"Rosenbrock_{n}D",
            dimension=n,
            bounds=(-2.048, 2.048),
            optimal_solution=np.ones(n),
            optimal_value=0.0,
            description=f"Rosenbrock function in {n} dimensions"
        )

    def evaluate(self, x: np.ndarray) -> float:
        """Evaluate the Rosenbrock function."""
        return np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)


class RastriginFunction(TestProblem):
    """
    Rastrigin function: f(x) = 10n + sum(x_i^2 - 10*cos(2*pi*x_i))

    Global minimum: x* = (0, 0, ..., 0), f(x*) = 0
    Highly multimodal with regular lattice of local minima.
    """

    def __init__(self, n: int):
        super().__init__(
            name=f"Rastrigin_{n}D",
            dimension=n,
            bounds=(-5.12, 5.12),
            optimal_solution=np.zeros(n),
            optimal_value=0.0,
            description=f"Rastrigin function in {n} dimensions"
        )

    def evaluate(self, x: np.ndarray) -> float:
        """Evaluate the Rastrigin function."""
        n = len(x)
        return 10 * n + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))


class AckleyFunction(TestProblem):
    """
    Ackley function: Complex multimodal function with exponential terms.

    Global minimum: x* = (0, 0, ..., 0), f(x*) = 0
    Multimodal with one global minimum.
    """

    def __init__(self, n: int):
        super().__init__(
            name=f"Ackley_{n}D",
            dimension=n,
            bounds=(-32.768, 32.768),
            optimal_solution=np.zeros(n),
            optimal_value=0.0,
            description=f"Ackley function in {n} dimensions"
        )

    def evaluate(self, x: np.ndarray) -> float:
        """Evaluate the Ackley function."""
        n = len(x)
        a, b, c = 20, 0.2, 2 * np.pi
        sum_sq = np.sum(x**2)
        sum_cos = np.sum(np.cos(c * x))
        return -a * np.exp(-b * np.sqrt(sum_sq / n)) - np.exp(sum_cos / n) + a + np.e


class GriewankFunction(TestProblem):
    """
    Griewank function: Product of cosines divided by square root.

    Global minimum: x* = (0, 0, ..., 0), f(x*) = 0
    Multimodal with widespread regularly distributed local minima.
    """

    def __init__(self, n: int):
        super().__init__(
            name=f"Griewank_{n}D",
            dimension=n,
            bounds=(-600.0, 600.0),
            optimal_solution=np.zeros(n),
            optimal_value=0.0,
            description=f"Griewank function in {n} dimensions"
        )

    def evaluate(self, x: np.ndarray) -> float:
        """Evaluate the Griewank function."""
        sum_term = np.sum(x**2) / 4000
        prod_term = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))
        return sum_term - prod_term + 1


class SchwefelFunction(TestProblem):
    """
    Schwefel function: f(x) = 418.9829n - sum(x_i * sin(sqrt(|x_i|)))

    Global minimum: x* = (420.9687, ..., 420.9687), f(x*) ≈ 0
    Multimodal with deceptive global structure.
    """

    def __init__(self, n: int):
        super().__init__(
            name=f"Schwefel_{n}D",
            dimension=n,
            bounds=(-500.0, 500.0),
            optimal_solution=np.full(n, 420.9687),
            optimal_value=0.0,
            description=f"Schwefel function in {n} dimensions"
        )

    def evaluate(self, x: np.ndarray) -> float:
        """Evaluate the Schwefel function."""
        n = len(x)
        return 418.9829 * n - np.sum(x * np.sin(np.sqrt(np.abs(x))))


class LevyFunction(TestProblem):
    """
    Levy function: Complex multimodal function.

    Global minimum: x* = (1, 1, ..., 1), f(x*) = 0
    """

    def __init__(self, n: int):
        super().__init__(
            name=f"Levy_{n}D",
            dimension=n,
            bounds=(-10.0, 10.0),
            optimal_solution=np.ones(n),
            optimal_value=0.0,
            description=f"Levy function in {n} dimensions"
        )

    def evaluate(self, x: np.ndarray) -> float:
        """Evaluate the Levy function."""
        w = 1 + (x - 1) / 4
        term1 = np.sin(np.pi * w[0])**2
        term2 = np.sum((w[:-1] - 1)**2 * (1 + 10 * np.sin(np.pi * w[:-1] + 1)**2))
        term3 = (w[-1] - 1)**2 * (1 + np.sin(2 * np.pi * w[-1])**2)
        return term1 + term2 + term3


class MichalewiczFunction(TestProblem):
    """
    Michalewicz function: f(x) = -sum(sin(x_i) * sin^(2m)((i+1)*x_i^2/pi))

    Global minimum location depends on dimension.
    Multimodal with steep valleys.
    """

    def __init__(self, n: int, m: int = 10):
        # Approximate optimal values for common dimensions
        optimal_vals = {
            2: -1.8013,
            5: -4.687658,
            10: -9.66015
        }

        super().__init__(
            name=f"Michalewicz_{n}D",
            dimension=n,
            bounds=(0, np.pi),
            optimal_solution=None,  # No closed-form solution
            optimal_value=optimal_vals.get(n, -0.9665 * n),  # Approximation
            description=f"Michalewicz function in {n} dimensions"
        )
        self.m = m

    def evaluate(self, x: np.ndarray) -> float:
        """Evaluate the Michalewicz function."""
        i = np.arange(1, len(x) + 1)
        return -np.sum(np.sin(x) * np.sin(i * x**2 / np.pi)**(2 * self.m))


class ZakharovFunction(TestProblem):
    """
    Zakharov function: f(x) = sum(x_i^2) + (sum(0.5*i*x_i))^2 + (sum(0.5*i*x_i))^4

    Global minimum: x* = (0, 0, ..., 0), f(x*) = 0
    """

    def __init__(self, n: int):
        super().__init__(
            name=f"Zakharov_{n}D",
            dimension=n,
            bounds=(-5.0, 10.0),
            optimal_solution=np.zeros(n),
            optimal_value=0.0,
            description=f"Zakharov function in {n} dimensions"
        )

    def evaluate(self, x: np.ndarray) -> float:
        """Evaluate the Zakharov function."""
        i = np.arange(1, len(x) + 1)
        sum1 = np.sum(x**2)
        sum2 = np.sum(0.5 * i * x)
        return sum1 + sum2**2 + sum2**4


class DixonPriceFunction(TestProblem):
    """
    Dixon-Price function: f(x) = (x_1 - 1)^2 + sum(i * (2*x_i^2 - x_{i-1})^2)

    Global minimum: x_i = 2^(-(2^i - 2)/(2^i))
    """

    def __init__(self, n: int):
        # Calculate optimal solution
        optimal_x = np.zeros(n)
        for i in range(n):
            optimal_x[i] = 2.0**(-(2.0**(i + 1) - 2) / (2.0**(i + 1)))

        super().__init__(
            name=f"DixonPrice_{n}D",
            dimension=n,
            bounds=(-10.0, 10.0),
            optimal_solution=optimal_x,
            optimal_value=0.0,
            description=f"Dixon-Price function in {n} dimensions"
        )

    def evaluate(self, x: np.ndarray) -> float:
        """Evaluate the Dixon-Price function."""
        term1 = (x[0] - 1)**2
        i = np.arange(2, len(x) + 1)
        term2 = np.sum(i * (2 * x[1:]**2 - x[:-1])**2)
        return term1 + term2


class StyblinskiTangFunction(TestProblem):
    """
    Styblinski-Tang function: f(x) = 0.5 * sum(x_i^4 - 16*x_i^2 + 5*x_i)

    Global minimum: x* = (-2.903534, ..., -2.903534), f(x*) = -39.16599n
    """

    def __init__(self, n: int):
        super().__init__(
            name=f"StyblinskiTang_{n}D",
            dimension=n,
            bounds=(-5.0, 5.0),
            optimal_solution=np.full(n, -2.903534),
            optimal_value=-39.16599 * n,
            description=f"Styblinski-Tang function in {n} dimensions"
        )

    def evaluate(self, x: np.ndarray) -> float:
        """Evaluate the Styblinski-Tang function."""
        return 0.5 * np.sum(x**4 - 16 * x**2 + 5 * x)


class BealeFunction(TestProblem):
    """
    Beale function (2D only): Complex landscape with sharp peaks.

    Global minimum: x* = (3, 0.5), f(x*) = 0
    """

    def __init__(self):
        super().__init__(
            name="Beale_2D",
            dimension=2,
            bounds=(-4.5, 4.5),
            optimal_solution=np.array([3.0, 0.5]),
            optimal_value=0.0,
            description="Beale function in 2 dimensions"
        )

    def evaluate(self, x: np.ndarray) -> float:
        """Evaluate the Beale function."""
        if len(x) != 2:
            raise ValueError("Beale function is only defined for 2D")
        term1 = (1.5 - x[0] + x[0] * x[1])**2
        term2 = (2.25 - x[0] + x[0] * x[1]**2)**2
        term3 = (2.625 - x[0] + x[0] * x[1]**3)**2
        return term1 + term2 + term3


class GoldsteinPriceFunction(TestProblem):
    """
    Goldstein-Price function (2D only): Complex multimodal landscape.

    Global minimum: x* = (0, -1), f(x*) = 3
    """

    def __init__(self):
        super().__init__(
            name="GoldsteinPrice_2D",
            dimension=2,
            bounds=(-2.0, 2.0),
            optimal_solution=np.array([0.0, -1.0]),
            optimal_value=3.0,
            description="Goldstein-Price function in 2 dimensions"
        )

    def evaluate(self, x: np.ndarray) -> float:
        """Evaluate the Goldstein-Price function."""
        if len(x) != 2:
            raise ValueError("Goldstein-Price function is only defined for 2D")

        term1 = 1 + (x[0] + x[1] + 1)**2 * (19 - 14*x[0] + 3*x[0]**2 - 14*x[1] + 6*x[0]*x[1] + 3*x[1]**2)
        term2 = 30 + (2*x[0] - 3*x[1])**2 * (18 - 32*x[0] + 12*x[0]**2 + 48*x[1] - 36*x[0]*x[1] + 27*x[1]**2)
        return term1 * term2


class BoothFunction(TestProblem):
    """
    Booth function (2D only): f(x) = (x + 2y - 7)^2 + (2x + y - 5)^2

    Global minimum: x* = (1, 3), f(x*) = 0
    """

    def __init__(self):
        super().__init__(
            name="Booth_2D",
            dimension=2,
            bounds=(-10.0, 10.0),
            optimal_solution=np.array([1.0, 3.0]),
            optimal_value=0.0,
            description="Booth function in 2 dimensions"
        )

    def evaluate(self, x: np.ndarray) -> float:
        """Evaluate the Booth function."""
        if len(x) != 2:
            raise ValueError("Booth function is only defined for 2D")
        return (x[0] + 2*x[1] - 7)**2 + (2*x[0] + x[1] - 5)**2


class MatyasFunction(TestProblem):
    """
    Matyas function (2D only): f(x) = 0.26*(x^2 + y^2) - 0.48*x*y

    Global minimum: x* = (0, 0), f(x*) = 0
    """

    def __init__(self):
        super().__init__(
            name="Matyas_2D",
            dimension=2,
            bounds=(-10.0, 10.0),
            optimal_solution=np.array([0.0, 0.0]),
            optimal_value=0.0,
            description="Matyas function in 2 dimensions"
        )

    def evaluate(self, x: np.ndarray) -> float:
        """Evaluate the Matyas function."""
        if len(x) != 2:
            raise ValueError("Matyas function is only defined for 2D")
        return 0.26 * (x[0]**2 + x[1]**2) - 0.48 * x[0] * x[1]


class HimmelblauFunction(TestProblem):
    """
    Himmelblau function (2D only): f(x,y) = (x^2 + y - 11)^2 + (x + y^2 - 7)^2

    Has four global minima all with f(x*) = 0:
    - (3.0, 2.0)
    - (-2.805118, 3.131312)
    - (-3.779310, -3.283186)
    - (3.584428, -1.848126)
    """

    def __init__(self):
        super().__init__(
            name="Himmelblau_2D",
            dimension=2,
            bounds=(-5.0, 5.0),
            optimal_solution=np.array([3.0, 2.0]),  # One of four global minima
            optimal_value=0.0,
            description="Himmelblau function in 2 dimensions (4 global minima)"
        )

    def evaluate(self, x: np.ndarray) -> float:
        """Evaluate the Himmelblau function."""
        if len(x) != 2:
            raise ValueError("Himmelblau function is only defined for 2D")
        return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2


class EasomFunction(TestProblem):
    """
    Easom function (2D only): Sharp global minimum in a flat landscape.

    Global minimum: x* = (π, π), f(x*) = -1
    """

    def __init__(self):
        super().__init__(
            name="Easom_2D",
            dimension=2,
            bounds=(-100.0, 100.0),
            optimal_solution=np.array([np.pi, np.pi]),
            optimal_value=-1.0,
            description="Easom function in 2 dimensions"
        )

    def evaluate(self, x: np.ndarray) -> float:
        """Evaluate the Easom function."""
        if len(x) != 2:
            raise ValueError("Easom function is only defined for 2D")
        return -np.cos(x[0]) * np.cos(x[1]) * np.exp(-((x[0] - np.pi)**2 + (x[1] - np.pi)**2))


class CrossInTrayFunction(TestProblem):
    """
    Cross-in-tray function (2D only): Multiple global minima.

    Global minima: x* = (±1.34941, ±1.34941), f(x*) = -2.06261
    """

    def __init__(self):
        super().__init__(
            name="CrossInTray_2D",
            dimension=2,
            bounds=(-10.0, 10.0),
            optimal_solution=np.array([1.34941, 1.34941]),
            optimal_value=-2.06261,
            description="Cross-in-tray function in 2 dimensions"
        )

    def evaluate(self, x: np.ndarray) -> float:
        """Evaluate the Cross-in-tray function."""
        if len(x) != 2:
            raise ValueError("Cross-in-tray function is only defined for 2D")
        fact1 = np.sin(x[0]) * np.sin(x[1])
        fact2 = np.exp(abs(100 - np.sqrt(x[0]**2 + x[1]**2) / np.pi))
        return -0.0001 * (abs(fact1 * fact2) + 1)**0.1


class EggholderFunction(TestProblem):
    """
    Eggholder function (2D only): Highly complex landscape with many local minima.

    Global minimum: x* = (512, 404.2319), f(x*) ≈ -959.6407
    """

    def __init__(self):
        super().__init__(
            name="Eggholder_2D",
            dimension=2,
            bounds=(-512.0, 512.0),
            optimal_solution=np.array([512.0, 404.2319]),
            optimal_value=-959.6407,
            description="Eggholder function in 2 dimensions"
        )

    def evaluate(self, x: np.ndarray) -> float:
        """Evaluate the Eggholder function."""
        if len(x) != 2:
            raise ValueError("Eggholder function is only defined for 2D")
        term1 = -(x[1] + 47) * np.sin(np.sqrt(abs(x[0]/2 + (x[1] + 47))))
        term2 = -x[0] * np.sin(np.sqrt(abs(x[0] - (x[1] + 47))))
        return term1 + term2


class HolderTableFunction(TestProblem):
    """
    Holder table function (2D only): Four global minima.

    Global minima: x* = (±8.05502, ±9.66459), f(x*) = -19.2085
    """

    def __init__(self):
        super().__init__(
            name="HolderTable_2D",
            dimension=2,
            bounds=(-10.0, 10.0),
            optimal_solution=np.array([8.05502, 9.66459]),
            optimal_value=-19.2085,
            description="Holder table function in 2 dimensions"
        )

    def evaluate(self, x: np.ndarray) -> float:
        """Evaluate the Holder table function."""
        if len(x) != 2:
            raise ValueError("Holder table function is only defined for 2D")
        fact1 = np.sin(x[0]) * np.cos(x[1])
        fact2 = np.exp(abs(1 - np.sqrt(x[0]**2 + x[1]**2) / np.pi))
        return -abs(fact1 * fact2)


class McCormickFunction(TestProblem):
    """
    McCormick function (2D only): Asymmetric valley structure.

    Global minimum: x* = (-0.54719, -1.54719), f(x*) = -1.9133
    """

    def __init__(self):
        super().__init__(
            name="McCormick_2D",
            dimension=2,
            bounds=((-1.5, 4.0), (-3.0, 4.0)),
            optimal_solution=np.array([-0.54719, -1.54719]),
            optimal_value=-1.9133,
            description="McCormick function in 2 dimensions"
        )

    def evaluate(self, x: np.ndarray) -> float:
        """Evaluate the McCormick function."""
        if len(x) != 2:
            raise ValueError("McCormick function is only defined for 2D")
        return np.sin(x[0] + x[1]) + (x[0] - x[1])**2 - 1.5*x[0] + 2.5*x[1] + 1


class SchafferN2Function(TestProblem):
    """
    Schaffer N.2 function (2D only): Concentric circular waves.

    Global minimum: x* = (0, 0), f(x*) = 0
    """

    def __init__(self):
        super().__init__(
            name="SchafferN2_2D",
            dimension=2,
            bounds=(-100.0, 100.0),
            optimal_solution=np.array([0.0, 0.0]),
            optimal_value=0.0,
            description="Schaffer N.2 function in 2 dimensions"
        )

    def evaluate(self, x: np.ndarray) -> float:
        """Evaluate the Schaffer N.2 function."""
        if len(x) != 2:
            raise ValueError("Schaffer N.2 function is only defined for 2D")
        fact1 = np.sin(x[0]**2 - x[1]**2)**2 - 0.5
        fact2 = (1 + 0.001*(x[0]**2 + x[1]**2))**2
        return 0.5 + fact1 / fact2


class SchafferN4Function(TestProblem):
    """
    Schaffer N.4 function (2D only): Multiple local minima.

    Global minimum: x* = (0, ±1.25313), f(x*) = 0.292579
    """

    def __init__(self):
        super().__init__(
            name="SchafferN4_2D",
            dimension=2,
            bounds=(-100.0, 100.0),
            optimal_solution=np.array([0.0, 1.25313]),
            optimal_value=0.292579,
            description="Schaffer N.4 function in 2 dimensions"
        )

    def evaluate(self, x: np.ndarray) -> float:
        """Evaluate the Schaffer N.4 function."""
        if len(x) != 2:
            raise ValueError("Schaffer N.4 function is only defined for 2D")
        fact1 = np.cos(np.sin(abs(x[0]**2 - x[1]**2)))**2 - 0.5
        fact2 = (1 + 0.001*(x[0]**2 + x[1]**2))**2
        return 0.5 + fact1 / fact2


class ThreeHumpCamelFunction(TestProblem):
    """
    Three-hump camel function (2D only): Three local minima.

    Global minimum: x* = (0, 0), f(x*) = 0
    """

    def __init__(self):
        super().__init__(
            name="ThreeHumpCamel_2D",
            dimension=2,
            bounds=(-5.0, 5.0),
            optimal_solution=np.array([0.0, 0.0]),
            optimal_value=0.0,
            description="Three-hump camel function in 2 dimensions"
        )

    def evaluate(self, x: np.ndarray) -> float:
        """Evaluate the Three-hump camel function."""
        if len(x) != 2:
            raise ValueError("Three-hump camel function is only defined for 2D")
        return 2*x[0]**2 - 1.05*x[0]**4 + x[0]**6/6 + x[0]*x[1] + x[1]**2


class SixHumpCamelFunction(TestProblem):
    """
    Six-hump camel function (2D only): Six local minima, two global.

    Global minima: x* = (0.0898, -0.7126) and (-0.0898, 0.7126), f(x*) = -1.0316
    """

    def __init__(self):
        super().__init__(
            name="SixHumpCamel_2D",
            dimension=2,
            bounds=(-3.0, 3.0),
            optimal_solution=np.array([0.0898, -0.7126]),
            optimal_value=-1.0316,
            description="Six-hump camel function in 2 dimensions"
        )

    def evaluate(self, x: np.ndarray) -> float:
        """Evaluate the Six-hump camel function."""
        if len(x) != 2:
            raise ValueError("Six-hump camel function is only defined for 2D")
        term1 = (4 - 2.1*x[0]**2 + x[0]**4/3) * x[0]**2
        term2 = x[0] * x[1]
        term3 = (-4 + 4*x[1]**2) * x[1]**2
        return term1 + term2 + term3


# Registry of all available test problems
TEST_PROBLEM_REGISTRY = {
    'convex_quadratic': ConvexQuadratic,
    'sphere': SphereFunction,
    'rosenbrock': RosenbrockFunction,
    'rastrigin': RastriginFunction,
    'ackley': AckleyFunction,
    'griewank': GriewankFunction,
    'schwefel': SchwefelFunction,
    'levy': LevyFunction,
    'michalewicz': MichalewiczFunction,
    'zakharov': ZakharovFunction,
    'dixon_price': DixonPriceFunction,
    'styblinski_tang': StyblinskiTangFunction,
    'beale': BealeFunction,
    'goldstein_price': GoldsteinPriceFunction,
    'booth': BoothFunction,
    'matyas': MatyasFunction,
    'himmelblau': HimmelblauFunction,
    'easom': EasomFunction,
    'cross_in_tray': CrossInTrayFunction,
    'eggholder': EggholderFunction,
    'holder_table': HolderTableFunction,
    'mccormick': McCormickFunction,
    'schaffer_n2': SchafferN2Function,
    'schaffer_n4': SchafferN4Function,
    'three_hump_camel': ThreeHumpCamelFunction,
    'six_hump_camel': SixHumpCamelFunction,
}


def get_test_problems(dimensions: Optional[list] = None,
                      categories: Optional[list] = None) -> list:
    """
    Get a list of test problem instances.

    Args:
        dimensions: List of dimensions to test (None for 2D-only functions)
        categories: List of problem categories to include

    Returns:
        List of TestProblem instances
    """
    problems = []

    if dimensions is None:
        dimensions = [2, 5, 10, 20]

    # Add scalable problems for all requested dimensions
    scalable_problems = [
        'convex_quadratic', 'sphere', 'rosenbrock', 'rastrigin',
        'ackley', 'griewank', 'schwefel', 'levy', 'michalewicz',
        'zakharov', 'dixon_price', 'styblinski_tang'
    ]

    for problem_name in scalable_problems:
        problem_class = TEST_PROBLEM_REGISTRY[problem_name]
        for dim in dimensions:
            if problem_name == 'convex_quadratic':
                problems.append(problem_class(dim, seed=42))
            elif problem_name == 'michalewicz':
                if dim <= 10:  # Only test smaller dimensions for Michalewicz
                    problems.append(problem_class(dim))
            else:
                problems.append(problem_class(dim))

    # Add 2D-only problems
    two_d_problems = [
        'beale', 'goldstein_price', 'booth', 'matyas', 'himmelblau',
        'easom', 'cross_in_tray', 'eggholder', 'holder_table',
        'mccormick', 'schaffer_n2', 'schaffer_n4',
        'three_hump_camel', 'six_hump_camel'
    ]

    if 2 in dimensions:
        for problem_name in two_d_problems:
            problem_class = TEST_PROBLEM_REGISTRY[problem_name]
            problems.append(problem_class())

    return problems


def get_problem_categories():
    """Get categories of test problems."""
    return {
        'convex': ['convex_quadratic', 'sphere'],
        'unimodal': ['rosenbrock', 'zakharov', 'dixon_price'],
        'multimodal': ['rastrigin', 'ackley', 'griewank', 'schwefel',
                      'levy', 'michalewicz', 'styblinski_tang'],
        'two_dimensional': ['beale', 'goldstein_price', 'booth', 'matyas',
                           'himmelblau', 'easom', 'cross_in_tray', 'eggholder',
                           'holder_table', 'mccormick', 'schaffer_n2',
                           'schaffer_n4', 'three_hump_camel', 'six_hump_camel'],
        'separable': ['sphere', 'rastrigin', 'griewank'],
        'non_separable': ['rosenbrock', 'ackley', 'levy', 'zakharov'],
    }