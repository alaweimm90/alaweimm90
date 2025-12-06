"""
Berkeley SciComp - Optimization Package
======================================
A comprehensive optimization library for scientific computing applications
including linear programming, nonlinear optimization, genetic algorithms,
and multi-objective optimization.
Author: Berkeley SciComp Team
Date: 2024
Modules:
--------
- unconstrained: Unconstrained optimization algorithms
- constrained: Constrained optimization methods
- global_optimization: Global optimization techniques
- linear_programming: Linear programming solvers
- multi_objective: Multi-objective optimization
- genetic_algorithms: Genetic and evolutionary algorithms
- utils: Utility functions and helpers
- visualization: Berkeley-themed plotting tools
"""
# Berkeley color scheme for consistent theming
BERKELEY_BLUE = '#003262'
CALIFORNIA_GOLD = '#FDB515'
BERKELEY_LIGHT_BLUE = '#3B7EA1'
# Import main classes and functions
from .unconstrained import *
from .constrained import *
from .global_optimization import *
from .linear_programming import *
from .multi_objective import *
from .genetic_algorithms import *
from .utils import *
# from .visualization import *  # TODO: Implement visualization module
# Package metadata
__version__ = '1.0.0'
__author__ = 'Berkeley SciComp Team'
__email__ = 'scicomp@berkeley.edu'
# Export key classes
__all__ = [
    # Unconstrained optimization
    'GradientDescent',
    'NewtonMethod',
    'QuasiNewton',
    'ConjugateGradient',
    'TrustRegion',
    # Constrained optimization
    'LagrangeMultipliers',
    'PenaltyMethod',
    'BarrierMethod',
    'AugmentedLagrangian',
    'SequentialQuadraticProgramming',
    # Global optimization
    'SimulatedAnnealing',
    'ParticleSwarmOptimization',
    'DifferentialEvolution',
    'BasinHopping',
    # Linear programming
    'SimplexMethod',
    'InteriorPointLP',
    'RevisedSimplex',
    # Multi-objective
    'NSGA2',
    'ParetoOptimization',
    'WeightedSum',
    'EpsilonConstraint',
    # Genetic algorithms
    'GeneticAlgorithm',
    'EvolutionStrategy',
    'GeneticProgramming',
    # Utilities
    'OptimizationProblem',
    'OptimizationResult',
    'BenchmarkFunctions',
    'ConvergenceAnalysis',
    # Visualization
    # 'OptimizationVisualizer'  # TODO: Implement visualization module
]
def demo():
    """
    Run a comprehensive demonstration of optimization capabilities.
    This function showcases various optimization algorithms on classic
    test problems with Berkeley-themed visualizations.
    """
    print("Berkeley SciComp - Optimization Demo")
    print("====================================")
    print()
    # Import required modules
    import numpy as np
    import matplotlib.pyplot as plt
    # Set Berkeley color scheme
    plt.style.use('default')
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=[BERKELEY_BLUE, CALIFORNIA_GOLD, BERKELEY_LIGHT_BLUE])
    # Demo 1: Unconstrained optimization
    print("1. Unconstrained Optimization - Rosenbrock Function")
    print("-" * 50)
    def rosenbrock(x):
        """Rosenbrock function (banana function)"""
        return 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2
    def rosenbrock_grad(x):
        """Gradient of Rosenbrock function"""
        return np.array([
            -400 * x[0] * (x[1] - x[0]**2) - 2 * (1 - x[0]),
            200 * (x[1] - x[0]**2)
        ])
    # Test gradient descent
    gd = GradientDescent(learning_rate=0.001, max_iterations=1000)
    result_gd = gd.minimize(rosenbrock, x0=np.array([-1.0, 1.0]), gradient=rosenbrock_grad)
    print(f"Gradient Descent Result:")
    print(f"  Solution: [{result_gd.x[0]:.6f}, {result_gd.x[1]:.6f}]")
    print(f"  Function Value: {result_gd.fun:.6e}")
    print(f"  Iterations: {result_gd.nit}")
    print()
    # Demo 2: Global optimization
    print("2. Global Optimization - Rastrigin Function")
    print("-" * 45)
    def rastrigin(x):
        """Rastrigin function (highly multimodal)"""
        A = 10
        n = len(x)
        return A * n + sum([(xi**2 - A * np.cos(2 * np.pi * xi)) for xi in x])
    # Test simulated annealing
    sa = SimulatedAnnealing(initial_temp=100.0, cooling_rate=0.95, min_temp=0.01)
    result_sa = sa.minimize(rastrigin, bounds=[(-5, 5), (-5, 5)], x0=np.array([3.0, 3.0]))
    print(f"Simulated Annealing Result:")
    print(f"  Solution: [{result_sa.x[0]:.6f}, {result_sa.x[1]:.6f}]")
    print(f"  Function Value: {result_sa.fun:.6e}")
    print(f"  Iterations: {result_sa.nit}")
    print()
    # Demo 3: Genetic Algorithm
    print("3. Genetic Algorithm - Sphere Function")
    print("-" * 38)
    def sphere(x):
        """Sphere function"""
        return sum(xi**2 for xi in x)
    ga = GeneticAlgorithm(population_size=50, generations=100, mutation_rate=0.1)
    result_ga = ga.minimize(sphere, bounds=[(-10, 10)] * 3)
    print(f"Genetic Algorithm Result:")
    print(f"  Solution: [{result_ga.x[0]:.6f}, {result_ga.x[1]:.6f}, {result_ga.x[2]:.6f}]")
    print(f"  Function Value: {result_ga.fun:.6e}")
    print(f"  Generations: {result_ga.nit}")
    print()
    print("Demo completed! Explore individual modules for more advanced features.")
if __name__ == "__main__":
    demo()