#!/usr/bin/env python3
"""
Comprehensive demonstration of multi-objective optimization and constraint handling.

This example shows how to:
1. Define multi-objective optimization problems
2. Apply various constraint types
3. Use NSGA-II, NSGA-III, and MOEA/D algorithms
4. Evaluate results with quality indicators
5. Visualize Pareto fronts
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Import Librex multi-objective components
from Librex.multi_objective import (
    MultiObjectiveProblem,
    MultiObjectiveSolution,
    NSGA2Optimizer,
    NSGA3Optimizer,
    MOEADOptimizer,
    calculate_hypervolume,
    calculate_igd,
)
from Librex.multi_objective.moead import DecompositionMethod
from Librex.multi_objective.indicators import PerformanceMetrics

# Import constraint handling components
from Librex.constraints import (
    InequalityConstraint,
    BoxConstraint,
    LinearConstraint,
    ConstraintSet,
    StaticPenalty,
    AdaptivePenalty,
)


# =============================================================================
# Example 1: Simple Bi-objective Problem (ZDT1)
# =============================================================================

def example_zdt1():
    """
    ZDT1 test problem: Two objectives with Pareto-optimal front.

    minimize f1(x) = x1
    minimize f2(x) = g(x) * (1 - sqrt(x1/g(x)))
    where g(x) = 1 + 9 * mean(x[2:n])

    Pareto front: convex, f2 = 1 - sqrt(f1) for f1 in [0,1]
    """
    print("\n" + "="*60)
    print("Example 1: ZDT1 Bi-objective Problem")
    print("="*60)

    # Define objectives
    def f1(x):
        return x[0]

    def f2(x):
        g = 1 + 9 * np.mean(x[1:])
        return g * (1 - np.sqrt(x[0] / g))

    # Create problem
    problem = MultiObjectiveProblem(
        objectives=[f1, f2],
        n_objectives=2,
        n_variables=30,
        bounds=(np.zeros(30), np.ones(30))
    )

    # Run NSGA-II
    print("\nRunning NSGA-II...")
    nsga2 = NSGA2Optimizer(
        problem,
        population_size=100,
        n_generations=100,
        seed=42
    )
    pareto_front_nsga2 = nsga2.optimize()
    print(f"NSGA-II found {pareto_front_nsga2.size()} solutions")

    # Run MOEA/D
    print("\nRunning MOEA/D...")
    moead = MOEADOptimizer(
        problem,
        population_size=100,
        n_generations=100,
        decomposition=DecompositionMethod.TCHEBYCHEFF,
        seed=42
    )
    pareto_front_moead = moead.optimize()
    print(f"MOEA/D found {pareto_front_moead.size()} solutions")

    # Calculate quality indicators
    true_pareto_front = generate_zdt1_true_front()
    metrics = PerformanceMetrics(
        reference_set=true_pareto_front,
        reference_point=np.array([1.1, 1.1])
    )

    nsga2_metrics = metrics.evaluate(pareto_front_nsga2.get_objectives())
    moead_metrics = metrics.evaluate(pareto_front_moead.get_objectives())

    print("\nQuality Indicators:")
    print(f"NSGA-II - IGD: {nsga2_metrics['igd']:.4f}, HV: {nsga2_metrics['hypervolume']:.4f}")
    print(f"MOEA/D  - IGD: {moead_metrics['igd']:.4f}, HV: {moead_metrics['hypervolume']:.4f}")

    # Plot results
    plot_2d_pareto_fronts(
        [pareto_front_nsga2.get_objectives(), pareto_front_moead.get_objectives()],
        ["NSGA-II", "MOEA/D"],
        true_front=true_pareto_front,
        title="ZDT1 Problem - Pareto Fronts"
    )


# =============================================================================
# Example 2: Three-objective Problem with Constraints (DTLZ2)
# =============================================================================

def example_dtlz2_constrained():
    """
    Modified DTLZ2 with constraints: Three objectives on unit sphere.

    Added constraints:
    - Box constraints: 0 <= x_i <= 1
    - Linear constraint: sum(x) <= n/2
    - Nonlinear constraint: ||x||^2 <= n/2
    """
    print("\n" + "="*60)
    print("Example 2: Constrained Three-objective Problem (DTLZ2)")
    print("="*60)

    n_obj = 3
    n_var = 12  # n_obj - 1 + k, where k = 10

    # Define objectives for DTLZ2
    def dtlz2_objectives(x):
        k = n_var - n_obj + 1
        g = np.sum((x[n_obj-1:] - 0.5) ** 2)

        objectives = np.zeros(n_obj)
        for i in range(n_obj):
            f = 1 + g
            for j in range(n_obj - i - 1):
                f *= np.cos(x[j] * np.pi / 2)
            if i > 0:
                f *= np.sin(x[n_obj - i - 1] * np.pi / 2)
            objectives[i] = f
        return objectives

    # Create individual objective functions
    def make_obj(idx):
        return lambda x: dtlz2_objectives(x)[idx]

    objectives = [make_obj(i) for i in range(n_obj)]

    # Define constraints
    constraints = [
        # Linear constraint: sum(x) <= n/2
        LinearConstraint(
            A=np.ones((1, n_var)),
            b=np.array([n_var / 2]),
            constraint_type="inequality"
        ),
        # Nonlinear constraint: ||x||^2 <= n/2
        InequalityConstraint(
            lambda x: np.sum(x ** 2) - n_var / 2
        )
    ]

    # Create problem
    problem = MultiObjectiveProblem(
        objectives=objectives,
        n_objectives=n_obj,
        n_variables=n_var,
        bounds=(np.zeros(n_var), np.ones(n_var)),
        constraints=constraints,
        reference_point=np.ones(n_obj) * 2.0
    )

    # Run NSGA-III (better for many objectives)
    print("\nRunning NSGA-III with constraints...")
    nsga3 = NSGA3Optimizer(
        problem,
        n_generations=100,
        n_partitions=12,
        seed=42
    )
    pareto_front = nsga3.optimize()
    print(f"NSGA-III found {pareto_front.size()} feasible solutions")

    # Evaluate constraint satisfaction
    n_feasible = sum(1 for sol in pareto_front.solutions if sol.constraint_violation == 0)
    print(f"Feasible solutions: {n_feasible}/{pareto_front.size()}")

    # Calculate hypervolume
    if pareto_front.size() > 0:
        hv = calculate_hypervolume(
            pareto_front.get_objectives(),
            problem.reference_point
        )
        print(f"Hypervolume: {hv:.4f}")

    # Plot 3D Pareto front
    plot_3d_pareto_front(
        pareto_front.get_objectives(),
        title="DTLZ2 with Constraints - 3D Pareto Front"
    )


# =============================================================================
# Example 3: Engineering Design Problem with Mixed Constraints
# =============================================================================

def example_engineering_design():
    """
    Engineering design optimization: Welded beam design problem.

    Minimize:
    - f1: Cost of welding
    - f2: Deflection of beam

    Subject to:
    - Shear stress constraint
    - Bending stress constraint
    - Buckling load constraint
    - Deflection constraint
    - Side constraints (bounds)
    """
    print("\n" + "="*60)
    print("Example 3: Welded Beam Design (Engineering Problem)")
    print("="*60)

    # Problem parameters
    P = 6000  # Applied load (lb)
    L = 14    # Length of beam (in)
    E = 30e6  # Modulus of elasticity (psi)
    G = 12e6  # Shear modulus (psi)

    # Design variables: x = [h, l, t, b]
    # h: weld thickness, l: weld length, t: beam thickness, b: beam width

    # Objective 1: Cost
    def f1_cost(x):
        h, l, t, b = x
        return 1.10471 * h**2 * l + 0.04811 * t * b * (14 + l)

    # Objective 2: Deflection
    def f2_deflection(x):
        h, l, t, b = x
        return 4 * P * L**3 / (E * t**3 * b)

    # Helper functions for constraints
    def tau(x):
        h, l, t, b = x
        tau_prime = P / (np.sqrt(2) * h * l)
        M = P * (L + l/2)
        R = np.sqrt(l**2/4 + (h + t)**2/4)
        J = 2 * np.sqrt(2) * h * l * (l**2/12 + (h + t)**2/4)
        tau_double = M * R / J
        return np.sqrt(tau_prime**2 + 2*tau_prime*tau_double*l/(2*R) + tau_double**2)

    def sigma(x):
        h, l, t, b = x
        return 6 * P * L / (b * t**2)

    def Pc(x):
        h, l, t, b = x
        return 4.013 * E * np.sqrt(t**2 * b**6 / 36) / L**2 * (1 - t/(2*L) * np.sqrt(E/(4*G)))

    # Define constraints
    constraints = [
        # g1: Shear stress
        InequalityConstraint(lambda x: tau(x) - 13600),
        # g2: Bending stress
        InequalityConstraint(lambda x: sigma(x) - 30000),
        # g3: h <= b
        InequalityConstraint(lambda x: x[0] - x[3]),
        # g4: Buckling load
        InequalityConstraint(lambda x: P - Pc(x)),
        # g5: Deflection
        InequalityConstraint(lambda x: f2_deflection(x) - 0.25),
    ]

    # Variable bounds
    bounds = (
        np.array([0.1, 0.1, 0.1, 0.1]),  # Lower bounds
        np.array([2.0, 10.0, 10.0, 2.0]) # Upper bounds
    )

    # Create problem
    problem = MultiObjectiveProblem(
        objectives=[f1_cost, f2_deflection],
        n_objectives=2,
        n_variables=4,
        bounds=bounds,
        constraints=constraints,
        minimize=[True, True],
        reference_point=np.array([50, 0.1])
    )

    # Run NSGA-II with adaptive penalty
    print("\nOptimizing welded beam design...")
    optimizer = NSGA2Optimizer(
        problem,
        population_size=100,
        n_generations=200,
        seed=42
    )
    pareto_front = optimizer.optimize()

    print(f"Found {pareto_front.size()} Pareto-optimal designs")

    # Display some solutions
    if pareto_front.size() > 0:
        print("\nSample Pareto-optimal designs:")
        print("Cost ($) | Deflection (in) | h     | l     | t     | b")
        print("-" * 60)

        # Sort by first objective
        solutions = sorted(pareto_front.solutions, key=lambda s: s.objectives[0])

        # Show min cost, min deflection, and compromise solutions
        indices = [0, len(solutions)//2, -1]
        for idx in indices:
            sol = solutions[idx]
            print(f"{sol.objectives[0]:8.2f} | {sol.objectives[1]:15.6f} | "
                  f"{sol.variables[0]:.3f} | {sol.variables[1]:.3f} | "
                  f"{sol.variables[2]:.3f} | {sol.variables[3]:.3f}")

    # Plot Pareto front
    plot_2d_pareto_fronts(
        [pareto_front.get_objectives()],
        ["NSGA-II"],
        title="Welded Beam Design - Cost vs Deflection",
        xlabel="Cost ($)",
        ylabel="Deflection (in)"
    )


# =============================================================================
# Example 4: Many-objective Problem (5 objectives)
# =============================================================================

def example_many_objective():
    """
    DTLZ7 problem with 5 objectives to demonstrate many-objective optimization.
    """
    print("\n" + "="*60)
    print("Example 4: Many-objective Optimization (5 objectives)")
    print("="*60)

    n_obj = 5
    n_var = n_obj + 4  # Typically k=5 for DTLZ problems

    # Define DTLZ7 objectives
    def dtlz7_objectives(x):
        objectives = x[:n_obj-1].copy()

        g = 1 + 9 * np.mean(x[n_obj-1:])
        h = n_obj - np.sum(objectives / (1 + g) * (1 + np.sin(3 * np.pi * objectives)))

        objectives = np.append(objectives, (1 + g) * h)
        return objectives

    # Create individual objective functions
    objectives = []
    for i in range(n_obj):
        objectives.append(lambda x, idx=i: dtlz7_objectives(x)[idx])

    # Create problem
    problem = MultiObjectiveProblem(
        objectives=objectives,
        n_objectives=n_obj,
        n_variables=n_var,
        bounds=(np.zeros(n_var), np.ones(n_var)),
        reference_point=np.ones(n_obj) * 10
    )

    # Compare NSGA-III and MOEA/D for many objectives
    print("\nComparing algorithms for many-objective optimization...")

    # Run NSGA-III
    print("Running NSGA-III...")
    nsga3 = NSGA3Optimizer(
        problem,
        n_generations=150,
        n_partitions=6,  # Generates structured reference points
        seed=42
    )
    front_nsga3 = nsga3.optimize()
    print(f"NSGA-III found {front_nsga3.size()} solutions")

    # Run MOEA/D
    print("Running MOEA/D...")
    moead = MOEADOptimizer(
        problem,
        population_size=120,
        n_generations=150,
        decomposition=DecompositionMethod.PBI,
        pbi_theta=5.0,
        seed=42
    )
    front_moead = moead.optimize()
    print(f"MOEA/D found {front_moead.size()} solutions")

    # Calculate hypervolume
    if front_nsga3.size() > 0:
        hv_nsga3 = calculate_hypervolume(
            front_nsga3.get_objectives(),
            problem.reference_point
        )
        print(f"NSGA-III Hypervolume: {hv_nsga3:.2f}")

    if front_moead.size() > 0:
        hv_moead = calculate_hypervolume(
            front_moead.get_objectives(),
            problem.reference_point
        )
        print(f"MOEA/D Hypervolume: {hv_moead:.2f}")

    # Visualize using parallel coordinates
    plot_parallel_coordinates(
        [front_nsga3.get_objectives()[:50], front_moead.get_objectives()[:50]],
        ["NSGA-III", "MOEA/D"],
        n_obj
    )


# =============================================================================
# Helper Functions for True Pareto Fronts and Visualization
# =============================================================================

def generate_zdt1_true_front(n_points=100):
    """Generate true Pareto front for ZDT1."""
    f1 = np.linspace(0, 1, n_points)
    f2 = 1 - np.sqrt(f1)
    return np.column_stack([f1, f2])


def plot_2d_pareto_fronts(fronts, labels, true_front=None, title="Pareto Fronts",
                          xlabel="f1", ylabel="f2"):
    """Plot 2D Pareto fronts."""
    plt.figure(figsize=(10, 6))

    # Plot true front if available
    if true_front is not None:
        plt.plot(true_front[:, 0], true_front[:, 1], 'k-',
                 label='True Front', linewidth=2, alpha=0.5)

    # Plot approximated fronts
    colors = ['b', 'r', 'g', 'm', 'c', 'y']
    for i, (front, label) in enumerate(zip(fronts, labels)):
        if len(front) > 0:
            plt.scatter(front[:, 0], front[:, 1],
                       c=colors[i % len(colors)],
                       label=label, alpha=0.7, s=30)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_3d_pareto_front(front, title="3D Pareto Front"):
    """Plot 3D Pareto front."""
    if len(front) == 0:
        print("No solutions to plot")
        return

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(front[:, 0], front[:, 1], front[:, 2],
               c='b', marker='o', alpha=0.6, s=30)

    ax.set_xlabel('f1')
    ax.set_ylabel('f2')
    ax.set_zlabel('f3')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    plt.show()


def plot_parallel_coordinates(fronts, labels, n_obj):
    """Plot parallel coordinates for many-objective visualization."""
    fig, axes = plt.subplots(1, n_obj-1, figsize=(15, 5), sharey=False)

    colors = ['b', 'r']

    for front_idx, (front, label) in enumerate(zip(fronts, labels)):
        if len(front) == 0:
            continue

        # Normalize objectives to [0, 1]
        front_norm = (front - front.min(axis=0)) / (front.max(axis=0) - front.min(axis=0) + 1e-10)

        # Plot lines connecting objectives
        for sol in front_norm:
            for i in range(n_obj - 1):
                axes[i].plot([i, i+1], [sol[i], sol[i+1]],
                           color=colors[front_idx], alpha=0.3, linewidth=1)

    # Set labels
    for i in range(n_obj - 1):
        axes[i].set_xlim([i, i+1])
        axes[i].set_ylim([0, 1])
        axes[i].set_xticks([i, i+1])
        axes[i].set_xticklabels([f'f{i+1}', f'f{i+2}'])
        axes[i].grid(True, alpha=0.3)

    plt.suptitle(f"Parallel Coordinates - {', '.join(labels)}")
    plt.tight_layout()
    plt.show()


# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Librex Multi-Objective Optimization Demo")
    print("=" * 60)

    # Run all examples
    example_zdt1()
    example_dtlz2_constrained()
    example_engineering_design()
    example_many_objective()

    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)