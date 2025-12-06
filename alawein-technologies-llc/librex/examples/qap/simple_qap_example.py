#!/usr/bin/env python3
"""
Simple QAP Example

Demonstrates basic usage of Librex for solving a small
Quadratic Assignment Problem.
"""

import numpy as np
from Librex import optimize
from Librex.adapters.qap import QAPAdapter


def main():
    """Run simple QAP example"""

    print("="*60)
    print("Simple Quadratic Assignment Problem Example")
    print("="*60)
    print()

    # Define a small QAP instance
    # Flow matrix: Represents flow between facilities
    flow_matrix = np.array([
        [0, 5, 2],
        [5, 0, 3],
        [2, 3, 0]
    ])

    # Distance matrix: Represents distance between locations
    distance_matrix = np.array([
        [0, 8, 15],
        [8, 0, 13],
        [15, 13, 0]
    ])

    print("Flow Matrix (Facility interactions):")
    print(flow_matrix)
    print()

    print("Distance Matrix (Location distances):")
    print(distance_matrix)
    print()

    # Create QAP problem
    problem = {
        'flow_matrix': flow_matrix,
        'distance_matrix': distance_matrix
    }

    # Create adapter
    adapter = QAPAdapter()

    print("-"*60)
    print("Solving with different methods...")
    print("-"*60)
    print()

    # Solve with different methods
    methods = ['random_search', 'simulated_annealing', 'local_search',
               'genetic_algorithm', 'tabu_search']

    results = {}

    for method in methods:
        print(f"Method: {method}")

        # Set appropriate config for each method
        if method == 'random_search':
            config = {'iterations': 500, 'seed': 42}
        elif method == 'simulated_annealing':
            config = {'iterations': 1000, 'seed': 42}
        elif method == 'local_search':
            config = {'max_iterations': 500, 'restarts': 5, 'seed': 42}
        elif method == 'genetic_algorithm':
            config = {'population_size': 50, 'generations': 20, 'seed': 42}
        else:  # tabu_search
            config = {'iterations': 500, 'tabu_tenure': 10, 'seed': 42}

        result = optimize(problem, adapter, method=method, config=config)

        results[method] = result

        print(f"  Solution: {result['solution']}")
        print(f"  Objective: {result['objective']:.2f}")
        print(f"  Iterations: {result['iterations']}")
        print()

    print("="*60)
    print("Comparison Summary")
    print("="*60)
    print()

    # Find best result
    best_method = min(results.keys(), key=lambda m: results[m]['objective'])
    best_objective = results[best_method]['objective']

    print(f"{'Method':<25} {'Objective':>15} {'Gap from Best':>15}")
    print("-"*60)

    for method in methods:
        obj = results[method]['objective']
        gap = ((obj - best_objective) / best_objective * 100) if best_objective > 0 else 0
        marker = " *" if method == best_method else ""
        print(f"{method:<25} {obj:>15.2f} {gap:>14.2f}%{marker}")

    print()
    print(f"* Best solution found by: {best_method}")
    print()

    # Interpret solution
    print("="*60)
    print("Solution Interpretation")
    print("="*60)
    print()

    best_solution = results[best_method]['solution']
    print("Assignment (Facility → Location):")
    for facility, location in enumerate(best_solution):
        print(f"  Facility {facility} → Location {location}")

    print()
    print("Total cost: Sum of (flow between facilities) × (distance between assigned locations)")
    print(f"Total cost: {best_objective:.2f}")
    print()


if __name__ == "__main__":
    main()
