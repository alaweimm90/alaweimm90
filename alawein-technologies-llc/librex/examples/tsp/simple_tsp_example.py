#!/usr/bin/env python3
"""
Simple TSP Example

Demonstrates solving a small Traveling Salesman Problem.
"""

import numpy as np
from Librex import optimize
from Librex.adapters.tsp import TSPAdapter


def main():
    """Run simple TSP example"""

    print("="*60)
    print("Simple Traveling Salesman Problem Example")
    print("="*60)
    print()

    # Define cities in a 2D plane
    # Using a simple square layout for visualization
    cities = np.array([
        [0, 0],    # City 0: Origin
        [4, 0],    # City 1: East
        [4, 3],    # City 2: Northeast
        [0, 3],    # City 3: North
    ])

    print("Cities (coordinates):")
    for i, city in enumerate(cities):
        print(f"  City {i}: ({city[0]}, {city[1]})")
    print()

    # Create TSP problem
    problem = {'coordinates': cities}

    # Create adapter
    adapter = TSPAdapter()

    print("-"*60)
    print("Solving with different methods...")
    print("-"*60)
    print()

    # Solve with different methods
    methods = ['random_search', 'simulated_annealing', 'genetic_algorithm']

    results = {}

    for method in methods:
        print(f"Method: {method}")

        if method == 'random_search':
            config = {'iterations': 100, 'seed': 42}
        elif method == 'simulated_annealing':
            config = {'iterations': 500, 'seed': 42, 'initial_temp': 50.0}
        else:  # genetic_algorithm
            config = {'population_size': 30, 'generations': 20, 'seed': 42}

        result = optimize(problem, adapter, method=method, config=config)

        results[method] = result

        print(f"  Tour: {result['solution']}")
        print(f"  Tour length: {result['objective']:.4f}")
        print()

    print("="*60)
    print("Comparison")
    print("="*60)
    print()

    # Optimal tour for a square is the perimeter
    optimal_tour_length = 2 * (4 + 3)  # 14.0

    print(f"{'Method':<25} {'Tour Length':>15} {'vs Optimal':>15}")
    print("-"*60)

    for method in methods:
        length = results[method]['objective']
        gap = ((length - optimal_tour_length) / optimal_tour_length * 100)
        print(f"{method:<25} {length:>15.4f} {gap:>14.2f}%")

    print()
    print(f"Optimal tour length (perimeter): {optimal_tour_length:.4f}")
    print()

    # Show best tour
    best_method = min(results.keys(), key=lambda m: results[m]['objective'])
    best_tour = results[best_method]['solution']

    print("="*60)
    print("Best Tour Visualization")
    print("="*60)
    print()

    print(f"Best method: {best_method}")
    print(f"Tour sequence: {' → '.join(map(str, best_tour))} → {best_tour[0]}")
    print()

    # Calculate distances for each leg
    print("Tour breakdown:")
    total_distance = 0
    for i in range(len(best_tour)):
        from_city = best_tour[i]
        to_city = best_tour[(i + 1) % len(best_tour)]

        from_coords = cities[from_city]
        to_coords = cities[to_city]

        distance = np.sqrt(
            (from_coords[0] - to_coords[0])**2 +
            (from_coords[1] - to_coords[1])**2
        )

        total_distance += distance

        print(f"  {from_city} → {to_city}: {distance:.4f}")

    print(f"  Total: {total_distance:.4f}")
    print()


if __name__ == "__main__":
    main()
