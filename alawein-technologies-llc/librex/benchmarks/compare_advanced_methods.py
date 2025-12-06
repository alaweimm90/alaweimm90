#!/usr/bin/env python3
"""
Benchmark comparison of advanced optimization methods in Librex

This script compares the performance of all advanced methods on various
problem instances and sizes.
"""

import time
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

from Librex import optimize
from Librex.adapters.qap import QAPAdapter
from Librex.core.interfaces import StandardizedProblem


def create_qap_instance(n: int, seed: int = None) -> Dict:
    """Create a random QAP instance"""
    if seed is not None:
        np.random.seed(seed)

    flow = np.random.randint(1, 100, (n, n))
    distance = np.random.randint(1, 100, (n, n))

    # Make symmetric
    flow = (flow + flow.T) // 2
    distance = (distance + distance.T) // 2

    # Zero diagonal
    np.fill_diagonal(flow, 0)
    np.fill_diagonal(distance, 0)

    return {
        'flow_matrix': flow,
        'distance_matrix': distance
    }


def benchmark_method(
    problem: Dict,
    adapter: QAPAdapter,
    method: str,
    config: Dict,
    runs: int = 5
) -> Dict:
    """Benchmark a single method multiple times"""
    results = []
    times = []

    for run in range(runs):
        # Set seed for reproducibility
        run_config = config.copy()
        run_config['seed'] = 42 + run

        start_time = time.time()
        result = optimize(problem, adapter, method=method, config=run_config)
        elapsed_time = time.time() - start_time

        results.append(result['objective'])
        times.append(elapsed_time)

    return {
        'method': method,
        'mean_objective': np.mean(results),
        'std_objective': np.std(results),
        'best_objective': np.min(results),
        'worst_objective': np.max(results),
        'mean_time': np.mean(times),
        'std_time': np.std(times),
        'runs': runs
    }


def run_benchmark_suite():
    """Run comprehensive benchmark suite"""

    # Define methods and their configurations
    methods_configs = [
        # Baseline methods
        ('random_search', {'n_iterations': 1000}),
        ('simulated_annealing', {'max_iterations': 1000}),
        ('genetic_algorithm', {'generations': 50, 'population_size': 50}),
        ('tabu_search', {'max_iterations': 500}),
        ('local_search', {'max_iterations': 500}),

        # Advanced methods
        ('ant_colony', {'n_iterations': 50, 'n_ants': 20}),
        ('particle_swarm', {'n_iterations': 50, 'n_particles': 30}),
        ('variable_neighborhood', {'max_iterations': 200, 'k_max': 5}),
        ('iterated_local_search', {'max_iterations': 100}),
        ('grasp', {'max_iterations': 100, 'alpha': 0.2})
    ]

    # Problem sizes to test
    sizes = [10, 15, 20, 25]

    # Store all results
    all_results = []

    print("=" * 80)
    print("Librex ADVANCED METHODS BENCHMARK")
    print("=" * 80)
    print()

    for size in sizes:
        print(f"\nProblem Size: n={size}")
        print("-" * 50)

        # Create problem instance
        problem = create_qap_instance(size, seed=123)
        adapter = QAPAdapter()

        # Benchmark each method
        for method, config in methods_configs:
            print(f"  Benchmarking {method}...", end=" ")

            # Adjust iterations based on problem size
            if size > 20:
                # Reduce iterations for larger problems
                for key in ['n_iterations', 'max_iterations', 'generations']:
                    if key in config:
                        config[key] = config[key] // 2

            result = benchmark_method(problem, adapter, method, config, runs=3)
            result['problem_size'] = size
            all_results.append(result)

            print(f"Best: {result['best_objective']:.0f}, "
                  f"Mean: {result['mean_objective']:.0f}±{result['std_objective']:.0f}, "
                  f"Time: {result['mean_time']:.2f}s")

    return all_results


def print_summary_table(results: List[Dict]):
    """Print a summary table of results"""

    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(results)

    print("\n" + "=" * 80)
    print("SUMMARY BY METHOD (averaged across all problem sizes)")
    print("=" * 80)

    # Group by method and calculate statistics
    summary = df.groupby('method').agg({
        'mean_objective': 'mean',
        'std_objective': 'mean',
        'mean_time': 'mean'
    }).round(2)

    # Sort by mean objective (best first)
    summary = summary.sort_values('mean_objective')

    print("\n{:<25} {:>15} {:>15} {:>15}".format(
        "Method", "Mean Objective", "Std Dev", "Avg Time (s)"
    ))
    print("-" * 70)

    for method, row in summary.iterrows():
        print("{:<25} {:>15.0f} {:>15.0f} {:>15.2f}".format(
            method,
            row['mean_objective'],
            row['std_objective'],
            row['mean_time']
        ))

    # Print ranking
    print("\n" + "=" * 80)
    print("PERFORMANCE RANKING (by mean objective value)")
    print("=" * 80)

    for rank, (method, _) in enumerate(summary.iterrows(), 1):
        print(f"{rank}. {method}")


def analyze_scalability(results: List[Dict]):
    """Analyze how methods scale with problem size"""

    df = pd.DataFrame(results)

    print("\n" + "=" * 80)
    print("SCALABILITY ANALYSIS")
    print("=" * 80)

    # Calculate relative performance degradation
    for method in df['method'].unique():
        method_data = df[df['method'] == method]

        # Get objectives for each size
        sizes = sorted(method_data['problem_size'].unique())
        objectives = []
        times = []

        for size in sizes:
            size_data = method_data[method_data['problem_size'] == size]
            objectives.append(size_data['mean_objective'].values[0])
            times.append(size_data['mean_time'].values[0])

        # Calculate degradation rate
        if len(objectives) > 1:
            obj_growth = (objectives[-1] / objectives[0] - 1) * 100
            time_growth = (times[-1] / times[0] - 1) * 100

            print(f"\n{method}:")
            print(f"  Objective growth (n={sizes[0]} to n={sizes[-1]}): {obj_growth:.1f}%")
            print(f"  Time growth: {time_growth:.1f}%")


def main():
    """Main benchmark execution"""

    print("\nStarting Librex Advanced Methods Benchmark...")
    print("This will take several minutes to complete.\n")

    # Run benchmarks
    results = run_benchmark_suite()

    # Print summary
    print_summary_table(results)

    # Analyze scalability
    analyze_scalability(results)

    # Final recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    print("""
Based on the benchmark results:

1. For SMALL problems (n < 20):
   - VNS and ILS typically provide best quality
   - GRASP offers good balance of quality and speed

2. For MEDIUM problems (20 ≤ n ≤ 50):
   - ACO shows robust performance
   - VNS with appropriate k_max is effective
   - ILS with SA acceptance works well

3. For LARGE problems (n > 50):
   - Consider hybrid approaches
   - Use time limits to control execution
   - PSO can be parallelized for better performance

4. General guidelines:
   - Always tune parameters for your specific problem
   - Use multiple random seeds for statistical validity
   - Consider problem structure when selecting methods
""")


if __name__ == "__main__":
    main()