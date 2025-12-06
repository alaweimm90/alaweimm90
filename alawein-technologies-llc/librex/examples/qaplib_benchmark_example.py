#!/usr/bin/env python3
"""
QAPLIB Benchmark Example

Demonstrates how to use the QAPLIB benchmark suite to evaluate
optimization methods on standard QAP instances.
"""

import numpy as np
from Librex import optimize
from Librex.adapters.qap import QAPAdapter
from Librex.benchmarks.qaplib import (
    load_qaplib_instance,
    list_qaplib_instances,
    get_qaplib_metadata,
    QAPLIBBenchmark,
    run_qaplib_benchmark,
    get_small_instances,
    QAPLIB_REGISTRY,
)


def demo_basic_loading():
    """Demonstrate basic instance loading"""
    print("=" * 80)
    print("BASIC INSTANCE LOADING")
    print("=" * 80)

    # Load a specific instance
    instance_data = load_qaplib_instance("nug12")
    print(f"\nLoaded nug12:")
    print(f"  Flow matrix shape: {instance_data['flow_matrix'].shape}")
    print(f"  Distance matrix shape: {instance_data['distance_matrix'].shape}")

    # Get metadata
    metadata = get_qaplib_metadata("nug12")
    print(f"  Optimal value: {metadata['optimal_value']}")
    print(f"  Problem class: {metadata['problem_class']}")

    # List available instances
    small_instances = list_qaplib_instances(filter_by_size=(10, 20))
    print(f"\nSmall instances (10 ≤ n ≤ 20): {len(small_instances)} instances")
    print(f"  First 10: {', '.join(small_instances[:10])}")

    # Filter by problem class
    real_world = list_qaplib_instances(filter_by_class="real-world")
    print(f"\nReal-world instances: {len(real_world)} instances")
    print(f"  Examples: {', '.join(real_world[:5])}")


def demo_Librex_integration():
    """Demonstrate integration with Librex optimization methods"""
    print("\n" + "=" * 80)
    print("Librex INTEGRATION")
    print("=" * 80)

    # Load QAPLIB instance
    instance_name = "chr12a"
    instance_data = load_qaplib_instance(instance_name)

    # Get metadata
    metadata = QAPLIB_REGISTRY[instance_name]
    print(f"\nInstance: {instance_name}")
    print(f"  Size: {metadata.size}")
    print(f"  Optimal value: {metadata.optimal_value}")
    print(f"  Description: {metadata.description}")

    # Create adapter
    adapter = QAPAdapter()

    # Run different optimization methods
    methods = ["random_search", "simulated_annealing", "genetic_algorithm"]
    results = {}

    print(f"\nOptimizing with different methods:")
    print("-" * 40)

    for method in methods:
        config = {
            "iterations": 1000 if method != "genetic_algorithm" else 20,
            "seed": 42
        }
        if method == "genetic_algorithm":
            config["population_size"] = 50
            config["generations"] = config["iterations"]
            del config["iterations"]

        result = optimize(instance_data, adapter, method=method, config=config)
        results[method] = result

        gap = 100 * (result["objective"] - metadata.optimal_value) / metadata.optimal_value
        print(f"{method:20} obj={result['objective']:6.0f} gap={gap:6.2f}%")

    # Find best method
    best_method = min(results.keys(), key=lambda m: results[m]["objective"])
    print(f"\nBest method: {best_method} with objective {results[best_method]['objective']}")


def demo_benchmarking():
    """Demonstrate comprehensive benchmarking"""
    print("\n" + "=" * 80)
    print("BENCHMARKING MULTIPLE METHODS")
    print("=" * 80)

    # Create benchmark runner
    benchmark = QAPLIBBenchmark(verbose=True)

    # Define wrapper functions for Librex methods
    def run_random_search(instance_data):
        adapter = QAPAdapter()
        config = {"iterations": 500, "seed": 42}
        result = optimize(instance_data, adapter, method="random_search", config=config)
        return result

    def run_simulated_annealing(instance_data):
        adapter = QAPAdapter()
        config = {"iterations": 1000, "seed": 42}
        result = optimize(instance_data, adapter, method="simulated_annealing", config=config)
        return result

    def run_genetic_algorithm(instance_data):
        adapter = QAPAdapter()
        config = {"population_size": 30, "generations": 20, "seed": 42}
        result = optimize(instance_data, adapter, method="genetic_algorithm", config=config)
        return result

    # Select test instances
    test_instances = ["nug12", "chr12a", "had12", "esc16a", "tai12a"]

    print(f"\nBenchmarking on instances: {', '.join(test_instances)}")

    # Compare methods
    methods = {
        "Random Search": run_random_search,
        "Simulated Annealing": run_simulated_annealing,
        "Genetic Algorithm": run_genetic_algorithm,
    }

    summaries = benchmark.compare_methods(methods, test_instances)

    # Additional analysis
    print("\n" + "=" * 80)
    print("DETAILED ANALYSIS")
    print("=" * 80)

    for method_name, summary in summaries.items():
        print(f"\n{method_name}:")
        print(f"  Average gap: {summary.avg_gap:.2f}%" if summary.avg_gap else "  No optimal values")
        print(f"  Solved optimally: {summary.solved_optimally}")
        print(f"  Total runtime: {summary.total_runtime:.2f}s")

        # Best and worst performance
        if summary.avg_gap is not None:
            gaps = [(r.instance_name, r.gap_percent)
                   for r in summary.results if r.gap_percent is not None]
            if gaps:
                best_instance, best_gap = min(gaps, key=lambda x: x[1])
                worst_instance, worst_gap = max(gaps, key=lambda x: x[1])
                print(f"  Best: {best_instance} (gap={best_gap:.2f}%)")
                print(f"  Worst: {worst_instance} (gap={worst_gap:.2f}%)")


def demo_quick_benchmark():
    """Demonstrate using the convenience function"""
    print("\n" + "=" * 80)
    print("QUICK BENCHMARK WITH CONVENIENCE FUNCTION")
    print("=" * 80)

    def my_custom_optimizer(instance_data):
        """Example custom optimizer"""
        n = len(instance_data["flow_matrix"])

        # Start with random solution
        solution = np.random.permutation(n)

        # Simple local search
        flow = instance_data["flow_matrix"]
        dist = instance_data["distance_matrix"]

        def compute_obj(perm):
            obj = 0
            for i in range(n):
                for j in range(n):
                    obj += flow[i, j] * dist[perm[i], perm[j]]
            return obj

        best_obj = compute_obj(solution)
        best_solution = solution.copy()

        # Do some swaps
        for _ in range(100):
            i, j = np.random.choice(n, 2, replace=False)
            solution[i], solution[j] = solution[j], solution[i]
            obj = compute_obj(solution)
            if obj < best_obj:
                best_obj = obj
                best_solution = solution.copy()
            else:
                solution[i], solution[j] = solution[j], solution[i]

        return {
            "solution": best_solution,
            "objective": best_obj,
            "iterations": 100
        }

    # Run benchmark
    print("\nRunning quick benchmark on small instances...")
    summary = run_qaplib_benchmark(
        method=my_custom_optimizer,
        instances=["nug12", "chr12a", "had12"],  # Just 3 instances for speed
        method_name="CustomLocalSearch",
        verbose=True
    )

    print(f"\nSummary:")
    print(f"  Method: {summary.method_name}")
    print(f"  Instances tested: {len(summary.results)}")
    if summary.avg_gap:
        print(f"  Average gap: {summary.avg_gap:.2f}%")
    print(f"  Total runtime: {summary.total_runtime:.2f}s")


def demo_instance_statistics():
    """Show statistics about the QAPLIB collection"""
    print("\n" + "=" * 80)
    print("QAPLIB COLLECTION STATISTICS")
    print("=" * 80)

    # Count by size ranges
    sizes = [inst.size for inst in QAPLIB_REGISTRY.values()]
    print(f"\nInstance sizes:")
    print(f"  Smallest: n={min(sizes)}")
    print(f"  Largest: n={max(sizes)}")
    print(f"  Total instances: {len(QAPLIB_REGISTRY)}")

    # Size distribution
    small = len([s for s in sizes if s <= 20])
    medium = len([s for s in sizes if 20 < s <= 50])
    large = len([s for s in sizes if 50 < s <= 100])
    very_large = len([s for s in sizes if s > 100])

    print(f"\nSize distribution:")
    print(f"  Small (n≤20): {small} instances")
    print(f"  Medium (20<n≤50): {medium} instances")
    print(f"  Large (50<n≤100): {large} instances")
    print(f"  Very large (n>100): {very_large} instances")

    # Problem classes
    classes = {}
    for inst in QAPLIB_REGISTRY.values():
        classes[inst.problem_class] = classes.get(inst.problem_class, 0) + 1

    print(f"\nProblem classes:")
    for cls, count in sorted(classes.items(), key=lambda x: -x[1]):
        print(f"  {cls:25} {count:3} instances")

    # Optimal values known
    optimal_known = sum(1 for inst in QAPLIB_REGISTRY.values()
                       if inst.optimal_value is not None)
    print(f"\nOptimal values:")
    print(f"  Known: {optimal_known} instances")
    print(f"  Unknown: {len(QAPLIB_REGISTRY) - optimal_known} instances")


def main():
    """Run all demos"""
    print("\n" + "=" * 80)
    print("QAPLIB BENCHMARK SUITE DEMONSTRATION")
    print("=" * 80)

    # Run demos
    demo_basic_loading()
    demo_Librex_integration()
    demo_benchmarking()
    demo_quick_benchmark()
    demo_instance_statistics()

    print("\n" + "=" * 80)
    print("DEMONSTRATION COMPLETE")
    print("=" * 80)
    print("\nThe QAPLIB benchmark suite is ready for use!")
    print("See the documentation for more details on using these benchmarks.")


if __name__ == "__main__":
    main()