"""
Benchmarking Framework Demonstration

Shows how to use the benchmarking framework to compare:
- Multiple solvers
- Different problem sizes
- Statistical analysis
- Performance profiling

Author: MEZAN Research Team
Date: 2025-11-18
"""

import logging
from atlas_core.benchmarking import (
    Benchmarker,
    create_qap_problem_generator,
    create_allocation_problem_generator,
)
from atlas_core.libria_solvers import create_libria_solver

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')


def main():
    """Run benchmarking demonstration"""
    print()
    print("=" * 70)
    print(" MEZAN BENCHMARKING FRAMEWORK DEMONSTRATION")
    print("=" * 70)
    print()

    # Create benchmarker
    benchmarker = Benchmarker(suite_name="MEZAN_Libria_Comparison")

    # Register problem generators
    print("Registering problem generators...")
    benchmarker.register_problem_generator("qap_tiny", create_qap_problem_generator(5))
    benchmarker.register_problem_generator("qap_small", create_qap_problem_generator(10))
    benchmarker.register_problem_generator("qap_medium", create_qap_problem_generator(15))
    benchmarker.register_problem_generator("alloc_small", create_allocation_problem_generator(4, 1000.0))
    benchmarker.register_problem_generator("alloc_medium", create_allocation_problem_generator(8, 2000.0))
    print(f"✓ {len(benchmarker.problem_generators)} problem generators registered")
    print()

    # Register solvers
    print("Registering solvers...")
    try:
        benchmarker.register_solver("QAPFlow", create_libria_solver("qapflow"))
        benchmarker.register_solver("AllocFlow", create_libria_solver("allocflow"))
        benchmarker.register_solver("MetaFlow", create_libria_solver("metaflow"))
        print(f"✓ {len(benchmarker.solvers)} solvers registered")
    except Exception as e:
        print(f"Warning: Some solvers may not be available: {e}")
    print()

    # Run benchmark suite
    print("=" * 70)
    print(" RUNNING BENCHMARK SUITE")
    print("=" * 70)
    print()
    print("Configuration:")
    print(f"  Solvers: {list(benchmarker.solvers.keys())}")
    print(f"  Problems: {list(benchmarker.problem_generators.keys())}")
    print(f"  Repetitions: 3")
    print(f"  Timeout: 10s per benchmark")
    print()

    suite = benchmarker.run_suite(
        repetitions=3,
        timeout=10.0,
    )

    # Display results
    print()
    print("=" * 70)
    print(" BENCHMARK RESULTS")
    print("=" * 70)
    print()

    # Generate and display report
    report = benchmarker.generate_report()
    print(report)

    # Additional analysis
    print()
    print("=" * 70)
    print(" DETAILED SOLVER COMPARISON")
    print("=" * 70)
    print()

    print("Average Time per Solver:")
    time_comparison = benchmarker.compare_solvers("time_elapsed")
    for solver, avg_time in sorted(time_comparison.items(), key=lambda x: x[1]):
        print(f"  {solver:15s}: {avg_time:8.4f}s")
    print()

    print("Average Objective Value per Solver:")
    obj_comparison = benchmarker.compare_solvers("objective_value")
    for solver, avg_obj in sorted(obj_comparison.items(), key=lambda x: x[1]):
        print(f"  {solver:15s}: {avg_obj:8.4f}")
    print()

    # Problem-specific analysis
    print("=" * 70)
    print(" PROBLEM-SPECIFIC ANALYSIS")
    print("=" * 70)
    print()

    by_problem = suite.get_by_problem()
    for problem_name, results in sorted(by_problem.items()):
        successful = [r for r in results if r.success]
        if not successful:
            continue

        print(f"Problem: {problem_name}")
        print(f"  Successful runs: {len(successful)}/{len(results)}")

        times = [r.time_elapsed for r in successful]
        import statistics
        print(f"  Time: {statistics.mean(times):.4f}s ± {statistics.stdev(times) if len(times) > 1 else 0:.4f}s")

        objectives = [r.objective_value for r in successful]
        print(f"  Objective: {statistics.mean(objectives):.4f} ± {statistics.stdev(objectives) if len(objectives) > 1 else 0:.4f}")
        print()

    # Export results
    print("=" * 70)
    print(" EXPORT RESULTS")
    print("=" * 70)
    print()

    output_file = "benchmark_results.json"
    suite.to_json(output_file)
    print(f"✓ Results exported to: {output_file}")
    print()

    print("=" * 70)
    print(" BENCHMARK DEMONSTRATION COMPLETE ✅")
    print("=" * 70)
    print()
    print("Key Features Demonstrated:")
    print("  ✓ Multi-solver comparison")
    print("  ✓ Problem suite execution")
    print("  ✓ Statistical analysis")
    print("  ✓ Performance profiling")
    print("  ✓ JSON export for CI/CD")
    print()


if __name__ == "__main__":
    main()
