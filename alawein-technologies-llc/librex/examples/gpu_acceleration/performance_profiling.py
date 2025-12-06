#!/usr/bin/env python3
"""
Performance Profiling Demo

Demonstrates performance profiling and optimization analysis.
"""

import numpy as np
import sys
import os
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# Add Librex to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from Librex.acceleration.profiling import PerformanceProfiler
from Librex.acceleration.parallel_eval import ParallelEvaluator
from Librex.acceleration.memory_efficient import MemoryEfficientOptimizer, MemoryConfig
from Librex.acceleration.gpu_methods import GPUGeneticAlgorithm, GPUParticleSwarm
from Librex.acceleration.gpu_methods.gpu_genetic import GeneticConfig
from Librex.acceleration.gpu_methods.gpu_pso import PSOConfig


class GriewankProblem:
    """Griewank function - many regularly distributed local minima"""

    def __init__(self, dimension: int = 10):
        self.dimension = dimension
        self.bounds = (-600, 600)

    def objective_function(self, x):
        """Griewank function"""
        sum_sq = np.sum(x**2) / 4000
        prod_cos = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))
        return 1 + sum_sq - prod_cos


def profile_single_method():
    """Profile a single optimization method"""
    print("\n" + "=" * 80)
    print("SINGLE METHOD PROFILING")
    print("=" * 80)

    problem = GriewankProblem(dimension=50)
    print(f"\nProblem: {problem.__class__.__name__} (dim={problem.dimension})")

    # Create profiler
    profiler = PerformanceProfiler(backend='jax', verbose=True)

    # Method to profile
    ga = GPUGeneticAlgorithm('jax')
    config = GeneticConfig(
        population_size=200,
        n_generations=100,
        mutation_rate=0.02
    )

    print("\nProfiling Genetic Algorithm...")
    print("-" * 60)

    # Profile with warmup
    profile_result = profiler.profile_optimization(
        ga.optimize,
        problem,
        config,
        warmup_runs=2
    )

    # Display results
    profile = profile_result['profile']

    print("\n📊 PROFILING RESULTS:")
    print("-" * 60)
    print(f"Total Runtime: {profile.total_runtime:.2f}s")
    print(f"Initialization: {profile.initialization_time:.3f}s")
    print(f"Evaluation Time: {profile.evaluation_time:.2f}s ({profile.evaluation_time/profile.total_runtime*100:.1f}%)")
    print(f"Selection Time: {profile.selection_time:.2f}s ({profile.selection_time/profile.total_runtime*100:.1f}%)")
    print(f"Crossover Time: {profile.crossover_time:.2f}s ({profile.crossover_time/profile.total_runtime*100:.1f}%)")
    print(f"Mutation Time: {profile.mutation_time:.2f}s ({profile.mutation_time/profile.total_runtime*100:.1f}%)")

    print(f"\n💻 PERFORMANCE METRICS:")
    print(f"Evaluations/sec: {profile.evaluations_per_second:.0f}")
    print(f"Generations/sec: {profile.generations_per_second:.2f}")

    print(f"\n🎯 BOTTLENECK ANALYSIS:")
    print(f"Primary Bottleneck: {profile.primary_bottleneck}")
    print(f"Bottleneck Impact: {profile.bottleneck_percentage:.1f}%")

    if profile.gpu_utilization > 0:
        print(f"\n🖥️ GPU UTILIZATION:")
        print(f"GPU Usage: {profile.gpu_utilization:.1f}%")
        print(f"GPU Memory: {profile.gpu_memory_mb:.0f} MB")
        print(f"Compute Efficiency: {profile.gpu_compute_efficiency:.1%}")

    print(f"\n💾 MEMORY USAGE:")
    print(f"Peak Memory: {profile.peak_memory_mb:.0f} MB")
    print(f"Avg Memory: {profile.avg_memory_mb:.0f} MB")
    print(f"Memory Efficiency: {profile.memory_efficiency:.1%}")

    # Generate report
    report_path = "/tmp/Librex_profile_report.txt"
    report = profiler.generate_report(profile_result, report_path)
    print(f"\n📄 Detailed report saved to: {report_path}")

    return profile_result


def compare_methods():
    """Compare performance of multiple optimization methods"""
    print("\n" + "=" * 80)
    print("METHOD COMPARISON PROFILING")
    print("=" * 80)

    problem = GriewankProblem(dimension=30)
    profiler = PerformanceProfiler(backend='jax', verbose=False)

    # Methods to compare
    methods = []

    # Genetic Algorithm
    ga = GPUGeneticAlgorithm('jax')
    ga_config = GeneticConfig(population_size=100, n_generations=50)
    methods.append(('Genetic Algorithm', ga.optimize, ga_config))

    # Particle Swarm
    pso = GPUParticleSwarm('jax')
    pso_config = PSOConfig(n_particles=100, n_iterations=50)
    methods.append(('Particle Swarm', pso.optimize, pso_config))

    print(f"\nComparing {len(methods)} methods on {problem.__class__.__name__}")
    print("-" * 60)

    # Extract method functions and configs
    method_funcs = [m[1] for m in methods]
    method_names = [m[0] for m in methods]
    config = methods[0][2]  # Use first config (they're similar enough)

    # Compare methods
    comparison_result = profiler.compare_methods(
        method_funcs,
        problem,
        config.__dict__
    )

    # Display comparison
    print("\n📊 PERFORMANCE COMPARISON:")
    print(f"{'Method':<20} {'Runtime (s)':<15} {'Memory (MB)':<15} {'Throughput':<15}")
    print("-" * 65)

    for method_name in method_names:
        if method_name in comparison_result['individual_results']:
            result = comparison_result['individual_results'][method_name]
            profile = result['profile']
            print(f"{method_name:<20} {profile.total_runtime:<15.2f} "
                  f"{profile.peak_memory_mb:<15.0f} {profile.evaluations_per_second:<15.0f}")

    print(f"\n🏆 Best Method: {comparison_result['comparison']['best_method']}")

    # Plot comparison
    create_comparison_plot(comparison_result, method_names)


def test_parallel_evaluation():
    """Test and profile parallel evaluation strategies"""
    print("\n" + "=" * 80)
    print("PARALLEL EVALUATION PROFILING")
    print("=" * 80)

    # Create test objective function
    def expensive_objective(x):
        """Simulate expensive objective function"""
        import time
        time.sleep(0.001)  # Simulate 1ms computation
        return np.sum(x**2)

    print("\nBenchmarking parallel evaluation strategies...")
    print("-" * 60)

    # Test different backends
    backends = ['cpu', 'thread', 'gpu_jax']
    problem_sizes = [10, 50, 100, 500]

    for backend in backends:
        print(f"\n--- {backend.upper()} Backend ---")

        try:
            evaluator = ParallelEvaluator(backend=backend)
            print(f"Workers: {evaluator.n_workers}")

            # Benchmark
            benchmark_result = evaluator.benchmark(
                expensive_objective,
                problem_sizes
            )

            print(f"Average Speedup: {benchmark_result['average_speedup']:.2f}x")
            print(f"Average Efficiency: {benchmark_result['average_efficiency']:.1%}")

            # Display per-size results
            print(f"\n{'Size':<10} {'Speedup':<10} {'Throughput':<15}")
            print("-" * 35)
            for r in benchmark_result['results']:
                print(f"{r['problem_size']:<10} {r['speedup']:<10.2f} {r['throughput']:<15.0f}")

        except Exception as e:
            print(f"Backend {backend} skipped: {e}")


def test_memory_efficiency():
    """Test memory-efficient optimization"""
    print("\n" + "=" * 80)
    print("MEMORY-EFFICIENT OPTIMIZATION PROFILING")
    print("=" * 80)

    # Large problem to stress memory
    problem = GriewankProblem(dimension=100)

    print(f"\nProblem: Large {problem.__class__.__name__} (dim={problem.dimension})")

    # Configure memory-efficient optimizer
    mem_config = MemoryConfig(
        chunk_size=500,
        enable_gradient_checkpointing=True,
        enable_memory_mapping=False,
        max_memory_mb=2048,
        enable_streaming=True,
    )

    mem_optimizer = MemoryEfficientOptimizer(backend='jax', config=mem_config)

    # Test different population sizes
    population_sizes = [1000, 5000, 10000]

    print("\n📊 MEMORY SCALING TEST:")
    print(f"{'Population':<15} {'Runtime (s)':<15} {'Peak Mem (MB)':<15} {'Obj Value':<15}")
    print("-" * 60)

    for pop_size in population_sizes:
        config = {
            'population_size': pop_size,
            'n_generations': 20,
            'mutation_rate': 0.02,
        }

        try:
            result = mem_optimizer.chunked_genetic_algorithm(problem, config)

            peak_mem = result['metadata']['max_memory_mb']
            runtime = result['runtime']
            objective = result['objective']

            print(f"{pop_size:<15} {runtime:<15.2f} {peak_mem:<15.0f} {objective:<15.6f}")

        except Exception as e:
            print(f"{pop_size:<15} {'Error':<15} {str(e)[:30]}")

    # Cleanup
    mem_optimizer.cleanup()


def create_comparison_plot(comparison_result, method_names):
    """Create visualization of method comparison"""
    try:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('Optimization Method Performance Comparison', fontsize=16)

        comp = comparison_result['comparison']

        # Runtime comparison
        ax = axes[0, 0]
        ax.bar(method_names, [comp['runtime'].get(m, 0) for m in method_names])
        ax.set_ylabel('Runtime (seconds)')
        ax.set_title('Execution Time')
        ax.tick_params(axis='x', rotation=45)

        # Memory usage
        ax = axes[0, 1]
        ax.bar(method_names, [comp['memory'].get(m, 0) for m in method_names])
        ax.set_ylabel('Peak Memory (MB)')
        ax.set_title('Memory Usage')
        ax.tick_params(axis='x', rotation=45)

        # Throughput
        ax = axes[1, 0]
        ax.bar(method_names, [comp['throughput'].get(m, 0) for m in method_names])
        ax.set_ylabel('Evaluations/sec')
        ax.set_title('Throughput')
        ax.tick_params(axis='x', rotation=45)

        # Efficiency
        ax = axes[1, 1]
        ax.bar(method_names, [comp['efficiency'].get(m, 0) for m in method_names])
        ax.set_ylabel('Parallel Efficiency')
        ax.set_title('Efficiency')
        ax.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plot_path = '/tmp/Librex_method_comparison.png'
        plt.savefig(plot_path, dpi=100)
        print(f"\n📈 Comparison plot saved to: {plot_path}")
        plt.close()

    except Exception as e:
        print(f"Could not create plot: {e}")


def main():
    """Run performance profiling demonstrations"""
    print("\n")
    print("📊 " + "=" * 76 + " 📊")
    print("   Librex PERFORMANCE PROFILING DEMONSTRATION")
    print("📊 " + "=" * 76 + " 📊")

    # 1. Profile single method
    print("\n\n🔍 Test 1: Single Method Profiling")
    profile_result = profile_single_method()

    # 2. Compare multiple methods
    print("\n\n⚖️ Test 2: Method Comparison")
    compare_methods()

    # 3. Test parallel evaluation
    print("\n\n⚡ Test 3: Parallel Evaluation")
    test_parallel_evaluation()

    # 4. Test memory efficiency
    print("\n\n💾 Test 4: Memory Efficiency")
    test_memory_efficiency()

    print("\n" + "=" * 80)
    print("✅ PERFORMANCE PROFILING DEMO COMPLETE")
    print("=" * 80)

    print("\n📌 OPTIMIZATION RECOMMENDATIONS:")
    print("  1. Use GPU acceleration for large populations (>1000)")
    print("  2. Enable mixed precision for 1.5-2x speedup")
    print("  3. Use chunking for populations >10000")
    print("  4. Distribute across GPUs for problems with dim >100")
    print("  5. Profile first to identify bottlenecks")


if __name__ == "__main__":
    main()