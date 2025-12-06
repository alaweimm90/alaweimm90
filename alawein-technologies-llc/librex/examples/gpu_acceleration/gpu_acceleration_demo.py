#!/usr/bin/env python3
"""
GPU Acceleration Demo for Librex

Demonstrates GPU-accelerated optimization with multiple backends.
"""

import numpy as np
import time
import sys
import os

# Add Librex to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from Librex.acceleration import (
    GPUBackend,
    enable_mixed_precision,
    disable_mixed_precision,
)
from Librex.acceleration.gpu_methods import (
    GPUGeneticAlgorithm,
    GPUParticleSwarm,
    GPUSimulatedAnnealing,
)
from Librex.acceleration.gpu_methods.gpu_genetic import GeneticConfig
from Librex.acceleration.gpu_methods.gpu_pso import PSOConfig
from Librex.acceleration.gpu_methods.gpu_simulated_annealing import SimulatedAnnealingConfig


# Define test problems
class RastriginProblem:
    """Rastrigin function - a classic optimization benchmark"""

    def __init__(self, dimension: int = 10):
        self.dimension = dimension
        self.bounds = (-5.12, 5.12)

    def objective_function(self, x):
        """Rastrigin function: f(x) = 10n + sum(x_i^2 - 10*cos(2*pi*x_i))"""
        A = 10
        n = len(x)
        return A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x))


class RosenbrockProblem:
    """Rosenbrock function - valley-shaped optimization problem"""

    def __init__(self, dimension: int = 10):
        self.dimension = dimension
        self.bounds = (-2.048, 2.048)

    def objective_function(self, x):
        """Rosenbrock function: f(x) = sum(100*(x_{i+1} - x_i^2)^2 + (1 - x_i)^2)"""
        return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)


def compare_backends():
    """Compare performance across different GPU backends"""
    print("=" * 80)
    print("Librex GPU ACCELERATION DEMO")
    print("=" * 80)

    # Create test problem
    problem = RastriginProblem(dimension=100)
    print(f"\nTest Problem: {problem.__class__.__name__}")
    print(f"Dimension: {problem.dimension}")
    print(f"Bounds: {problem.bounds}")

    # Test configurations
    backends = ['numpy', 'jax', 'pytorch', 'cupy']
    results = {}

    print("\n" + "=" * 80)
    print("BACKEND COMPARISON")
    print("=" * 80)

    for backend in backends:
        print(f"\n--- Testing {backend.upper()} Backend ---")

        try:
            # Initialize GPU backend
            gpu = GPUBackend(backend)
            print(f"Backend initialized: {gpu}")
            print(f"Is GPU: {gpu.is_gpu}")

            # Get memory info
            mem_info = gpu.get_memory_info()
            if mem_info['total_mb'] > 0:
                print(f"GPU Memory: {mem_info['used_mb']}/{mem_info['total_mb']} MB")

            # Run optimization
            ga = GPUGeneticAlgorithm(backend)
            config = GeneticConfig(
                population_size=100,
                n_generations=50,
                mutation_rate=0.01,
                crossover_rate=0.8,
                seed=42
            )

            start_time = time.time()
            result = ga.optimize(problem, config)
            runtime = time.time() - start_time

            results[backend] = {
                'objective': result['objective'],
                'runtime': runtime,
                'n_evaluations': result['n_evaluations'],
                'throughput': result['n_evaluations'] / runtime,
            }

            print(f"Best Objective: {result['objective']:.6f}")
            print(f"Runtime: {runtime:.2f} seconds")
            print(f"Throughput: {results[backend]['throughput']:.0f} evals/sec")

        except Exception as e:
            print(f"Backend {backend} not available: {e}")
            results[backend] = None

    # Summary
    print("\n" + "=" * 80)
    print("PERFORMANCE SUMMARY")
    print("=" * 80)

    print(f"\n{'Backend':<15} {'Objective':<15} {'Runtime (s)':<15} {'Throughput':<15}")
    print("-" * 60)

    for backend, result in results.items():
        if result:
            print(f"{backend:<15} {result['objective']:<15.6f} {result['runtime']:<15.2f} {result['throughput']:<15.0f}")
        else:
            print(f"{backend:<15} {'N/A':<15} {'N/A':<15} {'N/A':<15}")

    # Calculate speedup
    if results.get('numpy') and any(r for r in results.values() if r and r != results['numpy']):
        print("\n" + "-" * 60)
        print("SPEEDUP vs CPU (NumPy):")
        cpu_runtime = results['numpy']['runtime']

        for backend, result in results.items():
            if result and backend != 'numpy':
                speedup = cpu_runtime / result['runtime']
                print(f"  {backend}: {speedup:.2f}x")


def test_mixed_precision():
    """Test mixed precision performance"""
    print("\n" + "=" * 80)
    print("MIXED PRECISION TESTING")
    print("=" * 80)

    problem = RosenbrockProblem(dimension=50)

    # Test without mixed precision
    print("\n--- Float32 Precision ---")
    disable_mixed_precision()

    ga = GPUGeneticAlgorithm('jax')
    config = GeneticConfig(population_size=200, n_generations=100)

    start = time.time()
    result_fp32 = ga.optimize(problem, config)
    fp32_time = time.time() - start

    print(f"Objective: {result_fp32['objective']:.6f}")
    print(f"Runtime: {fp32_time:.2f}s")

    # Test with mixed precision
    print("\n--- Mixed Precision (Float16) ---")
    enable_mixed_precision('jax')

    start = time.time()
    result_amp = ga.optimize(problem, config)
    amp_time = time.time() - start

    disable_mixed_precision()

    print(f"Objective: {result_amp['objective']:.6f}")
    print(f"Runtime: {amp_time:.2f}s")

    # Compare
    speedup = fp32_time / amp_time if amp_time > 0 else 1.0
    print(f"\nMixed Precision Speedup: {speedup:.2f}x")
    print(f"Accuracy difference: {abs(result_fp32['objective'] - result_amp['objective']):.6f}")


def test_algorithm_comparison():
    """Compare different GPU-accelerated algorithms"""
    print("\n" + "=" * 80)
    print("GPU ALGORITHM COMPARISON")
    print("=" * 80)

    problem = RastriginProblem(dimension=30)

    algorithms = [
        ('Genetic Algorithm', GPUGeneticAlgorithm, GeneticConfig(
            population_size=100, n_generations=100
        )),
        ('Particle Swarm', GPUParticleSwarm, PSOConfig(
            n_particles=100, n_iterations=100
        )),
        ('Simulated Annealing', GPUSimulatedAnnealing, SimulatedAnnealingConfig(
            n_chains=32, n_iterations=3200
        )),
    ]

    print(f"\nProblem: {problem.__class__.__name__} (dim={problem.dimension})")
    print("-" * 60)

    results = []

    for name, algo_class, config in algorithms:
        print(f"\n{name}:")

        try:
            algo = algo_class('jax')
            start = time.time()
            result = algo.optimize(problem, config)
            runtime = time.time() - start

            print(f"  Best objective: {result['objective']:.6f}")
            print(f"  Runtime: {runtime:.2f}s")
            print(f"  Evaluations: {result.get('n_evaluations', 'N/A')}")

            results.append({
                'algorithm': name,
                'objective': result['objective'],
                'runtime': runtime,
            })

        except Exception as e:
            print(f"  Error: {e}")

    # Find best
    if results:
        best = min(results, key=lambda x: x['objective'])
        print(f"\nBest Algorithm: {best['algorithm']} (objective={best['objective']:.6f})")


def test_scalability():
    """Test scalability with different problem sizes"""
    print("\n" + "=" * 80)
    print("SCALABILITY TEST")
    print("=" * 80)

    dimensions = [10, 50, 100, 500]
    backend = 'jax'  # Use JAX for this test

    print(f"\nBackend: {backend.upper()}")
    print(f"Algorithm: Genetic Algorithm")
    print("-" * 60)
    print(f"{'Dimension':<15} {'Runtime (s)':<15} {'Throughput':<15}")
    print("-" * 60)

    for dim in dimensions:
        problem = RastriginProblem(dimension=dim)
        ga = GPUGeneticAlgorithm(backend)

        # Scale population with dimension
        pop_size = min(50 * dim, 1000)
        config = GeneticConfig(
            population_size=pop_size,
            n_generations=20,
        )

        try:
            start = time.time()
            result = ga.optimize(problem, config)
            runtime = time.time() - start

            throughput = result['n_evaluations'] / runtime
            print(f"{dim:<15} {runtime:<15.2f} {throughput:<15.0f}")

        except Exception as e:
            print(f"{dim:<15} {'Error':<15} {str(e)[:30]}")


def main():
    """Run all GPU acceleration demos"""
    print("\n")
    print("🚀 " + "=" * 76 + " 🚀")
    print("   Librex GPU ACCELERATION DEMONSTRATION")
    print("🚀 " + "=" * 76 + " 🚀")

    # 1. Compare backends
    print("\n\n📊 Test 1: Backend Comparison")
    compare_backends()

    # 2. Test mixed precision
    print("\n\n⚡ Test 2: Mixed Precision")
    try:
        test_mixed_precision()
    except Exception as e:
        print(f"Mixed precision test skipped: {e}")

    # 3. Compare algorithms
    print("\n\n🏆 Test 3: Algorithm Comparison")
    test_algorithm_comparison()

    # 4. Test scalability
    print("\n\n📈 Test 4: Scalability")
    test_scalability()

    print("\n" + "=" * 80)
    print("✅ GPU ACCELERATION DEMO COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()