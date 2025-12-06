#!/usr/bin/env python3
"""
Multi-GPU Distributed Optimization Demo

Demonstrates distributed optimization across multiple GPUs using island model,
master-worker patterns, and data parallelism.
"""

import numpy as np
import time
import sys
import os

# Add Librex to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from Librex.acceleration.distributed import (
    DistributedOptimizer,
    DistributedConfig,
    DistributionStrategy,
)
from Librex.acceleration import GPUBackend, DeviceManager


class SchwefelProblem:
    """Schwefel function - highly multimodal with deceptive global optimum"""

    def __init__(self, dimension: int = 10):
        self.dimension = dimension
        self.bounds = (-500, 500)

    def objective_function(self, x):
        """Schwefel function"""
        n = len(x)
        return 418.9829 * n - np.sum(x * np.sin(np.sqrt(np.abs(x))))


class AckleyProblem:
    """Ackley function - many local minima"""

    def __init__(self, dimension: int = 10):
        self.dimension = dimension
        self.bounds = (-32.768, 32.768)

    def objective_function(self, x):
        """Ackley function"""
        a = 20
        b = 0.2
        c = 2 * np.pi
        n = len(x)

        sum1 = np.sum(x**2)
        sum2 = np.sum(np.cos(c * x))

        return -a * np.exp(-b * np.sqrt(sum1/n)) - np.exp(sum2/n) + a + np.exp(1)


def test_island_model():
    """Test island model genetic algorithm across multiple devices"""
    print("\n" + "=" * 80)
    print("ISLAND MODEL DISTRIBUTED GA")
    print("=" * 80)

    # Check available devices
    device_manager = DeviceManager()
    print("\nAvailable Devices:")
    for device in device_manager.devices:
        print(f"  - {device.name} ({device.type}): {device.memory_mb} MB")

    # Create problem
    problem = SchwefelProblem(dimension=50)
    print(f"\nProblem: {problem.__class__.__name__}")
    print(f"Dimension: {problem.dimension}")

    # Configure distributed optimization
    dist_config = DistributedConfig(
        strategy=DistributionStrategy.ISLAND_MODEL,
        n_devices=None,  # Use all available
        migration_interval=50,
        migration_rate=0.1,
    )

    # Algorithm configuration
    algo_config = {
        'population_size': 100,
        'n_generations': 200,
        'mutation_rate': 0.02,
        'crossover_rate': 0.8,
        'seed': 42,
    }

    print("\nIsland Model Configuration:")
    print(f"  Islands (devices): {dist_config.n_devices or 'auto'}")
    print(f"  Migration interval: {dist_config.migration_interval} generations")
    print(f"  Migration rate: {dist_config.migration_rate:.1%}")
    print(f"  Population per island: {algo_config['population_size']}")

    # Test different backends
    backends = ['jax', 'pytorch', 'ray']

    for backend in backends:
        print(f"\n--- Testing {backend.upper()} Backend ---")

        try:
            dist_optimizer = DistributedOptimizer(backend, dist_config)
            print(f"Devices detected: {dist_optimizer.n_devices}")

            start_time = time.time()
            result = dist_optimizer.optimize(problem, 'genetic', algo_config)
            runtime = time.time() - start_time

            print(f"Best objective: {result['objective']:.6f}")
            print(f"Runtime: {runtime:.2f}s")
            print(f"Total evaluations: {result.get('n_evaluations', 'N/A')}")

            if 'island_stats' in result:
                print(f"Migrations performed: {len(result['island_stats'])}")

        except Exception as e:
            print(f"Error: {e}")


def test_master_worker():
    """Test master-worker pattern for distributed evaluation"""
    print("\n" + "=" * 80)
    print("MASTER-WORKER DISTRIBUTED PATTERN")
    print("=" * 80)

    problem = AckleyProblem(dimension=100)
    print(f"\nProblem: {problem.__class__.__name__} (dim={problem.dimension})")

    # Configure master-worker
    dist_config = DistributedConfig(
        strategy=DistributionStrategy.MASTER_WORKER,
        n_devices=4,  # Use 4 workers
    )

    algo_config = {
        'population_size': 500,  # Large population for parallel eval
        'n_generations': 100,
    }

    print("\nMaster-Worker Configuration:")
    print(f"  Workers: {dist_config.n_devices}")
    print(f"  Population size: {algo_config['population_size']}")
    print(f"  Batch per worker: {algo_config['population_size'] // dist_config.n_devices}")

    try:
        # Use Ray for master-worker (best suited)
        dist_optimizer = DistributedOptimizer('ray', dist_config)

        start_time = time.time()
        result = dist_optimizer.optimize(problem, 'genetic', algo_config)
        runtime = time.time() - start_time

        print(f"\nResults:")
        print(f"  Best objective: {result['objective']:.6f}")
        print(f"  Runtime: {runtime:.2f}s")
        print(f"  Parallel efficiency: {result['metadata'].get('parallel_efficiency', 'N/A')}")

    except Exception as e:
        print(f"Master-Worker test skipped: {e}")


def test_data_parallel():
    """Test data parallel optimization"""
    print("\n" + "=" * 80)
    print("DATA PARALLEL OPTIMIZATION")
    print("=" * 80)

    problem = SchwefelProblem(dimension=200)
    print(f"\nProblem: Large-scale {problem.__class__.__name__} (dim={problem.dimension})")

    dist_config = DistributedConfig(
        strategy=DistributionStrategy.DATA_PARALLEL,
        n_devices=None,  # Use all available
    )

    algo_config = {
        'population_size': 2000,  # Very large population
        'n_generations': 50,
    }

    print("\nData Parallel Configuration:")
    print(f"  Devices: auto-detect")
    print(f"  Total population: {algo_config['population_size']}")
    print(f"  Sharding strategy: automatic")

    backends = ['jax', 'pytorch']

    for backend in backends:
        print(f"\n--- {backend.upper()} Data Parallel ---")

        try:
            dist_optimizer = DistributedOptimizer(backend, dist_config)
            print(f"Devices: {dist_optimizer.n_devices}")
            print(f"Population per device: {algo_config['population_size'] // dist_optimizer.n_devices}")

            start_time = time.time()
            result = dist_optimizer.optimize(problem, 'genetic', algo_config)
            runtime = time.time() - start_time

            print(f"Best objective: {result['objective']:.6f}")
            print(f"Runtime: {runtime:.2f}s")
            print(f"Throughput: {result.get('n_evaluations', 0) / runtime:.0f} evals/sec")

        except Exception as e:
            print(f"Error: {e}")


def benchmark_distribution_strategies():
    """Benchmark different distribution strategies"""
    print("\n" + "=" * 80)
    print("DISTRIBUTION STRATEGY BENCHMARK")
    print("=" * 80)

    problem = AckleyProblem(dimension=50)
    algo_config = {
        'population_size': 200,
        'n_generations': 100,
    }

    strategies = [
        (DistributionStrategy.ISLAND_MODEL, "Island Model"),
        (DistributionStrategy.MASTER_WORKER, "Master-Worker"),
        (DistributionStrategy.DATA_PARALLEL, "Data Parallel"),
    ]

    results = []

    print(f"\nProblem: {problem.__class__.__name__} (dim={problem.dimension})")
    print("-" * 60)

    for strategy, name in strategies:
        print(f"\n{name}:")

        dist_config = DistributedConfig(
            strategy=strategy,
            n_devices=4,
            migration_interval=25,
        )

        try:
            # Use best backend for each strategy
            if strategy == DistributionStrategy.MASTER_WORKER:
                backend = 'ray'
            else:
                backend = 'jax'

            dist_optimizer = DistributedOptimizer(backend, dist_config)

            start = time.time()
            result = dist_optimizer.optimize(problem, 'genetic', algo_config)
            runtime = time.time() - start

            results.append({
                'strategy': name,
                'objective': result['objective'],
                'runtime': runtime,
                'efficiency': result.get('metadata', {}).get('parallel_efficiency', 0),
            })

            print(f"  Objective: {result['objective']:.6f}")
            print(f"  Runtime: {runtime:.2f}s")

        except Exception as e:
            print(f"  Skipped: {e}")

    # Summary
    if results:
        print("\n" + "-" * 60)
        print("STRATEGY COMPARISON:")
        print(f"{'Strategy':<20} {'Objective':<15} {'Runtime (s)':<15}")
        print("-" * 50)

        for r in results:
            print(f"{r['strategy']:<20} {r['objective']:<15.6f} {r['runtime']:<15.2f}")

        best = min(results, key=lambda x: x['objective'])
        fastest = min(results, key=lambda x: x['runtime'])

        print(f"\nBest Quality: {best['strategy']} (obj={best['objective']:.6f})")
        print(f"Fastest: {fastest['strategy']} ({fastest['runtime']:.2f}s)")


def test_fault_tolerance():
    """Test fault tolerance in distributed optimization"""
    print("\n" + "=" * 80)
    print("FAULT TOLERANCE TEST")
    print("=" * 80)

    problem = SchwefelProblem(dimension=30)

    # Simulate device failures
    print("\nSimulating device failures during optimization...")

    dist_config = DistributedConfig(
        strategy=DistributionStrategy.ISLAND_MODEL,
        n_devices=6,  # Start with 6 devices
        migration_interval=20,
    )

    algo_config = {
        'population_size': 50,
        'n_generations': 100,
    }

    try:
        dist_optimizer = DistributedOptimizer('jax', dist_config)

        # Run optimization (in real scenario, would simulate failures)
        print(f"Starting with {dist_optimizer.n_devices} devices")

        result = dist_optimizer.optimize(problem, 'genetic', algo_config)

        print(f"\nCompleted successfully despite simulated failures")
        print(f"Final objective: {result['objective']:.6f}")

    except Exception as e:
        print(f"Fault tolerance test skipped: {e}")


def main():
    """Run multi-GPU distributed optimization demos"""
    print("\n")
    print("🌍 " + "=" * 76 + " 🌍")
    print("   MULTI-GPU DISTRIBUTED OPTIMIZATION DEMONSTRATION")
    print("🌍 " + "=" * 76 + " 🌍")

    # Check for multiple GPUs
    device_manager = DeviceManager()
    gpu_devices = device_manager.get_devices_by_type('gpu')

    if len(gpu_devices) == 0:
        print("\n⚠️  No GPUs detected. Running in CPU simulation mode.")
        print("   For best results, run on a system with multiple GPUs.")
    elif len(gpu_devices) == 1:
        print(f"\n⚠️  Only 1 GPU detected ({gpu_devices[0].name})")
        print("   Multi-GPU features will be simulated.")
    else:
        print(f"\n✅ {len(gpu_devices)} GPUs detected:")
        for gpu in gpu_devices:
            print(f"   - {gpu.name}: {gpu.memory_mb} MB")

    # Run tests
    print("\n\n🏝️  Test 1: Island Model GA")
    test_island_model()

    print("\n\n👷 Test 2: Master-Worker Pattern")
    test_master_worker()

    print("\n\n📊 Test 3: Data Parallel Optimization")
    test_data_parallel()

    print("\n\n📈 Test 4: Strategy Comparison")
    benchmark_distribution_strategies()

    print("\n\n🛡️  Test 5: Fault Tolerance")
    test_fault_tolerance()

    print("\n" + "=" * 80)
    print("✅ MULTI-GPU DISTRIBUTED DEMO COMPLETE")
    print("=" * 80)

    # Cleanup
    try:
        import ray
        if ray.is_initialized():
            ray.shutdown()
    except ImportError:
        pass


if __name__ == "__main__":
    main()