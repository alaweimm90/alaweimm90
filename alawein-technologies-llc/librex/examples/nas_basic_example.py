#!/usr/bin/env python3
"""Basic example of using Neural Architecture Search with Librex.

This example demonstrates:
1. Setting up a NAS problem
2. Running different search algorithms
3. Comparing results
4. Using hardware constraints
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from Librex.domains.nas import (
    NASProblem, NASAdapter, NASCell, MacroArchitecture,
    evolutionary_nas, differentiable_nas, bayesian_optimization_nas,
    random_search_nas, HardwareAwareNAS
)
from Librex.domains.nas.zero_cost_proxies import ProxyEnsemble


def example_cell_based_nas():
    """Example of cell-based NAS (like DARTS, ENAS)."""
    print("\n" + "="*60)
    print("CELL-BASED NAS EXAMPLE")
    print("="*60)

    # Define NAS problem
    problem = NASProblem(
        dataset='cifar10',
        search_space='cell',
        objectives=['accuracy'],  # Single objective
        constraints={'max_params': 5e6},  # Parameter constraint
        evaluation_strategy='proxy',  # Use fast proxy evaluation
        max_evaluations=100
    )

    print(f"Problem setup:")
    print(f"  Dataset: {problem.dataset}")
    print(f"  Search space: {problem.search_space}")
    print(f"  Objectives: {[obj.name for obj in problem.objectives]}")
    print(f"  Constraints: {len(problem.constraints)} constraints")

    # 1. Random Search (baseline)
    print("\n1. Random Search Baseline...")
    random_result = random_search_nas(problem, {'n_samples': 20})
    print(f"   Best objective: {random_result['best_objective']:.4f}")
    print(f"   Architecture params: {random_result['best_architecture'].get_params():,}")

    # Reset problem for fair comparison
    problem.reset()

    # 2. Evolutionary Search
    print("\n2. Evolutionary NAS...")
    evo_result = evolutionary_nas(problem, {
        'population_size': 20,
        'n_generations': 10,
        'mutation_rate': 0.1
    })
    print(f"   Best fitness: {evo_result['best_fitness']:.4f}")
    print(f"   Final population size: {len(evo_result['final_population'])}")

    # Reset problem
    problem.reset()

    # 3. Bayesian Optimization
    print("\n3. Bayesian Optimization NAS...")
    bo_result = bayesian_optimization_nas(problem, {
        'n_initial': 10,
        'n_iterations': 30,
        'acquisition': 'ei'
    })
    print(f"   Best objective: {bo_result['best_objective']:.4f}")
    print(f"   Total evaluations: {len(bo_result['all_evaluations'])}")

    # Compare results
    print("\n" + "-"*40)
    print("COMPARISON:")
    print(f"Random Search:    {random_result['best_objective']:.4f}")
    print(f"Evolutionary:     {evo_result['best_fitness']:.4f}")
    print(f"Bayesian Opt:     {bo_result['best_objective']:.4f}")


def example_macro_nas():
    """Example of macro search (layer-wise architecture)."""
    print("\n" + "="*60)
    print("MACRO ARCHITECTURE NAS EXAMPLE")
    print("="*60)

    # Define problem for macro search
    problem = NASProblem(
        dataset='imagenet',
        search_space='macro',
        objectives=['accuracy', 'latency_ms'],  # Multi-objective
        constraints={'max_layers': 50},
        max_evaluations=50
    )

    print(f"Macro search problem:")
    print(f"  Max layers: {problem.macro_config['max_layers']}")
    print(f"  Input channels: {problem.macro_config['input_channels']}")
    print(f"  Num classes: {problem.macro_config['num_classes']}")
    print(f"  Multi-objective: {problem.is_multi_objective()}")

    # Run evolutionary search for multi-objective
    print("\nRunning multi-objective evolutionary search...")
    result = evolutionary_nas(problem, {
        'population_size': 30,
        'n_generations': 10
    })

    best_arch = result['best_architecture']
    print(f"\nBest architecture found:")
    print(f"  Depth: {best_arch.get_depth()} layers")
    print(f"  Parameters: {best_arch.get_params():,}")
    print(f"  Skip connections: {len(best_arch.skip_connections)}")

    # Get Pareto front
    pareto_front = problem.get_pareto_front()
    print(f"\nPareto front:")
    print(f"  {len(pareto_front)} non-dominated architectures")

    for i, solution in enumerate(pareto_front[:3]):
        metrics = solution['metrics']
        print(f"  Solution {i+1}:")
        print(f"    Accuracy: {metrics.get('accuracy', 0):.3f}")
        print(f"    Latency: {metrics.get('latency_ms', 0):.1f} ms")
        print(f"    Params: {metrics.get('params', 0):,}")


def example_hardware_aware_nas():
    """Example of hardware-aware NAS with device constraints."""
    print("\n" + "="*60)
    print("HARDWARE-AWARE NAS EXAMPLE")
    print("="*60)

    # Different target devices
    devices = ['gpu', 'mobile', 'edge']

    for device in devices:
        print(f"\n{device.upper()} Optimization:")

        # Create hardware-aware NAS
        hw_nas = HardwareAwareNAS(target_device=device)

        # Create test architecture
        cell = NASCell(n_nodes=4)

        # Evaluate on device
        metrics = hw_nas.evaluate_architecture(cell, batch_size=1, input_size=224)

        print(f"  Latency: {metrics['latency_ms']:.2f} ms")
        print(f"  Energy: {metrics['energy_mj']:.2f} mJ")
        print(f"  Memory: {metrics['memory_mb']:.2f} MB")
        print(f"  Throughput: {metrics['throughput']:.1f} images/sec")

        # Optimize with constraints
        if device == 'mobile':
            print(f"\n  Optimizing for mobile constraints...")
            result = hw_nas.optimize(
                dataset='cifar10',
                constraints={
                    'max_latency_ms': 20,
                    'max_memory_mb': 50,
                    'max_energy_mj': 10
                },
                search_space='cell',
                max_evaluations=20
            )

            if result['satisfies_constraints']:
                print(f"  ✓ Found architecture satisfying all constraints")
            else:
                print(f"  ✗ Could not satisfy all constraints")


def example_zero_cost_proxies():
    """Example of using zero-cost proxies for fast evaluation."""
    print("\n" + "="*60)
    print("ZERO-COST PROXY EXAMPLE")
    print("="*60)

    # Create different architectures
    architectures = []
    for i in range(10):
        cell = NASCell(n_nodes=4)
        # Random initialization happens in __init__
        architectures.append(cell)

    # Evaluate with proxy ensemble
    proxy_ensemble = ProxyEnsemble()

    print("Architecture evaluation using zero-cost proxies:")
    print("-" * 40)

    scores = []
    for i, arch in enumerate(architectures):
        # Get all proxy scores
        all_scores = proxy_ensemble.compute(arch, return_all=True)

        print(f"\nArchitecture {i+1}:")
        print(f"  Grad Norm:  {all_scores['grad_norm']:.3f}")
        print(f"  Jacob Cov:  {all_scores['jacob_cov']:.3f}")
        print(f"  NTK:        {all_scores['ntk']:.3f}")
        print(f"  SynFlow:    {all_scores['synflow']:.3f}")
        print(f"  Zen Score:  {all_scores['zen_score']:.3f}")
        print(f"  ENSEMBLE:   {all_scores['ensemble']:.3f}")

        scores.append(all_scores['ensemble'])

    # Rank architectures
    ranked_indices = np.argsort(scores)[::-1]
    print("\n" + "-"*40)
    print("Architecture ranking by ensemble score:")
    for rank, idx in enumerate(ranked_indices[:5]):
        print(f"  Rank {rank+1}: Architecture {idx+1} (score: {scores[idx]:.3f})")


def example_differentiable_nas():
    """Example of differentiable NAS (DARTS/GDAS)."""
    print("\n" + "="*60)
    print("DIFFERENTIABLE NAS EXAMPLE")
    print("="*60)

    problem = NASProblem(
        dataset='cifar10',
        search_space='cell'
    )

    # Run DARTS
    print("\nRunning DARTS...")
    darts_result = differentiable_nas(problem, {
        'n_epochs': 10,
        'lr_arch': 3e-4,
        'lr_weights': 0.025
    })

    print(f"Final architecture:")
    arch = darts_result['architecture']
    print(f"  Edges: {len(arch.edges)}")
    print(f"  Parameters: {arch.get_params():,}")

    # Show training history
    history = darts_result['history']
    print(f"\nTraining progress:")
    print(f"  Initial train loss: {history['train_loss'][0]:.4f}")
    print(f"  Final train loss: {history['train_loss'][-1]:.4f}")
    print(f"  Initial val loss: {history['val_loss'][0]:.4f}")
    print(f"  Final val loss: {history['val_loss'][-1]:.4f}")


def example_nas_with_Librex():
    """Example of using NAS with Librex's optimize function."""
    print("\n" + "="*60)
    print("NAS WITH Librex INTEGRATION")
    print("="*60)

    # Create NAS problem
    problem = NASProblem(
        dataset='cifar10',
        search_space='cell',
        objectives=['accuracy'],
        max_evaluations=50
    )

    # Create adapter
    adapter = NASAdapter()

    # Encode as standardized problem
    standardized = adapter.encode_problem(problem)

    print(f"Standardized problem:")
    print(f"  Dimension: {standardized.dimension}")
    print(f"  Encoding type: {standardized.encoding_type}")
    print(f"  Has constraints: {standardized.constraint_function is not None}")

    # Get encoding hints
    hints = adapter.get_encoding_hints(problem)
    print(f"\nEncoding hints:")
    print(f"  Discrete indices: {len(hints['discrete_indices'])} dimensions")
    print(f"  Continuous indices: {len(hints['continuous_indices'])} dimensions")

    # Optimize using standardized interface
    print("\nOptimizing...")
    best_obj = float('-inf')
    best_solution = None

    for i in range(20):
        # Generate solution respecting bounds
        solution = np.zeros(standardized.dimension)
        for j, (low, high) in enumerate(standardized.bounds):
            solution[j] = np.random.uniform(low, high)

        # Evaluate
        obj = standardized.objective_function(solution)

        if obj > best_obj:
            best_obj = obj
            best_solution = solution
            print(f"  Iteration {i+1}: New best = {best_obj:.4f}")

    # Decode best solution
    best_arch = adapter.decode_solution(best_solution, problem)
    print(f"\nBest architecture found:")
    print(f"  Objective: {best_obj:.4f}")
    print(f"  Parameters: {best_arch.get_params():,}")
    print(f"  Architecture hash: {best_arch.get_hash()}")


def main():
    """Run all examples."""
    print("\n" + "#"*60)
    print("# NEURAL ARCHITECTURE SEARCH EXAMPLES")
    print("#"*60)

    # Run examples
    example_cell_based_nas()
    example_macro_nas()
    example_hardware_aware_nas()
    example_zero_cost_proxies()
    example_differentiable_nas()
    example_nas_with_Librex()

    print("\n" + "#"*60)
    print("# ALL EXAMPLES COMPLETED SUCCESSFULLY")
    print("#"*60)


if __name__ == '__main__':
    main()