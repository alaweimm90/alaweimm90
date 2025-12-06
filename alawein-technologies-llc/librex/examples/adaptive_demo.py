#!/usr/bin/env python3
"""
Demonstration of Adaptive Learning System for Librex

This script showcases the various adaptive learning capabilities including:
- Online learning with multi-armed bandits
- Portfolio-based optimization
- Ensemble methods
- Surrogate-based optimization
- Meta-learning from history

Author: Meshal Alawein
Date: 2025-11-18
"""

import numpy as np
import time
from typing import Dict, Any

# Import Librex components
from Librex import optimize
from Librex.adapters.qap import QAPAdapter
from Librex.adaptive import (
    OnlineLearner,
    AlgorithmPortfolioManager,
    EnsembleOptimizer,
    PerformanceDatabase,
    MetaLearner,
    PerformancePredictor,
)
from Librex.adaptive.surrogate import SurrogateOptimizer, AcquisitionFunction
from Librex.adaptive.performance_db import OptimizationRun
from datetime import datetime


def demo_online_learning():
    """Demonstrate online learning with multi-armed bandits."""
    print("\n" + "="*60)
    print("DEMO 1: Online Learning with Multi-Armed Bandits")
    print("="*60)

    # Create a QAP problem
    n = 12  # Problem size
    np.random.seed(42)
    flow_matrix = np.random.rand(n, n)
    flow_matrix = (flow_matrix + flow_matrix.T) / 2  # Symmetric
    np.fill_diagonal(flow_matrix, 0)

    distance_matrix = np.random.rand(n, n)
    distance_matrix = (distance_matrix + distance_matrix.T) / 2
    np.fill_diagonal(distance_matrix, 0)

    problem = {
        'flow_matrix': flow_matrix,
        'distance_matrix': distance_matrix
    }
    adapter = QAPAdapter()

    # Test different bandit strategies
    strategies = ['ucb1', 'thompson', 'exp3']
    results = {}

    for strategy in strategies:
        print(f"\nTesting {strategy.upper()} strategy...")
        start_time = time.time()

        result = optimize(
            problem,
            adapter,
            method='adaptive',
            config={
                'learning_config': {
                    'mode': 'online',
                    'strategy': strategy,
                    'time_budget': 30  # 30 seconds
                },
                'max_iterations': 1000
            }
        )

        runtime = time.time() - start_time
        results[strategy] = {
            'objective': result['objective'],
            'runtime': runtime
        }

        print(f"  Objective: {result['objective']:.4f}")
        print(f"  Runtime: {runtime:.2f}s")

    # Compare strategies
    best_strategy = min(results.keys(), key=lambda k: results[k]['objective'])
    print(f"\n✓ Best strategy: {best_strategy.upper()} with objective {results[best_strategy]['objective']:.4f}")


def demo_portfolio_optimization():
    """Demonstrate portfolio-based optimization."""
    print("\n" + "="*60)
    print("DEMO 2: Algorithm Portfolio Management")
    print("="*60)

    # Create a TSP-like problem
    n_cities = 20
    np.random.seed(123)

    # Generate random city coordinates
    cities = np.random.rand(n_cities, 2) * 100

    # Calculate distance matrix
    distance_matrix = np.zeros((n_cities, n_cities))
    for i in range(n_cities):
        for j in range(n_cities):
            if i != j:
                distance_matrix[i, j] = np.linalg.norm(cities[i] - cities[j])

    problem = {'distance_matrix': distance_matrix}

    # Note: We'll use QAPAdapter as a proxy since TSP adapter might not be available
    # In practice, you would use the appropriate TSPAdapter
    adapter = QAPAdapter()

    print("\nRunning portfolio optimization with 5 algorithms in parallel...")
    start_time = time.time()

    result = optimize(
        problem,
        adapter,
        method='adaptive',
        config={
            'learning_config': {
                'mode': 'portfolio',
                'time_budget': 60,  # 60 seconds total
                'n_parallel': 3     # Run 3 algorithms in parallel
            }
        }
    )

    runtime = time.time() - start_time

    print(f"\n✓ Portfolio optimization completed")
    print(f"  Best objective: {result['objective']:.4f}")
    print(f"  Best method: {result['metadata'].get('method', 'unknown')}")
    print(f"  Total runtime: {runtime:.2f}s")

    # Show all results
    if 'all_results' in result['metadata']:
        print("\n  Individual algorithm results:")
        for alg, obj in result['metadata']['all_results'].items():
            print(f"    - {alg}: {obj:.4f}")


def demo_ensemble_optimization():
    """Demonstrate ensemble optimization with voting."""
    print("\n" + "="*60)
    print("DEMO 3: Ensemble Optimization")
    print("="*60)

    # Create a small QAP problem for quick demonstration
    n = 8
    np.random.seed(456)

    flow_matrix = np.random.randint(1, 10, (n, n))
    flow_matrix = (flow_matrix + flow_matrix.T) // 2
    np.fill_diagonal(flow_matrix, 0)

    distance_matrix = np.random.randint(1, 10, (n, n))
    distance_matrix = (distance_matrix + distance_matrix.T) // 2
    np.fill_diagonal(distance_matrix, 0)

    problem = {
        'flow_matrix': flow_matrix,
        'distance_matrix': distance_matrix
    }
    adapter = QAPAdapter()

    # Test different voting methods
    voting_methods = ['weighted', 'majority', 'best']

    for voting_method in voting_methods:
        print(f"\nTesting {voting_method} voting...")

        result = optimize(
            problem,
            adapter,
            method='adaptive',
            config={
                'learning_config': {
                    'mode': 'ensemble',
                    'voting_method': voting_method,
                    'n_rounds': 2  # Multiple rounds with restarts
                },
                'max_iterations': 500
            }
        )

        print(f"  Objective: {result['objective']:.4f}")

        if 'ensemble_summary' in result['metadata']:
            summary = result['metadata']['ensemble_summary']
            print(f"  Ensemble size: {summary['n_members']}")
            print("  Top members:")
            for member in summary['members'][:3]:
                print(f"    - {member['name']}: weight={member['weight']:.3f}")


def demo_surrogate_optimization():
    """Demonstrate surrogate-based optimization for expensive functions."""
    print("\n" + "="*60)
    print("DEMO 4: Surrogate-Based Optimization")
    print("="*60)

    # Define an expensive objective function (Rosenbrock)
    eval_count = 0

    def expensive_rosenbrock(x):
        nonlocal eval_count
        eval_count += 1
        # Simulate expensive evaluation
        time.sleep(0.1)  # 100ms per evaluation

        # Rosenbrock function
        return sum(100.0 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2
                  for i in range(len(x) - 1))

    # Set up surrogate optimizer
    dimension = 2
    bounds = np.array([[-2.0, 2.0]] * dimension)

    optimizer = SurrogateOptimizer(
        expensive_rosenbrock,
        bounds,
        acquisition=AcquisitionFunction.EI,
        n_initial_points=5
    )

    print(f"Optimizing {dimension}D Rosenbrock function with GP surrogate...")
    print("(Each evaluation takes 100ms to simulate expense)")

    # Run optimization with limited budget
    result = optimizer.optimize(n_iterations=20)

    print(f"\n✓ Optimization completed")
    print(f"  Best point: {result['best_point']}")
    print(f"  Best value: {result['best_value']:.6f}")
    print(f"  Total evaluations: {result['n_evaluations']}")
    print(f"  Converged: {result['convergence']['converged']}")

    # True optimum is at [1, 1] with value 0
    distance_to_optimum = np.linalg.norm(result['best_point'] - np.ones(dimension))
    print(f"  Distance to true optimum: {distance_to_optimum:.4f}")


def demo_meta_learning():
    """Demonstrate meta-learning from historical data."""
    print("\n" + "="*60)
    print("DEMO 5: Meta-Learning from History")
    print("="*60)

    # Create performance database
    db = PerformanceDatabase()

    # Simulate historical optimization runs
    print("Simulating 50 historical optimization runs...")

    algorithms = ['simulated_annealing', 'genetic_algorithm', 'tabu_search']

    for i in range(50):
        # Random problem characteristics
        problem_size = np.random.randint(10, 50)
        sparsity = np.random.random()

        # Features representing the problem
        features = np.array([
            problem_size,
            sparsity,
            np.random.random(),  # Correlation
            np.random.random(),  # Variance
            np.random.random()   # Complexity
        ])

        # Simulate that some algorithms work better on certain problems
        algorithm = np.random.choice(algorithms)

        # Performance depends on problem characteristics
        if algorithm == 'simulated_annealing':
            # Good for high variance problems
            performance = 0.8 + 0.2 * features[3]
        elif algorithm == 'genetic_algorithm':
            # Good for large problems
            performance = 0.7 + 0.3 * (features[0] / 50)
        else:  # tabu_search
            # Good for sparse problems
            performance = 0.75 + 0.25 * features[1]

        # Add noise
        performance = np.clip(performance + np.random.normal(0, 0.1), 0, 1)
        objective = 100 / (1 + performance)  # Convert to minimization

        # Record run
        run = OptimizationRun(
            problem_id=f'problem_{i}',
            problem_features=features,
            method_used=algorithm,
            hyperparameters={'param1': np.random.random()},
            objective_value=objective,
            runtime_seconds=np.random.uniform(1, 60),
            n_evaluations=np.random.randint(100, 10000),
            convergence_rate=performance,
            solution_quality=performance,
            timestamp=datetime.now(),
            metadata={}
        )
        db.record_run(run)

    print("✓ Historical data generated")

    # Train performance predictor
    print("\nTraining performance predictor on historical data...")

    predictor = PerformancePredictor(model_type='gradient_boosting')

    # Get training data from database
    history = db.get_performance_history(limit=50)
    training_data = [
        (run.problem_features, run.method_used, run.solution_quality)
        for run in history
    ]

    predictor.train(training_data, algorithms)

    # Test on new problem
    print("\nPredicting best algorithm for new problem...")

    new_problem_features = np.array([
        30,    # Medium size
        0.8,   # High sparsity
        0.5,   # Medium correlation
        0.9,   # High variance
        0.6    # Medium complexity
    ])

    predictions = predictor.predict(new_problem_features)

    print("\nPredicted performance (higher is better):")
    for alg, (mean, std) in predictions.items():
        print(f"  {alg}: {mean:.3f} ± {std:.3f}")

    # Recommend best algorithm
    best_alg, confidence = predictor.recommend_algorithm(
        new_problem_features,
        risk_aversion=0.3  # Balance exploration/exploitation
    )

    print(f"\n✓ Recommended algorithm: {best_alg} (confidence: {confidence:.2%})")

    # Query similar problems from history
    print("\nFinding similar problems in history...")
    similar_problems = db.query_similar_problems(new_problem_features, n_similar=3)

    if similar_problems:
        print(f"Found {len(similar_problems)} similar problems:")
        for prob in similar_problems:
            print(f"  - {prob.problem_id}: {prob.method_used} achieved {prob.solution_quality:.3f}")


def demo_visualization():
    """Create visualization of adaptive learning performance."""
    print("\n" + "="*60)
    print("DEMO 6: Performance Visualization")
    print("="*60)

    try:
        import matplotlib.pyplot as plt

        # Simulate regret curves for different strategies
        n_rounds = 100
        strategies = ['UCB1', 'Thompson', 'EXP3', 'Random']

        plt.figure(figsize=(12, 5))

        # Subplot 1: Cumulative Regret
        plt.subplot(1, 2, 1)
        for strategy in strategies:
            # Simulate regret (better strategies have lower regret)
            if strategy == 'UCB1':
                regret = np.log(np.arange(1, n_rounds+1)) * 5
            elif strategy == 'Thompson':
                regret = np.log(np.arange(1, n_rounds+1)) * 4
            elif strategy == 'EXP3':
                regret = np.sqrt(np.arange(1, n_rounds+1)) * 3
            else:  # Random
                regret = np.arange(1, n_rounds+1) * 0.5

            # Add noise
            regret += np.random.randn(n_rounds) * 0.5
            regret = np.maximum.accumulate(regret)

            plt.plot(regret, label=strategy)

        plt.xlabel('Round')
        plt.ylabel('Cumulative Regret')
        plt.title('Multi-Armed Bandit Performance')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Subplot 2: Algorithm Selection Frequency
        plt.subplot(1, 2, 2)
        algorithms = ['SA', 'GA', 'TS', 'ACO', 'PSO']
        frequencies = np.random.dirichlet(np.ones(len(algorithms)) * 2) * 100

        colors = plt.cm.Set3(range(len(algorithms)))
        plt.bar(algorithms, frequencies, color=colors)
        plt.xlabel('Algorithm')
        plt.ylabel('Selection Frequency (%)')
        plt.title('Portfolio Algorithm Usage')
        plt.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig('adaptive_learning_performance.png', dpi=100, bbox_inches='tight')
        print("✓ Performance plot saved as 'adaptive_learning_performance.png'")

    except ImportError:
        print("⚠ Matplotlib not available, skipping visualization")


def main():
    """Run all demonstrations."""
    print("\n" + "="*70)
    print(" Librex ADAPTIVE LEARNING SYSTEM DEMONSTRATION")
    print("="*70)
    print("\nThis demo showcases the adaptive learning capabilities including:")
    print("• Online learning with multi-armed bandits (UCB1, Thompson, EXP3)")
    print("• Portfolio-based parallel algorithm execution")
    print("• Ensemble methods with voting strategies")
    print("• Surrogate-based optimization for expensive functions")
    print("• Meta-learning from historical optimization data")
    print("• Performance visualization and analysis")

    # Run demonstrations
    demo_online_learning()
    demo_portfolio_optimization()
    demo_ensemble_optimization()
    demo_surrogate_optimization()
    demo_meta_learning()
    demo_visualization()

    print("\n" + "="*70)
    print(" DEMONSTRATION COMPLETE")
    print("="*70)
    print("\nThe Adaptive Learning System enables Librex to:")
    print("✓ Learn from past optimization runs")
    print("✓ Automatically select the best algorithm for each problem")
    print("✓ Allocate computational resources efficiently")
    print("✓ Combine multiple algorithms for robust solutions")
    print("✓ Handle expensive objective functions with surrogates")
    print("✓ Transfer knowledge across similar problems")


if __name__ == "__main__":
    main()