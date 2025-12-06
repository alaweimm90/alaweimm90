"""
Test Optimal Configuration on Real ASlib Data

Compare Librex.Meta's default configuration vs. optimal configuration
found in ablation studies (Week 4) on real SAT11-HAND data.
"""

from benchmark.run_evaluation import ComprehensiveEvaluator
from libria_meta import Librex.Meta
from benchmark.aslib_parser import ASLibScenarioLoader
from benchmark.evaluate_Librex.Meta import create_mock_solver_for_aslib
import time
import numpy as np
import pandas as pd

def test_config(scenario_name, n_clusters, elo_k, ucb_c, n_tournament_rounds, config_name):
    """Test a specific configuration"""
    print(f"\n{'='*70}")
    print(f"Testing {config_name}")
    print(f"  n_clusters={n_clusters}, elo_k={elo_k}, ucb_c={ucb_c}, n_tournament_rounds={n_tournament_rounds}")
    print(f"{'='*70}")

    # Load scenario
    scenario_path = f"aslib_data/{scenario_name}"
    scenario = ASLibScenarioLoader(scenario_path)

    # Create mock solvers
    algorithm_names = scenario.get_algorithm_names()
    solvers = [
        create_mock_solver_for_aslib(name, base_performance=0.5)
        for name in algorithm_names
    ]

    # Create Librex.Meta with specific config
    Librex.Meta = Librex.Meta(
        solvers=solvers,
        n_clusters=n_clusters,
        elo_k=elo_k,
        ucb_c=ucb_c,
        n_tournament_rounds=n_tournament_rounds
    )

    # Get train/test split
    all_instance_ids = scenario.get_instance_ids()
    n_train = int(0.8 * len(all_instance_ids))
    train_ids = all_instance_ids[:n_train]
    test_ids = all_instance_ids[n_train:]

    # Load training data
    training_data = scenario.get_training_data(instance_ids=train_ids)

    # Train
    start_time = time.time()
    Librex.Meta.fit(training_data)
    train_time = time.time() - start_time

    # Test
    test_data = scenario.get_training_data(instance_ids=test_ids)

    selection_times = []
    regrets = []
    top1_correct = 0
    top3_correct = 0

    for test_instance in test_data:
        features = test_instance['features']
        true_performances = test_instance['performances']

        # Select solver
        class ASLibInstance:
            def __init__(self, features):
                self.features = features

        instance = ASLibInstance(features)

        start_time = time.time()
        selected = Librex.Meta.select_solver(instance, features=features)
        selection_time = time.time() - start_time
        selection_times.append(selection_time)

        # Get best
        best_algo = max(true_performances, key=true_performances.get)
        best_perf = true_performances[best_algo]

        # Get selected performance
        selected_name = selected.name
        selected_perf = true_performances.get(selected_name, 0.0)

        # Regret
        regret = best_perf - selected_perf
        regrets.append(regret)

        # Top-k accuracy
        if selected_name == best_algo:
            top1_correct += 1
            top3_correct += 1
        else:
            top3_algos = sorted(true_performances, key=true_performances.get, reverse=True)[:3]
            if selected_name in top3_algos:
                top3_correct += 1

    results = {
        'config': config_name,
        'n_clusters': n_clusters,
        'elo_k': elo_k,
        'ucb_c': ucb_c,
        'n_tournament_rounds': n_tournament_rounds,
        'train_time': train_time,
        'avg_selection_time': np.mean(selection_times),
        'avg_regret': np.mean(regrets),
        'top1_accuracy': top1_correct / len(test_data),
        'top3_accuracy': top3_correct / len(test_data)
    }

    print(f"\nResults:")
    print(f"  Training time: {train_time:.2f}s")
    print(f"  Avg selection time: {np.mean(selection_times)*1000:.2f}ms")
    print(f"  Avg regret: {np.mean(regrets):.4f}")
    print(f"  Top-1 accuracy: {results['top1_accuracy']*100:.1f}%")
    print(f"  Top-3 accuracy: {results['top3_accuracy']*100:.1f}%")

    return results

def main():
    print("="*70)
    print("Optimal Configuration Validation on Real Data")
    print("="*70)

    scenario_name = 'SAT11-HAND'

    # Test default configuration
    default_results = test_config(
        scenario_name=scenario_name,
        n_clusters=5,
        elo_k=32.0,
        ucb_c=1.414,
        n_tournament_rounds=5,
        config_name='Default'
    )

    # Test optimal configuration (from Week 4 ablation studies)
    optimal_results = test_config(
        scenario_name=scenario_name,
        n_clusters=20,
        elo_k=16.0,
        ucb_c=0.5,
        n_tournament_rounds=10,
        config_name='Optimal'
    )

    # Compare results
    print(f"\n{'='*70}")
    print("Comparison: Default vs. Optimal Configuration")
    print(f"{'='*70}")

    comparison_df = pd.DataFrame([default_results, optimal_results])
    print("\n" + comparison_df.to_string(index=False))

    # Calculate improvements
    print(f"\n{'='*70}")
    print("Performance Improvements (Optimal vs. Default)")
    print(f"{'='*70}")

    regret_improvement = (default_results['avg_regret'] - optimal_results['avg_regret']) / default_results['avg_regret'] * 100
    top1_improvement = (optimal_results['top1_accuracy'] - default_results['top1_accuracy']) / default_results['top1_accuracy'] * 100
    top3_improvement = (optimal_results['top3_accuracy'] - default_results['top3_accuracy']) / default_results['top3_accuracy'] * 100

    print(f"Regret reduction: {regret_improvement:.1f}%")
    print(f"Top-1 accuracy improvement: {top1_improvement:.1f}%")
    print(f"Top-3 accuracy improvement: {top3_improvement:.1f}%")

    print(f"\n{'='*70}")
    print("Validation Complete!")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
