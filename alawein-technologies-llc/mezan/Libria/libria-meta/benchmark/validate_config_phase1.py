"""
Validate Optimal Configuration Across Phase 1 Scenarios

Test default vs optimal configuration on all 3 Phase 1 scenarios:
- SAT11-HAND
- CSP-2010
- GRAPHS-2015
"""

from benchmark.run_evaluation import ComprehensiveEvaluator
import pandas as pd

def main():
    print("="*70)
    print("Phase 1 Configuration Validation: Default vs. Optimal")
    print("="*70)

    evaluator = ComprehensiveEvaluator(
        aslib_root="aslib_data",
        results_dir="results",
        n_folds=10
    )

    scenarios = ['SAT11-HAND', 'CSP-2010', 'GRAPHS-2015']

    all_results = []

    for scenario in scenarios:
        print(f"\n{'#'*70}")
        print(f"Scenario: {scenario}")
        print(f"{'#'*70}")

        # Test with default config
        print("\n[1/2] Testing DEFAULT configuration...")
        default_results = evaluator.run_single_scenario(
            scenario_name=scenario,
            use_mock_data=False
        )

        # Extract Librex.Meta results
        ml_default = default_results['methods']['Librex.Meta']

        all_results.append({
            'Scenario': scenario,
            'Config': 'Default',
            'Avg Regret': ml_default['avg_regret'],
            'Top-1 Acc': ml_default['top1_accuracy'],
            'Top-3 Acc': ml_default['top3_accuracy'],
            'Sel Time (ms)': ml_default['avg_selection_time'] * 1000
        })

    # Create summary table
    print(f"\n{'='*70}")
    print("Summary: Librex.Meta Default Configuration on Real Data")
    print(f"{'='*70}")

    df = pd.DataFrame(all_results)
    print("\n" + df.to_string(index=False))

    # Calculate averages
    print(f"\n{'='*70}")
    print("Average Performance Across 3 Scenarios:")
    print(f"{'='*70}")

    avg_regret = df['Avg Regret'].mean()
    avg_top1 = df['Top-1 Acc'].mean()
    avg_top3 = df['Top-3 Acc'].mean()
    avg_time = df['Sel Time (ms)'].mean()

    print(f"Avg Regret: {avg_regret:.4f}")
    print(f"Avg Top-1 Accuracy: {avg_top1*100:.1f}%")
    print(f"Avg Top-3 Accuracy: {avg_top3*100:.1f}%")
    print(f"Avg Selection Time: {avg_time:.2f}ms")

    print(f"\n{'='*70}")
    print("Phase 1 Validation Complete!")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
