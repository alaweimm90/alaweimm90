"""
Test Real ASlib Data Integration

Quick test to verify that the evaluation pipeline works with real ASlib data.
"""

from benchmark.run_evaluation import ComprehensiveEvaluator

def main():
    print("="*70)
    print("Testing Real ASlib Data Integration")
    print("="*70)

    # Initialize evaluator
    evaluator = ComprehensiveEvaluator(
        aslib_root="aslib_data",
        results_dir="results",
        n_folds=10
    )

    # Test with SAT11-HAND scenario
    print("\nTesting SAT11-HAND scenario with real data...")

    results = evaluator.run_single_scenario(
        scenario_name='SAT11-HAND',
        use_mock_data=False  # Use real ASlib data
    )

    # Print summary
    print("\n" + "="*70)
    print("Results Summary")
    print("="*70)
    print(results['summary'].to_string(index=False))

    print("\n" + "="*70)
    print("Integration Test Complete!")
    print("="*70)

if __name__ == "__main__":
    main()
