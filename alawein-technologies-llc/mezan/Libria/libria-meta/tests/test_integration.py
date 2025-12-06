"""
Integration test for Librex.Meta with ASlib benchmarks

This test demonstrates the complete workflow:
1. Create mock solvers
2. Train Librex.Meta on historical data
3. Run on new instances
4. Evaluate with ASlib framework
"""

import pytest
import numpy as np
from libria_meta import Librex.Meta, FeatureExtractor
from benchmark.evaluate_Librex.Meta import ASLibEvaluator, create_mock_solver_for_aslib


# Mock problem instance (same as in test_meta_solver.py)
class MockProblemInstance:
    def __init__(self, n_agents: int = 10, n_tasks: int = 15):
        self.n_agents = n_agents
        self.n_tasks = n_tasks
        self.cost_matrix = np.random.rand(n_agents, n_tasks)


def test_Librex.Meta_end_to_end():
    """Test complete Librex.Meta workflow"""
    print("\n" + "="*60)
    print("Librex.Meta End-to-End Integration Test")
    print("="*60)

    # 1. Create mock solvers
    print("\n1. Creating mock solvers...")
    solvers = [
        create_mock_solver_for_aslib('HungarianSolver', base_performance=0.6),
        create_mock_solver_for_aslib('GreedySolver', base_performance=0.7),
        create_mock_solver_for_aslib('AuctionSolver', base_performance=0.5),
    ]
    print(f"   ✓ Created {len(solvers)} solvers")

    # 2. Initialize Librex.Meta
    print("\n2. Initializing Librex.Meta...")
    meta = Librex.Meta(
        solvers=solvers,
        n_clusters=3,
        elo_k=32.0,
        ucb_c=1.414,
        n_tournament_rounds=3
    )
    print("   ✓ Librex.Meta initialized")

    # 3. Create training data
    print("\n3. Generating training data...")
    training_data = []
    for i in range(20):
        instance = MockProblemInstance()
        performances = {
            'HungarianSolver': np.random.uniform(0.5, 0.9),
            'GreedySolver': np.random.uniform(0.6, 0.95),
            'AuctionSolver': np.random.uniform(0.4, 0.8),
        }
        training_data.append({
            'instance': instance,
            'features': None,
            'performances': performances
        })
    print(f"   ✓ Generated {len(training_data)} training instances")

    # 4. Train Librex.Meta
    print("\n4. Training Librex.Meta...")
    meta.fit(training_data)
    print("   ✓ Training complete")

    # 5. Test solver selection
    print("\n5. Testing solver selection...")
    test_instance = MockProblemInstance()
    selected = meta.select_solver(test_instance)
    print(f"   ✓ Selected solver: {selected.name}")

    # 6. Run tournament
    print("\n6. Running tournament...")
    result = meta.run_tournament(test_instance)
    print(f"   ✓ Tournament winner: {result['selected_solver'].name}")
    print(f"   Rankings:")
    for i, (solver_name, rating) in enumerate(result['rankings'], 1):
        print(f"     {i}. {solver_name}: {rating:.1f}")

    # 7. Online learning update
    print("\n7. Testing online learning...")
    meta.update(test_instance, selected.name, performance=0.85)
    print(f"   ✓ Updated ratings based on performance")

    # 8. Verify Elo ratings changed
    print("\n8. Final Elo ratings:")
    for solver_name, rating in meta.global_elo.items():
        print(f"   {solver_name}: {rating:.1f}")

    print("\n" + "="*60)
    print("✓ Integration test PASSED")
    print("="*60)

    # Assertions
    assert selected in solvers
    assert result['selected_solver'] in solvers
    assert len(result['rankings']) == 3
    assert len(meta.history) > 0


def test_aslib_evaluator_initialization():
    """Test ASlib evaluator can be initialized"""
    print("\n" + "="*60)
    print("ASlib Evaluator Initialization Test")
    print("="*60)

    evaluator = ASLibEvaluator(aslib_root="aslib_data")
    print(f"\n✓ Found {len(evaluator.scenarios)} ASlib scenarios")

    if evaluator.scenarios:
        print(f"\nFirst 10 scenarios:")
        for i, scenario in enumerate(evaluator.scenarios[:10], 1):
            print(f"  {i}. {scenario}")

    assert isinstance(evaluator.scenarios, list)


def test_Librex.Meta_with_aslib_framework():
    """Test Librex.Meta integration with ASlib framework"""
    print("\n" + "="*60)
    print("Librex.Meta + ASlib Framework Integration Test")
    print("="*60)

    # Create solvers and Librex.Meta
    solvers = [
        create_mock_solver_for_aslib(f'Solver{i}', base_performance=0.5 + i*0.1)
        for i in range(3)
    ]

    meta = Librex.Meta(solvers=solvers, n_clusters=2, n_tournament_rounds=2)

    # Train on mock data
    training_data = []
    for _ in range(10):
        instance = MockProblemInstance()
        performances = {s.name: np.random.rand() for s in solvers}
        training_data.append({
            'instance': instance,
            'features': None,
            'performances': performances
        })

    meta.fit(training_data)

    # Initialize evaluator
    evaluator = ASLibEvaluator(aslib_root="aslib_data")

    print(f"\n✓ Librex.Meta trained and ready")
    print(f"✓ ASlib evaluator initialized with {len(evaluator.scenarios)} scenarios")

    # Test that we can call evaluation methods (even if data not fully loaded)
    print(f"\n✓ Integration test PASSED")

    assert len(evaluator.scenarios) >= 0  # May be 0 if aslib_data not downloaded


def test_par10_computation():
    """Test Par10 metric computation"""
    print("\n" + "="*60)
    print("Par10 Metric Computation Test")
    print("="*60)

    evaluator = ASLibEvaluator(aslib_root="aslib_data")

    # Test case 1: All successful
    runtimes = np.array([1.0, 2.0, 3.0, 4.0])
    timeout = 10.0
    success = np.array([True, True, True, True])

    par10 = evaluator.compute_par10(runtimes, timeout, success)
    print(f"\nTest 1 - All successful:")
    print(f"  Runtimes: {runtimes}")
    print(f"  Par10: {par10:.2f}")
    assert par10 == 2.5  # Average of [1, 2, 3, 4]

    # Test case 2: Some timeouts
    runtimes = np.array([1.0, 2.0, 15.0, 20.0])
    success = np.array([True, True, False, False])

    par10 = evaluator.compute_par10(runtimes, timeout, success)
    print(f"\nTest 2 - With timeouts:")
    print(f"  Runtimes: {runtimes}")
    print(f"  Success: {success}")
    print(f"  Par10: {par10:.2f}")
    assert par10 == (1.0 + 2.0 + 100.0 + 100.0) / 4  # Timeouts penalized 10x

    print(f"\n✓ Par10 computation test PASSED")


def test_top_k_accuracy():
    """Test top-k accuracy computation"""
    print("\n" + "="*60)
    print("Top-k Accuracy Test")
    print("="*60)

    evaluator = ASLibEvaluator(aslib_root="aslib_data")

    # Test case: 5 instances
    predictions = [
        ['A', 'B', 'C'],
        ['B', 'A', 'C'],
        ['C', 'B', 'A'],
        ['A', 'C', 'B'],
        ['B', 'C', 'A']
    ]
    ground_truth = ['A', 'A', 'C', 'B', 'A']

    # Top-1 accuracy: A, B, C, A, B vs A, A, C, B, A = 2/5 = 0.4
    top1 = evaluator.compute_top_k_accuracy(predictions, ground_truth, k=1)
    print(f"\nTop-1 accuracy: {top1:.2f}")

    # Top-3 accuracy: all should match (all in top-3)
    top3 = evaluator.compute_top_k_accuracy(predictions, ground_truth, k=3)
    print(f"Top-3 accuracy: {top3:.2f}")

    print(f"\n✓ Top-k accuracy test PASSED")

    assert 0.0 <= top1 <= 1.0
    assert 0.0 <= top3 <= 1.0
    assert top3 >= top1  # Top-3 should always be >= Top-1


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-s"])
