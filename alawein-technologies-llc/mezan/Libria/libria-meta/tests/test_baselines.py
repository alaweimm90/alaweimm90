"""
Tests for baseline algorithm selection methods

Tests all 5 baselines:
1. SATzilla (regression-based)
2. AutoFolio (automated portfolio configuration)
3. SMAC (Bayesian optimization)
4. Hyperband (successive halving)
5. BOHB (Bayesian + Hyperband)

Plus comparison with Librex.Meta
"""

import pytest
import numpy as np
from libria_meta import Librex.Meta
from baselines import SATzilla, AutoFolio, SMACBaseline, Hyperband, BOHB
from benchmark.evaluate_Librex.Meta import create_mock_solver_for_aslib


# Mock problem instance
class MockProblemInstance:
    def __init__(self, n_agents: int = 10, n_tasks: int = 15):
        self.n_agents = n_agents
        self.n_tasks = n_tasks
        self.cost_matrix = np.random.rand(n_agents, n_tasks)


@pytest.fixture
def mock_solvers():
    """Create mock solvers for testing"""
    return [
        create_mock_solver_for_aslib('Solver1', base_performance=0.6),
        create_mock_solver_for_aslib('Solver2', base_performance=0.7),
        create_mock_solver_for_aslib('Solver3', base_performance=0.5),
    ]


@pytest.fixture
def training_data(mock_solvers):
    """Generate training data"""
    data = []
    for _ in range(30):
        instance = MockProblemInstance()
        performances = {
            s.name: np.random.uniform(0.4, 0.9)
            for s in mock_solvers
        }
        data.append({
            'instance': instance,
            'features': None,
            'performances': performances
        })
    return data


def test_satzilla_basic(mock_solvers, training_data):
    """Test SATzilla basic functionality"""
    print("\n" + "="*60)
    print("Testing SATzilla")
    print("="*60)

    satzilla = SATzilla(solvers=mock_solvers, n_estimators=50)

    # Train
    satzilla.fit(training_data)

    # Test selection
    test_instance = MockProblemInstance()
    selected = satzilla.select_solver(test_instance)

    print(f"Selected solver: {selected.name}")

    assert selected in mock_solvers
    assert satzilla.fitted is True

    # Test prediction
    pred = satzilla.predict_performance(test_instance, 'Solver1')
    print(f"Predicted performance for Solver1: {pred:.3f}")
    assert 0.0 <= pred <= 1.0

    print("✓ SATzilla test PASSED")


def test_autofolio_basic(mock_solvers, training_data):
    """Test AutoFolio basic functionality"""
    print("\n" + "="*60)
    print("Testing AutoFolio")
    print("="*60)

    autofolio = AutoFolio(
        solvers=mock_solvers,
        n_features=10,
        use_feature_selection=True
    )

    # Train
    autofolio.fit(training_data)

    # Test selection
    test_instance = MockProblemInstance()
    selected = autofolio.select_solver(test_instance)

    print(f"Selected solver: {selected.name}")

    assert selected in mock_solvers
    assert autofolio.fitted is True

    # Test prediction
    predictions = autofolio.predict_all(test_instance)
    print(f"Predictions: {predictions}")

    assert len(predictions) == 3
    assert all(0.0 <= p <= 1.5 for p in predictions.values())

    print("✓ AutoFolio test PASSED")


def test_smac_basic(mock_solvers, training_data):
    """Test SMAC basic functionality"""
    print("\n" + "="*60)
    print("Testing SMAC")
    print("="*60)

    smac = SMACBaseline(solvers=mock_solvers, n_estimators=50)

    # Train
    smac.fit(training_data)

    # Test selection
    test_instance = MockProblemInstance()
    selected = smac.select_solver(test_instance)

    print(f"Selected solver: {selected.name}")

    assert selected in mock_solvers
    assert smac.fitted is True

    # Test online update
    smac.update(test_instance, selected.name, performance=0.8)
    print(f"History size after update: {len(smac.X_history)}")

    assert len(smac.X_history) > len(training_data) * 3  # 3 solvers

    print("✓ SMAC test PASSED")


def test_hyperband_basic(mock_solvers, training_data):
    """Test Hyperband basic functionality"""
    print("\n" + "="*60)
    print("Testing Hyperband")
    print("="*60)

    hyperband = Hyperband(solvers=mock_solvers, max_budget=27, eta=3)

    # Train
    hyperband.fit(training_data)

    # Test selection
    test_instance = MockProblemInstance()
    selected = hyperband.select_solver(test_instance)

    print(f"Selected solver: {selected.name}")

    assert selected in mock_solvers

    # Test successive halving
    best_solver, history = hyperband.run_successive_halving(test_instance)

    print(f"Successive halving winner: {best_solver.name}")
    print(f"Evaluation history length: {len(history)}")

    assert best_solver in mock_solvers
    assert len(history) > 0

    print("✓ Hyperband test PASSED")


def test_bohb_basic(mock_solvers, training_data):
    """Test BOHB basic functionality"""
    print("\n" + "="*60)
    print("Testing BOHB")
    print("="*60)

    bohb = BOHB(
        solvers=mock_solvers,
        max_budget=27,
        eta=3,
        min_points_in_model=10
    )

    # Train
    bohb.fit(training_data)

    # Test selection
    test_instance = MockProblemInstance()
    selected = bohb.select_solver(test_instance)

    print(f"Selected solver: {selected.name}")

    assert selected in mock_solvers
    assert len(bohb.observations) > 0

    # Test online update
    bohb.update(test_instance, selected.name, performance=0.85)

    print(f"Observations after update: {len(bohb.observations)}")

    print("✓ BOHB test PASSED")


def test_all_methods_comparison(mock_solvers, training_data):
    """Compare all methods on the same data"""
    print("\n" + "="*70)
    print("Baseline Methods Comparison")
    print("="*70)

    # Initialize all methods
    methods = {
        'Librex.Meta': Librex.Meta(
            solvers=mock_solvers,
            n_clusters=3,
            n_tournament_rounds=3
        ),
        'SATzilla': SATzilla(solvers=mock_solvers, n_estimators=50),
        'AutoFolio': AutoFolio(solvers=mock_solvers, n_features=10),
        'SMAC': SMACBaseline(solvers=mock_solvers, n_estimators=50),
        'Hyperband': Hyperband(solvers=mock_solvers, max_budget=27),
        'BOHB': BOHB(solvers=mock_solvers, max_budget=27)
    }

    # Train all methods
    print("\nTraining all methods...")
    for name, method in methods.items():
        print(f"\n  Training {name}...")
        method.fit(training_data)

    # Test on new instances
    print("\nEvaluating on test instances...")
    test_instances = [MockProblemInstance() for _ in range(10)]

    results = {name: [] for name in methods}

    for i, instance in enumerate(test_instances):
        print(f"\n  Test instance {i+1}/10:")

        # Get true performance (run all solvers)
        true_performances = {}
        for solver in mock_solvers:
            result = solver.solve(instance)
            true_performances[solver.name] = result['objective']

        best_true_solver = max(true_performances, key=true_performances.get)

        # Test each method
        for name, method in methods.items():
            selected = method.select_solver(instance)
            selected_name = selected.name if hasattr(selected, 'name') else str(selected)

            # Calculate regret
            regret = true_performances[best_true_solver] - true_performances[selected_name]
            results[name].append({
                'selected': selected_name,
                'correct': selected_name == best_true_solver,
                'regret': regret
            })

            print(f"    {name:15} -> {selected_name:10} (regret: {regret:+.3f})")

    # Summary statistics
    print("\n" + "="*70)
    print("Summary Statistics:")
    print("="*70)

    for name in methods:
        accuracy = np.mean([r['correct'] for r in results[name]])
        avg_regret = np.mean([r['regret'] for r in results[name]])

        print(f"\n{name}:")
        print(f"  Accuracy: {accuracy:.2%}")
        print(f"  Avg Regret: {avg_regret:+.4f}")

    print("\n" + "="*70)

    # All methods should make some valid selections
    for name in methods:
        assert len(results[name]) == 10
        assert all(r['selected'] in [s.name for s in mock_solvers] for r in results[name])


def test_methods_interface_consistency(mock_solvers):
    """Test that all methods have consistent interfaces"""
    print("\n" + "="*60)
    print("Testing Interface Consistency")
    print("="*60)

    methods = [
        Librex.Meta(solvers=mock_solvers, n_clusters=2),
        SATzilla(solvers=mock_solvers),
        AutoFolio(solvers=mock_solvers),
        SMACBaseline(solvers=mock_solvers),
        Hyperband(solvers=mock_solvers),
        BOHB(solvers=mock_solvers)
    ]

    # All methods should have these attributes/methods
    for method in methods:
        assert hasattr(method, 'name')
        assert hasattr(method, 'fit')
        assert hasattr(method, 'select_solver')

        print(f"✓ {method.name}: interface OK")

    print("\n✓ All methods have consistent interfaces")


def test_online_learning_comparison(mock_solvers, training_data):
    """Compare online learning capabilities"""
    print("\n" + "="*60)
    print("Online Learning Comparison")
    print("="*60)

    # Methods that support online learning
    online_methods = {
        'Librex.Meta': Librex.Meta(solvers=mock_solvers, n_clusters=2),
        'SMAC': SMACBaseline(solvers=mock_solvers),
        'BOHB': BOHB(solvers=mock_solvers)
    }

    # Train
    for name, method in online_methods.items():
        method.fit(training_data)

    # Test online updates
    test_instance = MockProblemInstance()

    for name, method in online_methods.items():
        selected = method.select_solver(test_instance)

        # Perform online update
        method.update(test_instance, selected.name, performance=0.9)

        print(f"{name}: updated with new observation")

    print("\n✓ Online learning test PASSED")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
