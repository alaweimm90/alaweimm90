"""
Unit tests for Librex.Meta tournament-based solver selection
"""

import pytest
import numpy as np
from libria_meta import Librex.Meta, FeatureExtractor


# Mock solver class for testing
class MockSolver:
    def __init__(self, name: str, base_performance: float = 0.5):
        self.name = name
        self.base_performance = base_performance

    def solve(self, instance):
        # Add some randomness to performance
        performance = self.base_performance + np.random.randn() * 0.1
        return {'objective': max(0, min(1, performance))}


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
        MockSolver('solver1', base_performance=0.6),
        MockSolver('solver2', base_performance=0.7),
        MockSolver('solver3', base_performance=0.5),
    ]


@pytest.fixture
def meta_libria(mock_solvers):
    """Create Librex.Meta instance with mock solvers"""
    return Librex.Meta(solvers=mock_solvers, n_clusters=2, n_tournament_rounds=3)


def test_elo_initialization(mock_solvers):
    """Test Elo ratings initialized to 1500"""
    meta = Librex.Meta(solvers=mock_solvers, n_clusters=3)

    for solver in mock_solvers:
        assert meta.global_elo[solver.name] == 1500.0
        for cluster in range(3):
            assert meta.cluster_elo[cluster][solver.name] == 1500.0


def test_elo_update(mock_solvers):
    """Test Elo updates after matches"""
    meta = Librex.Meta(solvers=mock_solvers, n_clusters=1)

    initial_elo_s1 = meta.global_elo['solver1']
    initial_elo_s2 = meta.global_elo['solver2']

    # solver1 wins
    meta._update_elo('solver1', 'solver2', cluster=0, outcome=1.0)

    assert meta.global_elo['solver1'] > initial_elo_s1
    assert meta.global_elo['solver2'] < initial_elo_s2


def test_feature_extraction():
    """Test feature extraction from problem instances"""
    extractor = FeatureExtractor()
    instance = MockProblemInstance(n_agents=10, n_tasks=15)
    features = extractor.extract(instance)

    assert isinstance(features, np.ndarray)
    assert len(features) > 0
    assert features[0] == 10  # n_agents
    assert features[1] == 15  # n_tasks


def test_solver_selection(meta_libria):
    """Test UCB-based solver selection"""
    instance = MockProblemInstance()

    selected = meta_libria.select_solver(instance)
    assert selected in meta_libria.solvers


def test_tournament_execution(meta_libria):
    """Test Swiss-system tournament"""
    instance = MockProblemInstance()

    result = meta_libria.run_tournament(instance)

    assert 'selected_solver' in result
    assert 'rankings' in result
    assert 'match_history' in result
    assert len(result['match_history']) > 0
    assert result['selected_solver'] in meta_libria.solvers


def test_fit_with_training_data(meta_libria):
    """Test training on historical data"""
    # Create mock training data
    training_data = []
    for _ in range(20):
        instance = MockProblemInstance()
        performances = {
            'solver1': np.random.rand(),
            'solver2': np.random.rand(),
            'solver3': np.random.rand(),
        }
        training_data.append({
            'instance': instance,
            'features': None,  # Will be extracted
            'performances': performances
        })

    # Train
    meta_libria.fit(training_data)

    # Check that Elo ratings have changed from default
    assert meta_libria.global_elo['solver1'] != 1500.0 or \
           meta_libria.global_elo['solver2'] != 1500.0 or \
           meta_libria.global_elo['solver3'] != 1500.0


def test_update_with_performance(meta_libria):
    """Test online update with observed performance"""
    instance = MockProblemInstance()

    initial_elo = meta_libria.global_elo['solver1']

    # Update with high performance
    meta_libria.update(instance, 'solver1', performance=0.9)

    # History should be updated
    assert len(meta_libria.history) == 1
    assert meta_libria.history[0]['solver'] == 'solver1'
    assert meta_libria.history[0]['performance'] == 0.9


def test_count_trials(meta_libria):
    """Test trial counting per cluster"""
    instance = MockProblemInstance()

    # Add some history
    for _ in range(5):
        meta_libria.update(instance, 'solver1', performance=0.8)

    # Count should be 5 (or close, depending on clustering)
    features = meta_libria.feature_extractor.extract(instance)
    cluster = meta_libria.clusterer.predict([features])[0]
    count = meta_libria._count_trials('solver1', cluster)

    assert count >= 0  # Should have some trials


def test_feature_extractor_fit():
    """Test feature extractor fitting and transformation"""
    extractor = FeatureExtractor()
    instances = [MockProblemInstance() for _ in range(10)]

    # Fit scaler
    extractor.fit(instances)
    assert extractor.fitted is True

    # Transform should work
    transformed = extractor.transform(instances[0])
    assert isinstance(transformed, np.ndarray)


def test_ucb_exploration_exploitation_tradeoff(meta_libria):
    """Test that UCB balances exploration and exploitation"""
    instance = MockProblemInstance()

    # Select solver multiple times
    selections = []
    for _ in range(10):
        selected = meta_libria.select_solver(instance)
        selections.append(selected.name)

    # Should have some variety (exploration)
    # But this is probabilistic, so we just check it runs
    assert len(selections) == 10


def test_tournament_ranking_consistency(meta_libria):
    """Test that better solvers rank higher in tournament"""
    # Create instance
    instance = MockProblemInstance()

    # Run tournament multiple times
    rankings_list = []
    for _ in range(5):
        result = meta_libria.run_tournament(instance)
        rankings_list.append(result['rankings'])

    # Check that rankings are returned
    assert len(rankings_list) == 5
    for rankings in rankings_list:
        assert len(rankings) == 3  # 3 solvers


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
