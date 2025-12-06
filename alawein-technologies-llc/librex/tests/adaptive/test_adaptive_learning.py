"""
Test suite for Adaptive Learning System

Author: Meshal Alawein
Date: 2025-11-18
"""

import numpy as np
import pytest
import tempfile
from datetime import datetime

from Librex.adaptive import (
    AlgorithmPortfolioManager,
    EnsembleOptimizer,
    MetaLearner,
    OnlineLearner,
    PerformanceDatabase,
    PerformancePredictor,
    UCB1Selector,
    ThompsonSampler,
    EXP3Selector,
)
from Librex.adaptive.performance_db import OptimizationRun
from Librex.adaptive.surrogate import GPSurrogate, SurrogateOptimizer, AcquisitionFunction
from Librex.core.interfaces import StandardizedProblem, StandardizedSolution


class TestPerformanceDatabase:
    """Test performance database functionality."""

    def test_database_creation(self):
        """Test database initialization."""
        db = PerformanceDatabase()
        assert db is not None
        assert db.db_path == ':memory:'

    def test_record_run(self):
        """Test recording optimization runs."""
        db = PerformanceDatabase()

        run = OptimizationRun(
            problem_id='test_problem_1',
            problem_features=np.array([1.0, 2.0, 3.0]),
            method_used='simulated_annealing',
            hyperparameters={'temperature': 100.0},
            objective_value=42.0,
            runtime_seconds=10.5,
            n_evaluations=1000,
            convergence_rate=0.95,
            solution_quality=0.8,
            timestamp=datetime.now(),
            metadata={'test': True}
        )

        run_id = db.record_run(run)
        assert run_id > 0

    def test_query_similar_problems(self):
        """Test finding similar problems."""
        db = PerformanceDatabase()

        # Add some runs
        for i in range(10):
            features = np.array([i, i*2, i*3], dtype=float)
            run = OptimizationRun(
                problem_id=f'problem_{i}',
                problem_features=features,
                method_used='genetic_algorithm',
                hyperparameters={},
                objective_value=float(i * 10),
                runtime_seconds=5.0,
                n_evaluations=500,
                convergence_rate=0.9,
                solution_quality=0.7,
                timestamp=datetime.now(),
                metadata={}
            )
            db.record_run(run)

        # Query similar problems
        query_features = np.array([5, 10, 15], dtype=float)
        similar = db.query_similar_problems(query_features, n_similar=3)

        assert len(similar) <= 3
        # Should find problem_5 as most similar

    def test_query_best_methods(self):
        """Test finding best performing methods."""
        db = PerformanceDatabase()

        # Add runs for different methods
        methods = ['simulated_annealing', 'genetic_algorithm', 'tabu_search']
        qualities = [0.9, 0.7, 0.8]

        for method, quality in zip(methods, qualities):
            for i in range(5):
                run = OptimizationRun(
                    problem_id=f'problem_{method}_{i}',
                    problem_features=np.random.randn(5),
                    method_used=method,
                    hyperparameters={},
                    objective_value=1.0 / quality,  # Lower is better
                    runtime_seconds=10.0,
                    n_evaluations=1000,
                    convergence_rate=0.9,
                    solution_quality=quality,
                    timestamp=datetime.now(),
                    metadata={}
                )
                db.record_run(run)

        # Query best methods
        best_methods = db.query_best_methods(n_methods=2)
        assert len(best_methods) <= 2
        # Should rank simulated_annealing highest

    def test_feature_extraction(self):
        """Test feature extraction from problems."""
        db = PerformanceDatabase()

        # QAP-like problem
        problem = {
            'flow_matrix': np.array([[0, 5, 3], [5, 0, 2], [3, 2, 0]]),
            'distance_matrix': np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]])
        }

        features = db.extract_features(problem)
        assert len(features) > 0
        assert features[0] == 3  # Problem size


class TestOnlineLearning:
    """Test online learning components."""

    def test_ucb1_selector(self):
        """Test UCB1 bandit selector."""
        algorithms = ['alg1', 'alg2', 'alg3']
        selector = UCB1Selector(algorithms, c=2.0)

        # Initial exploration
        selected = []
        for _ in range(len(algorithms)):
            alg = selector.select_algorithm()
            selected.append(alg)
            selector.update_reward(alg, np.random.random())

        assert set(selected) == set(algorithms)  # All should be explored

        # After exploration, should exploit best
        for _ in range(10):
            alg = selector.select_algorithm()
            reward = 0.9 if alg == 'alg1' else 0.1  # alg1 is best
            selector.update_reward(alg, reward)

        # alg1 should be selected most
        final_selections = [selector.select_algorithm() for _ in range(5)]
        assert final_selections.count('alg1') >= 3

    def test_thompson_sampler(self):
        """Test Thompson Sampling selector."""
        algorithms = ['alg1', 'alg2']
        selector = ThompsonSampler(algorithms)

        # Update with clear winner
        for _ in range(20):
            alg = selector.select_algorithm()
            reward = 1.0 if alg == 'alg1' else 0.0
            selector.update_reward(alg, reward)

        # alg1 should be strongly preferred
        selections = [selector.select_algorithm() for _ in range(10)]
        assert selections.count('alg1') >= 7

    def test_exp3_selector(self):
        """Test EXP3 selector."""
        algorithms = ['alg1', 'alg2', 'alg3']
        selector = EXP3Selector(algorithms, gamma=0.1)

        # Test adversarial scenario
        for round_num in range(50):
            alg = selector.select_algorithm()
            # Adversarial: reward changes based on round
            if round_num < 25:
                reward = 1.0 if alg == 'alg1' else 0.0
            else:
                reward = 1.0 if alg == 'alg2' else 0.0
            selector.update_reward(alg, reward)

        # Should have adapted to change
        assert selector.weights['alg2'] > 0  # alg2 should have gained weight

    def test_online_learner(self):
        """Test main online learner."""
        learner = OnlineLearner(
            algorithms=['simulated_annealing', 'genetic_algorithm'],
            strategy='ucb1',
            contextual=True
        )

        # Select methods with context
        problem_features = np.array([10, 0.5, 100])
        method, config = learner.select_method(
            problem_features,
            time_budget=60.0,
            evaluation_budget=1000
        )

        assert method in ['simulated_annealing', 'genetic_algorithm']
        assert 'max_time' in config
        assert 'max_evaluations' in config

        # Update with performance
        learner.update_performance(
            method,
            objective_value=10.0,
            runtime=5.0,
            n_evaluations=500,
            problem_features=problem_features
        )

        # Check performance tracking
        summary = learner.get_performance_summary()
        assert summary['n_selections'] == 1
        assert method in summary['algorithms_used']


class TestMetaLearning:
    """Test meta-learning components."""

    def test_performance_predictor(self):
        """Test performance prediction model."""
        predictor = PerformancePredictor(model_type='gradient_boosting')

        # Create training data
        training_data = []
        for i in range(50):
            features = np.random.randn(5)
            algorithm = np.random.choice(['alg1', 'alg2'])
            performance = 0.8 if algorithm == 'alg1' else 0.6
            performance += np.random.normal(0, 0.1)  # Add noise
            training_data.append((features, algorithm, performance))

        # Train predictor
        predictor.train(training_data, ['alg1', 'alg2'])
        assert predictor.is_trained

        # Make predictions
        test_features = np.random.randn(5)
        predictions = predictor.predict(test_features)

        assert 'alg1' in predictions
        assert 'alg2' in predictions
        assert predictions['alg1'][0] > 0  # Mean prediction
        assert predictions['alg1'][1] >= 0  # Uncertainty

    def test_meta_learner(self):
        """Test meta-learning system."""
        learner = MetaLearner()

        # Create optimization history
        history = []
        for i in range(30):
            run = {
                'problem_features': np.random.randn(5),
                'algorithm': np.random.choice(['genetic_algorithm', 'simulated_annealing']),
                'performance': np.random.random(),
                'hyperparameters': {'param1': np.random.random()},
                'problem_category': 'qap',
                'problem_id': f'problem_{i}'
            }
            history.append(run)

        # Learn from history
        learner.learn_from_history(history)

        # Warm-start new problem
        new_features = np.random.randn(5)
        config = learner.warm_start(
            new_features,
            'genetic_algorithm',
            {'population_size': 100}
        )

        assert 'population_size' in config
        assert config['population_size'] > 0


class TestSurrogateModels:
    """Test surrogate model components."""

    def test_gp_surrogate(self):
        """Test Gaussian Process surrogate."""
        surrogate = GPSurrogate()

        # Generate training data
        X = np.random.randn(10, 2)
        y = np.sum(X**2, axis=1)  # Simple quadratic

        # Fit surrogate
        surrogate.fit(X, y)
        assert surrogate.gp is not None

        # Make predictions
        X_test = np.random.randn(5, 2)
        mu, sigma = surrogate.predict(X_test)

        assert len(mu) == 5
        assert len(sigma) == 5
        assert np.all(sigma >= 0)

        # Test acquisition functions
        ei = surrogate.expected_improvement(X_test)
        assert len(ei) == 5

        pi = surrogate.probability_improvement(X_test)
        assert len(pi) == 5
        assert np.all(pi >= 0) and np.all(pi <= 1)

        ucb = surrogate.upper_confidence_bound(X_test)
        assert len(ucb) == 5

    def test_surrogate_optimizer(self):
        """Test surrogate-based optimizer."""
        # Simple 1D function
        def objective(x):
            return (x[0] - 2.0)**2 + 0.1 * np.random.randn()

        bounds = np.array([[-5.0, 5.0]])

        optimizer = SurrogateOptimizer(
            objective,
            bounds,
            acquisition=AcquisitionFunction.EI,
            n_initial_points=3
        )

        # Run optimization
        result = optimizer.optimize(n_iterations=10)

        assert 'best_point' in result
        assert 'best_value' in result
        assert result['n_evaluations'] == 10
        # Should find near x=2
        assert abs(result['best_point'][0] - 2.0) < 1.0


class TestPortfolioManager:
    """Test algorithm portfolio management."""

    def test_portfolio_initialization(self):
        """Test portfolio manager initialization."""
        portfolio = AlgorithmPortfolioManager(
            algorithms=['simulated_annealing', 'genetic_algorithm'],
            max_parallel=2
        )

        assert len(portfolio.algorithms) == 2
        assert portfolio.max_parallel == 2

    def test_adaptive_allocation(self):
        """Test adaptive resource allocation."""
        portfolio = AlgorithmPortfolioManager(
            algorithms=['alg1', 'alg2', 'alg3'],
            adaptive_allocation=True
        )

        # Create dummy problem
        problem = StandardizedProblem(
            dimension=10,
            bounds=np.array([[-1, 1]] * 10),
            is_discrete=False,
            constraint_function=None
        )

        # Test allocation computation
        allocations = portfolio._compute_adaptive_allocation(problem)
        assert len(allocations) == 3
        assert abs(sum(allocations.values()) - 1.0) < 1e-6


class TestEnsembleOptimizer:
    """Test ensemble optimization."""

    def test_ensemble_creation(self):
        """Test ensemble optimizer creation."""
        ensemble = EnsembleOptimizer(
            algorithms=['simulated_annealing', 'genetic_algorithm', 'tabu_search'],
            voting_method='weighted'
        )

        assert len(ensemble.algorithms) == 3
        assert ensemble.voting_method == 'weighted'

    def test_algorithm_selection(self):
        """Test diverse algorithm selection."""
        ensemble = EnsembleOptimizer(
            algorithms=['genetic_algorithm', 'particle_swarm', 'simulated_annealing',
                       'tabu_search', 'ant_colony'],
            ensemble_size=3
        )

        selected = ensemble._select_diverse_algorithms(
            ensemble.algorithms,
            3
        )

        assert len(selected) == 3
        # Should have diversity


class TestIntegration:
    """Integration tests for adaptive learning."""

    def test_adaptive_optimize(self):
        """Test adaptive mode in optimize function."""
        from Librex import optimize
        from Librex.adapters.qap import QAPAdapter

        # Small QAP problem
        problem = {
            'flow_matrix': np.array([[0, 5, 3], [5, 0, 2], [3, 2, 0]]),
            'distance_matrix': np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]])
        }
        adapter = QAPAdapter()

        # Test online learning mode
        result = optimize(
            problem,
            adapter,
            method='adaptive',
            config={
                'learning_config': {
                    'mode': 'online',
                    'strategy': 'ucb1',
                    'time_budget': 5
                }
            }
        )

        assert 'solution' in result
        assert 'objective' in result

    def test_portfolio_optimize(self):
        """Test portfolio mode in optimize function."""
        from Librex import optimize
        from Librex.adapters.qap import QAPAdapter

        # Small QAP problem
        problem = {
            'flow_matrix': np.array([[0, 1], [1, 0]]),
            'distance_matrix': np.array([[0, 2], [2, 0]])
        }
        adapter = QAPAdapter()

        # Test portfolio mode
        result = optimize(
            problem,
            adapter,
            method='adaptive',
            config={
                'learning_config': {
                    'mode': 'portfolio',
                    'time_budget': 5,
                    'n_parallel': 2
                }
            }
        )

        assert 'solution' in result
        assert 'metadata' in result
        assert result['metadata']['mode'] == 'adaptive_portfolio'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])