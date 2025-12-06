"""
Integration tests for advanced optimization methods

Tests all advanced algorithms with various problem sizes and configurations.
"""

import numpy as np
import pytest

from Librex import optimize
from Librex.adapters.qap import QAPAdapter
from Librex.core.interfaces import StandardizedProblem


def create_test_qap(n: int = 5) -> dict:
    """Create a small test QAP instance"""
    np.random.seed(42)
    flow = np.random.randint(1, 10, (n, n))
    distance = np.random.randint(1, 10, (n, n))
    # Make symmetric
    flow = (flow + flow.T) // 2
    distance = (distance + distance.T) // 2
    # Zero diagonal
    np.fill_diagonal(flow, 0)
    np.fill_diagonal(distance, 0)

    return {
        'flow_matrix': flow,
        'distance_matrix': distance
    }


def create_standardized_problem(n: int = 5) -> StandardizedProblem:
    """Create a standardized problem for testing"""
    np.random.seed(42)
    matrix = np.random.randint(1, 10, (n, n))

    def objective_function(x):
        cost = 0.0
        for i in range(n):
            for j in range(n):
                cost += matrix[i, j] * matrix[x[i], x[j]]
        return cost

    return StandardizedProblem(
        dimension=n,
        objective_matrix=matrix,
        objective_function=objective_function
    )


class TestAntColonyOptimization:
    """Test Ant Colony Optimization algorithm"""

    def test_aco_basic(self):
        """Test ACO with default configuration"""
        problem = create_test_qap(5)
        adapter = QAPAdapter()

        result = optimize(
            problem,
            adapter,
            method='ant_colony',
            config={'n_iterations': 10, 'n_ants': 5, 'seed': 42}
        )

        assert 'solution' in result
        assert 'objective' in result
        assert result['is_valid']
        assert result['objective'] > 0
        assert len(result['solution']) == 5

    def test_aco_standardized_problem(self):
        """Test ACO with standardized problem"""
        problem = create_standardized_problem(6)

        result = optimize(
            problem,
            method='ant_colony',
            config={'n_iterations': 15, 'n_ants': 10, 'seed': 42}
        )

        assert result['solution'] is not None
        assert result['objective'] > 0
        assert result['metadata']['method'] == 'ant_colony_optimization'

    def test_aco_convergence(self):
        """Test ACO convergence tracking"""
        problem = create_standardized_problem(4)

        result = optimize(
            problem,
            method='ant_colony',
            config={'n_iterations': 20, 'n_ants': 8, 'alpha': 1.5, 'beta': 2.5}
        )

        assert 'convergence' in result
        assert 'history' in result['convergence']
        assert len(result['convergence']['history']) > 0


class TestParticleSwarmOptimization:
    """Test Particle Swarm Optimization algorithm"""

    def test_pso_basic(self):
        """Test PSO with default configuration"""
        problem = create_test_qap(5)
        adapter = QAPAdapter()

        result = optimize(
            problem,
            adapter,
            method='particle_swarm',
            config={'n_iterations': 10, 'n_particles': 10, 'seed': 42}
        )

        assert 'solution' in result
        assert 'objective' in result
        assert result['is_valid']
        assert result['objective'] > 0

    def test_pso_parameters(self):
        """Test PSO with custom parameters"""
        problem = create_standardized_problem(5)

        result = optimize(
            problem,
            method='particle_swarm',
            config={
                'n_iterations': 15,
                'n_particles': 15,
                'w': 0.8,
                'c1': 1.5,
                'c2': 2.5,
                'seed': 42
            }
        )

        assert result['metadata']['method'] == 'particle_swarm_optimization'
        assert result['metadata']['n_particles'] == 15
        assert result['metadata']['c1'] == 1.5
        assert result['metadata']['c2'] == 2.5

    def test_pso_convergence(self):
        """Test PSO convergence"""
        problem = create_standardized_problem(4)

        result = optimize(
            problem,
            method='particle_swarm',
            config={'n_iterations': 30, 'n_particles': 20}
        )

        assert 'convergence' in result
        assert isinstance(result['convergence']['history'], list)


class TestVariableNeighborhoodSearch:
    """Test Variable Neighborhood Search algorithm"""

    def test_vns_basic(self):
        """Test VNS with default configuration"""
        problem = create_test_qap(5)
        adapter = QAPAdapter()

        result = optimize(
            problem,
            adapter,
            method='variable_neighborhood',
            config={'max_iterations': 10, 'k_max': 3, 'seed': 42}
        )

        assert 'solution' in result
        assert 'objective' in result
        assert result['is_valid']

    def test_vns_neighborhoods(self):
        """Test VNS with different neighborhood structures"""
        problem = create_standardized_problem(6)

        result = optimize(
            problem,
            method='variable_neighborhood',
            config={
                'max_iterations': 20,
                'k_max': 5,
                'local_search_iterations': 50,
                'seed': 42
            }
        )

        assert result['metadata']['method'] == 'variable_neighborhood_search'
        assert result['metadata']['k_max'] == 5
        assert result['convergence']['neighborhoods_explored'] == 5

    def test_vns_local_search(self):
        """Test VNS local search component"""
        problem = create_standardized_problem(5)

        result = optimize(
            problem,
            method='variable_neighborhood',
            config={
                'max_iterations': 15,
                'local_search_iterations': 100
            }
        )

        assert result['iterations'] > 0
        assert result['metadata']['local_search_iterations'] == 100


class TestIteratedLocalSearch:
    """Test Iterated Local Search algorithm"""

    def test_ils_basic(self):
        """Test ILS with default configuration"""
        problem = create_test_qap(5)
        adapter = QAPAdapter()

        result = optimize(
            problem,
            adapter,
            method='iterated_local_search',
            config={'max_iterations': 10, 'seed': 42}
        )

        assert 'solution' in result
        assert 'objective' in result
        assert result['is_valid']

    def test_ils_perturbation(self):
        """Test ILS perturbation strategies"""
        problem = create_standardized_problem(6)

        result = optimize(
            problem,
            method='iterated_local_search',
            config={
                'max_iterations': 15,
                'perturbation_strength': 4,
                'local_search_iterations': 50,
                'seed': 42
            }
        )

        assert result['metadata']['method'] == 'iterated_local_search'
        assert 'perturbation_strength_final' in result['metadata']

    def test_ils_acceptance_criteria(self):
        """Test ILS with different acceptance criteria"""
        problem = create_standardized_problem(5)

        # Test 'better' acceptance
        result1 = optimize(
            problem,
            method='iterated_local_search',
            config={'max_iterations': 10, 'acceptance': 'better', 'seed': 42}
        )
        assert result1['metadata']['acceptance_criterion'] == 'better'

        # Test 'threshold' acceptance
        result2 = optimize(
            problem,
            method='iterated_local_search',
            config={
                'max_iterations': 10,
                'acceptance': 'threshold',
                'threshold': 0.05,
                'seed': 42
            }
        )
        assert result2['metadata']['acceptance_criterion'] == 'threshold'

        # Test 'simulated_annealing' acceptance
        result3 = optimize(
            problem,
            method='iterated_local_search',
            config={
                'max_iterations': 10,
                'acceptance': 'simulated_annealing',
                'temperature': 50.0,
                'cooling_rate': 0.9,
                'seed': 42
            }
        )
        assert result3['metadata']['acceptance_criterion'] == 'simulated_annealing'


class TestGRASP:
    """Test GRASP algorithm"""

    def test_grasp_basic(self):
        """Test GRASP with default configuration"""
        problem = create_test_qap(5)
        adapter = QAPAdapter()

        result = optimize(
            problem,
            adapter,
            method='grasp',
            config={'max_iterations': 10, 'seed': 42}
        )

        assert 'solution' in result
        assert 'objective' in result
        assert result['is_valid']

    def test_grasp_construction(self):
        """Test GRASP construction phase"""
        problem = create_standardized_problem(6)

        result = optimize(
            problem,
            method='grasp',
            config={
                'max_iterations': 15,
                'alpha': 0.3,
                'local_search_iterations': 50,
                'seed': 42
            }
        )

        assert result['metadata']['method'] == 'grasp'
        assert result['metadata']['alpha'] == 0.3

    def test_grasp_path_relinking(self):
        """Test GRASP with path relinking"""
        problem = create_standardized_problem(5)

        result = optimize(
            problem,
            method='grasp',
            config={
                'max_iterations': 20,
                'path_relinking': True,
                'elite_size': 5,
                'seed': 42
            }
        )

        assert result['metadata']['path_relinking'] is True
        assert result['metadata']['elite_solutions'] >= 0


class TestAdvancedMethodsComparison:
    """Compare performance of advanced methods"""

    def test_all_methods_same_problem(self):
        """Test all advanced methods on the same problem"""
        problem = create_test_qap(6)
        adapter = QAPAdapter()

        methods = [
            'ant_colony',
            'particle_swarm',
            'variable_neighborhood',
            'iterated_local_search',
            'grasp'
        ]

        results = {}
        for method in methods:
            result = optimize(
                problem,
                adapter,
                method=method,
                config={'max_iterations': 10, 'seed': 42}
            )
            results[method] = result

            # Basic assertions
            assert result['solution'] is not None
            assert result['objective'] > 0
            assert result['is_valid']

        # Check that different methods produce reasonable results
        objectives = [r['objective'] for r in results.values()]
        assert len(set(objectives)) > 1  # Methods should produce different results

    def test_scalability(self):
        """Test scalability with different problem sizes"""
        sizes = [4, 6, 8]

        for n in sizes:
            problem = create_standardized_problem(n)

            # Test with a fast method configuration
            result = optimize(
                problem,
                method='ant_colony',
                config={'n_iterations': 5, 'n_ants': 5, 'seed': 42}
            )

            assert result['solution'] is not None
            assert len(result['solution']) == n
            assert result['objective'] > 0


# Performance test (marked as slow)
@pytest.mark.slow
def test_advanced_methods_performance():
    """Test performance of advanced methods on larger problems"""
    problem = create_standardized_problem(20)

    methods_configs = [
        ('ant_colony', {'n_iterations': 50, 'n_ants': 20}),
        ('particle_swarm', {'n_iterations': 50, 'n_particles': 30}),
        ('variable_neighborhood', {'max_iterations': 100, 'k_max': 5}),
        ('iterated_local_search', {'max_iterations': 50}),
        ('grasp', {'max_iterations': 50})
    ]

    for method, config in methods_configs:
        config['seed'] = 42
        result = optimize(problem, method=method, config=config)

        assert result['solution'] is not None
        assert result['objective'] > 0
        assert result['iterations'] > 0

        # Check convergence information
        if 'convergence' in result:
            assert 'history' in result['convergence']
            assert len(result['convergence']['history']) > 0