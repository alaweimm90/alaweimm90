"""Integration tests for Neural Architecture Search."""

import pytest
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from Librex.domains.nas import (
    NASCell, MacroArchitecture, NASProblem, NASAdapter,
    evolutionary_nas, differentiable_nas, bayesian_optimization_nas,
    random_search_nas, HardwareAwareNAS
)
from Librex.domains.nas.architecture import Operation, OperationType
from Librex.domains.nas.zero_cost_proxies import (
    grad_norm_proxy, jacob_cov_proxy, ntk_proxy, ProxyEnsemble
)


class TestNASIntegration:
    """Integration tests for NAS functionality."""

    def test_cell_architecture_creation(self):
        """Test creating and manipulating cell architectures."""
        cell = NASCell(n_nodes=4, n_inputs=2)

        # Add edges
        op1 = Operation(op_type=OperationType.CONV_3x3, channels=64)
        op2 = Operation(op_type=OperationType.SKIP_CONNECT)

        cell.add_edge(0, 2, op1)
        cell.add_edge(1, 2, op2)

        assert len(cell.edges) == 2
        assert cell.edges[0].from_node == 0
        assert cell.edges[0].to_node == 2

        # Test encoding/decoding
        encoding = cell.to_encoding()
        assert isinstance(encoding, np.ndarray)

        cell2 = NASCell(n_nodes=4, n_inputs=2)
        cell2.from_encoding(encoding)

        # Check parameter estimation
        params = cell.get_params()
        assert params > 0

        # Check FLOPs estimation
        flops = cell.get_flops(input_size=32)
        assert flops >= 0

    def test_macro_architecture_creation(self):
        """Test creating and manipulating macro architectures."""
        from Librex.domains.nas.architecture import Layer

        arch = MacroArchitecture(max_layers=20, input_channels=3, num_classes=10)

        # Add layers
        arch.layers = []
        arch.add_layer(Layer('conv', 64, kernel_size=3))
        arch.add_layer(Layer('pool', 64))
        arch.add_layer(Layer('conv', 128, kernel_size=3))
        arch.add_layer(Layer('fc', 10))

        assert len(arch.layers) == 4
        assert arch.get_depth() == 4

        # Add skip connection
        arch.add_skip_connection(0, 2)
        assert len(arch.skip_connections) == 1

        # Test encoding
        encoding = arch.to_encoding()
        assert 'depth' in encoding
        assert 'types' in encoding
        assert 'channels' in encoding

        # Test flat encoding
        flat = arch.to_flat_encoding()
        assert isinstance(flat, np.ndarray)

        arch2 = MacroArchitecture(max_layers=20)
        arch2.from_flat_encoding(flat)

        # Check parameters
        params = arch.get_params()
        assert params > 0

    def test_nas_problem_setup(self):
        """Test NAS problem configuration."""
        # Single objective
        problem = NASProblem(
            dataset='cifar10',
            search_space='cell',
            objectives=['accuracy']
        )

        assert problem.dataset == 'cifar10'
        assert len(problem.objectives) == 1
        assert not problem.is_multi_objective()

        # Multi-objective with constraints
        problem2 = NASProblem(
            dataset='imagenet',
            search_space='macro',
            objectives=['accuracy', 'latency_ms'],
            constraints={'max_params': 5e6, 'max_latency_ms': 10}
        )

        assert problem2.is_multi_objective()
        assert len(problem2.constraints) == 2

        # Test architecture creation
        arch = problem.create_architecture()
        assert isinstance(arch, NASCell)

        arch2 = problem2.create_architecture()
        assert isinstance(arch2, MacroArchitecture)

        # Test evaluation
        metrics = problem.evaluate_architecture(arch, return_all_metrics=True)
        assert 'accuracy' in metrics
        assert 'params' in metrics

    def test_nas_adapter(self):
        """Test NAS adapter for Librex integration."""
        problem = NASProblem(
            dataset='cifar10',
            search_space='cell',
            objectives=['accuracy']
        )

        adapter = NASAdapter()

        # Encode problem
        standardized = adapter.encode_problem(problem)
        assert standardized.dimension == problem.get_dimension()
        assert standardized.bounds is not None

        # Test objective function
        random_solution = np.random.uniform(0, 10, standardized.dimension)
        obj_value = standardized.objective_function(random_solution)
        assert isinstance(obj_value, float)

        # Decode solution
        arch = adapter.decode_solution(random_solution, problem)
        assert isinstance(arch, NASCell)

        # Validate solution
        is_valid, issues = adapter.validate_solution(random_solution, problem)
        # May not be valid due to random generation, but should run without error

    def test_zero_cost_proxies(self):
        """Test zero-cost proxy evaluations."""
        cell = NASCell(n_nodes=4)

        # Test individual proxies
        grad_score = grad_norm_proxy(cell)
        assert isinstance(grad_score, float)

        jacob_score = jacob_cov_proxy(cell)
        assert isinstance(jacob_score, float)

        ntk_score = ntk_proxy(cell)
        assert isinstance(ntk_score, float)

        # Test proxy ensemble
        ensemble = ProxyEnsemble()
        ensemble_score = ensemble.compute(cell)
        assert isinstance(ensemble_score, float)

        # Test with macro architecture
        arch = MacroArchitecture()
        arch_score = grad_norm_proxy(arch)
        assert isinstance(arch_score, float)

    def test_evolutionary_nas(self):
        """Test evolutionary NAS algorithm."""
        problem = NASProblem(
            dataset='cifar10',
            search_space='cell',
            max_evaluations=50
        )

        config = {
            'population_size': 10,
            'n_generations': 5,
            'mutation_rate': 0.1
        }

        result = evolutionary_nas(problem, config)

        assert 'best_architecture' in result
        assert 'best_fitness' in result
        assert 'history' in result
        assert isinstance(result['best_architecture'], NASCell)

    def test_differentiable_nas(self):
        """Test differentiable NAS (DARTS)."""
        problem = NASProblem(
            dataset='cifar10',
            search_space='cell'
        )

        config = {
            'n_epochs': 5,
            'lr_arch': 3e-4,
            'lr_weights': 0.025
        }

        result = differentiable_nas(problem, config)

        assert 'architecture' in result
        assert 'metrics' in result
        assert 'history' in result
        assert isinstance(result['architecture'], NASCell)

    def test_bayesian_optimization_nas(self):
        """Test Bayesian optimization NAS."""
        problem = NASProblem(
            dataset='cifar10',
            search_space='cell',
            max_evaluations=30
        )

        config = {
            'n_initial': 5,
            'n_iterations': 20,
            'acquisition': 'ei'
        }

        result = bayesian_optimization_nas(problem, config)

        assert 'best_architecture' in result
        assert 'best_objective' in result
        assert 'convergence' in result
        assert isinstance(result['best_architecture'], NASCell)

    def test_random_search_nas(self):
        """Test random search baseline."""
        problem = NASProblem(
            dataset='cifar10',
            search_space='macro'
        )

        config = {
            'n_samples': 10
        }

        result = random_search_nas(problem, config)

        assert 'best_architecture' in result
        assert 'best_objective' in result
        assert 'statistics' in result
        assert isinstance(result['best_architecture'], MacroArchitecture)

    def test_hardware_aware_nas(self):
        """Test hardware-aware NAS."""
        hw_nas = HardwareAwareNAS(target_device='mobile')

        # Test architecture evaluation
        cell = NASCell(n_nodes=4)
        metrics = hw_nas.evaluate_architecture(cell)

        assert 'latency_ms' in metrics
        assert 'energy_mj' in metrics
        assert 'memory_mb' in metrics
        assert 'throughput' in metrics

        # Test optimization with constraints
        result = hw_nas.optimize(
            dataset='cifar10',
            constraints={'max_latency_ms': 50, 'max_memory_mb': 100},
            search_space='cell',
            max_evaluations=10
        )

        assert 'architecture' in result
        assert 'hardware_metrics' in result
        assert 'satisfies_constraints' in result

    def test_multi_objective_nas(self):
        """Test multi-objective NAS optimization."""
        problem = NASProblem(
            dataset='cifar10',
            search_space='cell',
            objectives=['accuracy', 'params', 'latency_ms'],
            constraints={'max_params': 1e6}
        )

        assert problem.is_multi_objective()

        # Create and evaluate architectures
        archs = []
        for _ in range(10):
            arch = problem.create_architecture()
            metrics = problem.evaluate_architecture(arch, return_all_metrics=True)
            problem.history.append({
                'architecture': arch,
                'metrics': metrics,
                'objective': metrics.get('objective', 0)
            })

        # Get Pareto front
        pareto_front = problem.get_pareto_front()
        assert len(pareto_front) > 0
        assert all('metrics' in p for p in pareto_front)

    def test_architecture_genotype(self):
        """Test architecture genotype representation."""
        cell = NASCell(n_nodes=4)

        # Add some edges
        cell.add_edge(0, 2, Operation(OperationType.CONV_3x3, channels=64))
        cell.add_edge(1, 2, Operation(OperationType.SKIP_CONNECT))
        cell.add_edge(2, 3, Operation(OperationType.SEP_CONV_3x3, channels=128))

        # Get genotype
        genotype = cell.to_genotype()
        assert 'normal' in genotype
        assert isinstance(genotype['normal'], list)

        # Get hash for caching
        arch_hash = cell.get_hash()
        assert isinstance(arch_hash, str)

        # Same architecture should have same hash
        cell2 = NASCell(n_nodes=4)
        cell2.add_edge(0, 2, Operation(OperationType.CONV_3x3, channels=64))
        cell2.add_edge(1, 2, Operation(OperationType.SKIP_CONNECT))
        cell2.add_edge(2, 3, Operation(OperationType.SEP_CONV_3x3, channels=128))

        assert cell.get_hash() == cell2.get_hash()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])