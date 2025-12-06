"""
Comprehensive End-to-End Integration Tests for Librex.
Tests complete workflows and integration between all components.
"""

import pytest
import numpy as np
import tempfile
import json
import time
from pathlib import Path
from unittest.mock import patch, MagicMock
import concurrent.futures

from Librex import (
    EnterpriseOptimizer,
    OptimizationOrchestrator,
    AutoOptimizer,
    HybridOptimizer
)
from Librex.benchmarks.qaplib import QAPLIBLoader, get_instance
from Librex.ai import AIMethodSelector
from Librex.methods.baselines import *
from Librex.methods.advanced import *
from Librex.benchmarking import BenchmarkRunner, BenchmarkSuite
from Librex.quantum import QuantumAdapter, QUBOConverter
from Librex.visualization import create_dashboard
from Librex.utils import generate_random_problem, validate_solution


class TestEnterpriseOptimizer:
    """Test the main EnterpriseOptimizer interface."""

    @pytest.fixture
    def optimizer(self):
        """Create an EnterpriseOptimizer instance."""
        return EnterpriseOptimizer(
            algorithms=['genetic', 'simulated_annealing', 'aco'],
            gpu_accelerated=False,  # For testing
            enterprise_ready=True
        )

    @pytest.fixture
    def qap_problem(self):
        """Load a real QAP instance."""
        return get_instance('chr12a')

    def test_basic_optimization(self, optimizer, qap_problem):
        """Test basic optimization workflow."""
        result = optimizer.optimize(
            qap_problem,
            max_iterations=100,
            verbose=False
        )

        assert 'best_solution' in result
        assert 'best_cost' in result
        assert 'method_used' in result
        assert 'time_elapsed' in result
        assert 'convergence_history' in result

        # Validate solution
        is_valid, _ = validate_solution(result['best_solution'], qap_problem)
        assert is_valid

    def test_multi_algorithm_optimization(self, optimizer, qap_problem):
        """Test optimization with multiple algorithms."""
        result = optimizer.optimize_with_ensemble(
            qap_problem,
            max_iterations=50,
            voting='soft'
        )

        assert 'ensemble_solution' in result
        assert 'individual_results' in result
        assert len(result['individual_results']) == 3  # 3 algorithms

    def test_constraints_handling(self, optimizer):
        """Test optimization with constraints."""
        problem = generate_random_problem('QAP', size=10)

        constraints = {
            'forbidden_assignments': [(0, 5), (1, 3)],  # Position 0 can't go to location 5
            'fixed_assignments': {2: 7},  # Position 2 must go to location 7
            'time_limit': 10.0
        }

        result = optimizer.optimize(
            problem,
            constraints=constraints,
            max_iterations=100
        )

        solution = result['best_solution']
        assert solution[2] == 7  # Fixed assignment respected
        assert solution[0] != 5  # Forbidden assignment avoided

    def test_warm_start_optimization(self, optimizer, qap_problem):
        """Test warm start with initial solution."""
        initial_solution = np.arange(qap_problem['size'])
        np.random.shuffle(initial_solution)

        result = optimizer.optimize(
            qap_problem,
            initial_solution=initial_solution,
            max_iterations=100
        )

        assert result['best_cost'] <= optimizer.evaluate(qap_problem, initial_solution)

    def test_adaptive_algorithm_selection(self, qap_problem):
        """Test adaptive algorithm selection based on problem features."""
        optimizer = EnterpriseOptimizer(
            algorithms='auto',  # Automatic selection
            use_ai_selector=True
        )

        result = optimizer.optimize(qap_problem, max_iterations=100)

        assert result['method_used'] in ['genetic', 'simulated_annealing', 'aco', 'pso', 'vns']
        assert 'selection_reasoning' in result

    def test_parallel_optimization(self, optimizer, qap_problem):
        """Test parallel optimization of multiple problems."""
        problems = [qap_problem] * 3

        results = optimizer.optimize_batch(
            problems,
            max_iterations=50,
            n_jobs=2
        )

        assert len(results) == 3
        for result in results:
            assert 'best_solution' in result

    def test_hybrid_optimization(self, qap_problem):
        """Test hybrid classical-quantum optimization."""
        optimizer = EnterpriseOptimizer(
            algorithms=['genetic', 'qaoa'],
            quantum_backend='simulator'
        )

        with patch('Librex.quantum.QAOAOptimizer.optimize') as mock_qaoa:
            mock_qaoa.return_value = {
                'solution': np.arange(qap_problem['size']),
                'energy': 1000
            }

            result = optimizer.optimize(
                qap_problem,
                max_iterations=50,
                use_hybrid=True
            )

            assert 'classical_result' in result
            assert 'quantum_result' in result


class TestOptimizationOrchestrator:
    """Test the OptimizationOrchestrator for complex workflows."""

    @pytest.fixture
    def orchestrator(self):
        """Create an OptimizationOrchestrator."""
        return OptimizationOrchestrator(
            stages=['preprocessing', 'optimization', 'postprocessing']
        )

    def test_pipeline_execution(self, orchestrator):
        """Test complete optimization pipeline."""
        problem = generate_random_problem('QAP', size=15)

        pipeline = [
            ('reduce', {'target_size': 10}),
            ('optimize', {'method': 'genetic', 'max_iterations': 100}),
            ('refine', {'method': 'local_search', 'iterations': 20})
        ]

        result = orchestrator.execute_pipeline(problem, pipeline)

        assert 'final_solution' in result
        assert 'stage_results' in result
        assert len(result['stage_results']) == 3

    def test_adaptive_pipeline(self, orchestrator):
        """Test adaptive pipeline that changes based on results."""
        problem = generate_random_problem('QAP', size=20)

        def quality_checker(result):
            return result['cost'] < 1000

        pipeline = orchestrator.create_adaptive_pipeline(
            initial_stage='genetic',
            refinement_stage='vns',
            quality_checker=quality_checker,
            max_attempts=3
        )

        result = orchestrator.execute_pipeline(problem, pipeline)

        assert result['attempts'] <= 3
        assert 'final_solution' in result

    def test_hierarchical_optimization(self, orchestrator):
        """Test hierarchical optimization approach."""
        large_problem = generate_random_problem('QAP', size=50)

        result = orchestrator.hierarchical_optimize(
            large_problem,
            levels=[50, 25, 10],  # Problem sizes at each level
            methods=['genetic', 'aco', 'vns']
        )

        assert 'level_results' in result
        assert len(result['level_results']) == 3
        assert result['final_solution'].shape[0] == 50


class TestAutoOptimizer:
    """Test automatic optimization with minimal configuration."""

    @pytest.fixture
    def auto_optimizer(self):
        """Create an AutoOptimizer."""
        return AutoOptimizer()

    def test_automatic_optimization(self, auto_optimizer):
        """Test fully automatic optimization."""
        problem = generate_random_problem('QAP', size=12)

        result = auto_optimizer.optimize(problem)

        assert 'best_solution' in result
        assert 'best_cost' in result
        assert 'method_selected' in result
        assert 'configuration_used' in result

    def test_problem_type_detection(self, auto_optimizer):
        """Test automatic problem type detection."""
        # QAP problem
        qap = {'flow': np.random.rand(10, 10), 'distance': np.random.rand(10, 10)}
        detected = auto_optimizer.detect_problem_type(qap)
        assert detected == 'QAP'

        # TSP problem
        tsp = {'distance': np.random.rand(10, 10), 'cities': 10}
        detected = auto_optimizer.detect_problem_type(tsp)
        assert detected == 'TSP'

    def test_budget_aware_optimization(self, auto_optimizer):
        """Test optimization with time/iteration budget."""
        problem = generate_random_problem('QAP', size=15)

        # Time budget
        result = auto_optimizer.optimize(
            problem,
            time_budget=2.0  # 2 seconds
        )

        assert result['time_elapsed'] <= 2.5  # Some overhead allowed

        # Iteration budget
        result = auto_optimizer.optimize(
            problem,
            iteration_budget=100
        )

        assert result['iterations_used'] <= 100


class TestQAPLIBIntegration:
    """Test integration with QAPLIB benchmark suite."""

    def test_benchmark_all_methods(self):
        """Test all methods on QAPLIB instances."""
        instances = ['chr12a', 'nug12', 'esc16a']
        methods = [
            RandomSearchOptimizer(max_iterations=50),
            SimulatedAnnealingOptimizer(max_iterations=50),
            GeneticAlgorithmOptimizer(max_iterations=50),
            AntColonyOptimizer(n_iterations=20),
            ParticleSwarmOptimizer(n_iterations=20, discrete=True)
        ]

        runner = BenchmarkRunner(methods)
        results = []

        for instance_name in instances:
            instance = get_instance(instance_name)
            result = runner.run_problem(instance)
            results.append(result)

        # Verify all methods ran
        for result in results:
            assert len(result['methods']) == len(methods)

    def test_performance_tracking(self):
        """Test performance tracking across instances."""
        loader = QAPLIBLoader()
        instances = ['chr12a', 'chr15a', 'chr20a']

        optimizer = EnterpriseOptimizer(algorithms=['genetic'])
        performance = {}

        for name in instances:
            instance = loader.load(name)
            result = optimizer.optimize(instance, max_iterations=100)

            if 'optimal' in instance:
                gap = (result['best_cost'] - instance['optimal']) / instance['optimal'] * 100
                performance[name] = gap

        # Check that gaps are reasonable (< 50% for small instances)
        for name, gap in performance.items():
            if name == 'chr12a':
                assert gap < 50

    def test_scalability_analysis(self):
        """Test scalability across different problem sizes."""
        sizes = [12, 15, 20, 25]
        instance_names = [f'chr{size}a' for size in sizes]

        optimizer = EnterpriseOptimizer(algorithms=['simulated_annealing'])
        times = []

        for name in instance_names:
            try:
                instance = get_instance(name)
                start = time.time()
                optimizer.optimize(instance, max_iterations=50)
                elapsed = time.time() - start
                times.append(elapsed)
            except:
                # Instance might not exist, skip
                continue

        # Time should increase with size
        if len(times) > 1:
            assert times[-1] > times[0]


class TestAIMethodSelectorIntegration:
    """Test AI method selector integration."""

    @pytest.fixture
    def selector(self):
        """Create an AI method selector."""
        return AIMethodSelector()

    def test_recommendation_workflow(self, selector):
        """Test complete recommendation workflow."""
        problems = [
            generate_random_problem('QAP', size=10, density=0.3),  # Sparse
            generate_random_problem('QAP', size=10, density=0.9),  # Dense
            generate_random_problem('QAP', size=50),  # Large
        ]

        recommendations = []
        for problem in problems:
            rec = selector.recommend(problem, top_k=3)
            recommendations.append(rec)

        # Different problems should get different recommendations
        assert len(set(r[0].method for r in recommendations)) > 1

    def test_learning_from_feedback(self, selector):
        """Test selector learning from optimization results."""
        problem = generate_random_problem('QAP', size=15)

        # Get initial recommendation
        initial_rec = selector.recommend(problem, top_k=1)[0]

        # Simulate optimization with feedback
        feedback = {
            'method': initial_rec.method,
            'actual_quality': 0.95,
            'actual_time': 2.5,
            'success': True
        }

        selector.update_from_feedback(problem, feedback)

        # Get new recommendation
        new_rec = selector.recommend(problem, top_k=1)[0]

        # Confidence should have changed
        assert new_rec.confidence != initial_rec.confidence


class TestQuantumIntegration:
    """Test quantum computing integration."""

    @pytest.mark.skipif(True, reason="Quantum tests require special setup")
    def test_qubo_conversion_workflow(self):
        """Test complete QUBO conversion workflow."""
        qap = generate_random_problem('QAP', size=3)

        converter = QUBOConverter()
        Q = converter.convert_qap_to_qubo(qap)

        assert Q.shape == (9, 9)  # 3x3 QAP needs 9 qubits

    @pytest.mark.skipif(True, reason="Quantum tests require special setup")
    def test_quantum_classical_hybrid(self):
        """Test quantum-classical hybrid optimization."""
        problem = generate_random_problem('QAP', size=4)

        hybrid = HybridOptimizer(
            classical_method='genetic',
            quantum_method='qaoa',
            backend='simulator'
        )

        with patch('Librex.quantum.QAOAOptimizer.optimize'):
            result = hybrid.optimize(problem, max_iterations=50)

            assert 'classical_phase' in result
            assert 'quantum_phase' in result
            assert 'hybrid_solution' in result


class TestVisualizationIntegration:
    """Test visualization integration with optimization results."""

    def test_optimization_dashboard(self):
        """Test creating dashboard from optimization results."""
        problem = generate_random_problem('QAP', size=10)
        optimizer = EnterpriseOptimizer(algorithms=['genetic', 'simulated_annealing'])

        result = optimizer.optimize_with_tracking(
            problem,
            max_iterations=100,
            track_diversity=True,
            track_memory=True
        )

        with patch('matplotlib.pyplot.show'):
            dashboard = create_dashboard(result, title="Optimization Results")
            assert dashboard is not None

    def test_benchmark_visualization(self):
        """Test visualizing benchmark results."""
        suite = BenchmarkSuite(name="TestSuite")
        suite.add_problem(generate_random_problem('QAP', size=10))
        suite.add_method(RandomSearchOptimizer)
        suite.add_method(SimulatedAnnealingOptimizer)

        suite.configure({'max_iterations': 50})
        results = suite.run()

        with patch('matplotlib.pyplot.show'):
            report = suite.generate_visual_report(results)
            assert report is not None


class TestErrorHandlingIntegration:
    """Test error handling across the system."""

    def test_invalid_problem_handling(self):
        """Test handling of invalid problems."""
        optimizer = EnterpriseOptimizer(algorithms=['genetic'])

        # Invalid problem structure
        invalid_problem = {'invalid': 'data'}

        with pytest.raises(ValueError):
            optimizer.optimize(invalid_problem)

    def test_method_failure_recovery(self):
        """Test recovery from method failures."""
        class FailingOptimizer:
            def optimize(self, problem):
                raise RuntimeError("Intentional failure")

        optimizer = EnterpriseOptimizer(
            algorithms=['genetic'],
            fallback_method='random_search'
        )
        optimizer.methods['failing'] = FailingOptimizer()

        problem = generate_random_problem('QAP', size=10)
        result = optimizer.optimize(problem, method='failing')

        # Should fallback to random_search
        assert result['method_used'] == 'random_search'
        assert 'fallback_reason' in result

    def test_timeout_handling(self):
        """Test handling of optimization timeouts."""
        optimizer = EnterpriseOptimizer(
            algorithms=['genetic'],
            timeout=0.1  # 100ms timeout
        )

        large_problem = generate_random_problem('QAP', size=100)

        result = optimizer.optimize(large_problem, max_iterations=10000)

        assert result['status'] == 'timeout'
        assert result['best_solution'] is not None  # Should return best so far


class TestPerformanceIntegration:
    """Test system performance under various conditions."""

    def test_concurrent_optimization(self):
        """Test concurrent optimization of multiple problems."""
        problems = [generate_random_problem('QAP', size=10) for _ in range(5)]
        optimizer = EnterpriseOptimizer(algorithms=['genetic'])

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(optimizer.optimize, p, max_iterations=50)
                for p in problems
            ]
            results = [f.result() for f in futures]

        assert len(results) == 5
        assert all('best_solution' in r for r in results)

    def test_memory_efficiency(self):
        """Test memory efficiency with large problems."""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Run optimization on large problem
        large_problem = generate_random_problem('QAP', size=100)
        optimizer = EnterpriseOptimizer(
            algorithms=['genetic'],
            memory_efficient=True
        )
        optimizer.optimize(large_problem, max_iterations=10)

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # Should not use excessive memory (< 500MB increase)
        assert memory_increase < 500

    def test_cache_effectiveness(self):
        """Test caching effectiveness."""
        problem = generate_random_problem('QAP', size=10)
        optimizer = EnterpriseOptimizer(
            algorithms=['genetic'],
            enable_caching=True
        )

        # First run
        start1 = time.time()
        result1 = optimizer.optimize(problem, max_iterations=100)
        time1 = time.time() - start1

        # Second run (should use cache)
        start2 = time.time()
        result2 = optimizer.optimize(problem, max_iterations=100)
        time2 = time.time() - start2

        # Cached run should be much faster
        assert time2 < time1 * 0.1  # At least 10x faster


class TestEndToEndScenarios:
    """Test complete end-to-end optimization scenarios."""

    def test_production_workflow(self):
        """Test a complete production optimization workflow."""
        # 1. Load problem
        problem = get_instance('chr15a')

        # 2. Select method using AI
        selector = AIMethodSelector()
        recommendations = selector.recommend(problem, top_k=3)

        # 3. Run optimization with recommended methods
        optimizer = EnterpriseOptimizer(
            algorithms=[r.method for r in recommendations[:2]]
        )
        result = optimizer.optimize_with_ensemble(problem, max_iterations=200)

        # 4. Validate solution
        is_valid, _ = validate_solution(result['ensemble_solution'], problem)
        assert is_valid

        # 5. Generate report
        report = {
            'problem': 'chr15a',
            'methods_used': [r.method for r in recommendations[:2]],
            'best_cost': result['best_cost'],
            'time_elapsed': result['time_elapsed'],
            'improvement': result.get('improvement', 0)
        }

        # 6. Save results
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(report, f)
            temp_file = f.name

        try:
            # Verify saved results
            with open(temp_file) as f:
                loaded = json.load(f)
            assert loaded['problem'] == 'chr15a'
        finally:
            Path(temp_file).unlink()

    def test_research_workflow(self):
        """Test a research-oriented workflow."""
        # 1. Create benchmark suite
        suite = BenchmarkSuite(name="Research")

        # 2. Add multiple problem instances
        for size in [10, 15, 20]:
            suite.add_problem(generate_random_problem('QAP', size=size))

        # 3. Add methods to compare
        suite.add_method(GeneticAlgorithmOptimizer)
        suite.add_method(AntColonyOptimizer)
        suite.add_method(ParticleSwarmOptimizer)

        # 4. Configure experiments
        suite.configure({
            'max_iterations': 100,
            'n_runs': 3,
            'parallel': True
        })

        # 5. Run experiments
        results = suite.run()

        # 6. Statistical analysis
        stats = suite.analyze_statistics()

        assert 'method_comparison' in stats
        assert 'significance_tests' in stats

        # 7. Generate publication-ready plots
        with patch('matplotlib.pyplot.show'):
            plots = suite.generate_plots()
            assert len(plots) > 0