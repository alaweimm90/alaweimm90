"""
Comprehensive tests for MEZAN Integration Layer

Tests the OptimizerInterface, OptimizerFactory, and feature flag system.
"""

import pytest
import numpy as np
from MEZAN.core import (
    OptimizerInterface,
    OptimizerFactory,
    HeuristicFallbackOptimizer,
    OptimizationProblem,
    OptimizationResult,
    ProblemType,
    SolverStatus,
    FeatureFlag,
    get_optimizer_factory,
    reset_optimizer_factory,
)


class TestOptimizerInterface:
    """Test the abstract OptimizerInterface"""

    def test_heuristic_fallback_initialization(self):
        """Test heuristic fallback solver initializes"""
        solver = HeuristicFallbackOptimizer()
        assert not solver._is_initialized

        solver.initialize()
        assert solver._is_initialized

    def test_heuristic_supports_all_types(self):
        """Test heuristic solver supports all problem types"""
        solver = HeuristicFallbackOptimizer()
        types = solver.get_problem_types()

        # Should support all problem types
        assert ProblemType.QAP in types
        assert ProblemType.FLOW in types
        assert ProblemType.ALLOC in types
        assert ProblemType.GRAPH in types
        assert ProblemType.DUAL in types
        assert ProblemType.EVO in types
        assert ProblemType.META in types

    def test_heuristic_qap_solver(self):
        """Test heuristic QAP solver produces valid solution"""
        solver = HeuristicFallbackOptimizer()
        solver.initialize()

        problem = OptimizationProblem(
            problem_type=ProblemType.QAP,
            data={
                "distance_matrix": [[0, 1, 2], [1, 0, 1], [2, 1, 0]],
                "flow_matrix": [[0, 5, 3], [5, 0, 2], [3, 2, 0]],
            },
        )

        result = solver.solve(problem)

        assert result.status == SolverStatus.FALLBACK
        assert result.solution is not None
        assert "assignment" in result.solution
        assert len(result.solution["assignment"]) == 3

    def test_complexity_estimation(self):
        """Test heuristic always reports low complexity"""
        solver = HeuristicFallbackOptimizer()

        small_problem = OptimizationProblem(
            problem_type=ProblemType.QAP,
            data={"distance_matrix": [[0]], "flow_matrix": [[0]]},
        )

        assert solver.estimate_complexity(small_problem) == "low"


class TestOptimizationProblem:
    """Test OptimizationProblem data class"""

    def test_qap_validation(self):
        """Test QAP problem validation"""
        # Valid problem
        valid_problem = OptimizationProblem(
            problem_type=ProblemType.QAP,
            data={
                "distance_matrix": [[0, 1], [1, 0]],
                "flow_matrix": [[0, 2], [2, 0]],
            },
        )

        is_valid, error = valid_problem.validate()
        assert is_valid
        assert error is None

        # Invalid: mismatched sizes
        invalid_problem = OptimizationProblem(
            problem_type=ProblemType.QAP,
            data={
                "distance_matrix": [[0, 1], [1, 0]],
                "flow_matrix": [[0, 1, 2], [1, 0, 1], [2, 1, 0]],
            },
        )

        is_valid, error = invalid_problem.validate()
        assert not is_valid
        assert "same dimension" in error

    def test_flow_validation(self):
        """Test FLOW problem validation"""
        valid_problem = OptimizationProblem(
            problem_type=ProblemType.FLOW,
            data={
                "workflow_graph": {"nodes": ["a", "b"], "edges": [("a", "b")]},
                "confidence_scores": {"a": 0.9, "b": 0.8},
            },
        )

        is_valid, error = valid_problem.validate()
        assert is_valid

    def test_alloc_validation(self):
        """Test ALLOC problem validation"""
        valid_problem = OptimizationProblem(
            problem_type=ProblemType.ALLOC,
            data={
                "resource_demands": [("agent1", 10.0)],
                "budget_constraint": 100.0,
            },
        )

        is_valid, error = valid_problem.validate()
        assert is_valid


class TestOptimizerFactory:
    """Test OptimizerFactory and feature flags"""

    def setup_method(self):
        """Reset factory before each test"""
        reset_optimizer_factory()

    def test_factory_initialization(self):
        """Test factory initializes with config"""
        config = {
            "feature_flags": {
                "enable_qap_libria": False,
                "force_heuristic": True,
            }
        }

        factory = OptimizerFactory(config=config)
        assert factory is not None
        assert factory.feature_flags["force_heuristic"] == True

    def test_fallback_when_all_disabled(self):
        """Test factory returns heuristic when all solvers disabled"""
        config = {
            "feature_flags": {
                "enable_qap_libria": False,
                "enable_flow_libria": False,
                "enable_all_libria": False,
            }
        }

        factory = OptimizerFactory(config=config)

        problem = OptimizationProblem(
            problem_type=ProblemType.QAP,
            data={
                "distance_matrix": [[0, 1], [1, 0]],
                "flow_matrix": [[0, 2], [2, 0]],
            },
        )

        optimizer = factory.create_optimizer(problem)
        assert isinstance(optimizer, HeuristicFallbackOptimizer)

    def test_force_heuristic_flag(self):
        """Test force_heuristic overrides all other flags"""
        config = {
            "feature_flags": {
                "enable_all_libria": True,  # Try to enable everything
                "force_heuristic": True,    # But force heuristic
            }
        }

        factory = OptimizerFactory(config=config)

        problem = OptimizationProblem(
            problem_type=ProblemType.QAP,
            data={
                "distance_matrix": [[0, 1], [1, 0]],
                "flow_matrix": [[0, 2], [2, 0]],
            },
        )

        optimizer = factory.create_optimizer(problem)
        assert isinstance(optimizer, HeuristicFallbackOptimizer)

    def test_gpu_flag(self):
        """Test GPU enablement flag"""
        config = {
            "feature_flags": {
                "enable_gpu": True,
            }
        }

        factory = OptimizerFactory(config=config)

        problem = OptimizationProblem(
            problem_type=ProblemType.QAP,
            data={"distance_matrix": [[0]], "flow_matrix": [[0]]},
        )

        optimizer = factory.create_optimizer(problem, enable_gpu=None)
        # Should respect config default
        assert optimizer.enable_gpu == True

    def test_timeout_configuration(self):
        """Test timeout configuration"""
        config = {"default_timeout": 123.45}

        factory = OptimizerFactory(config=config)

        problem = OptimizationProblem(
            problem_type=ProblemType.QAP,
            data={"distance_matrix": [[0]], "flow_matrix": [[0]]},
        )

        optimizer = factory.create_optimizer(problem)
        assert optimizer.timeout == 123.45

    def test_custom_solver_registration(self):
        """Test registering custom solver"""

        class CustomSolver(OptimizerInterface):
            def initialize(self):
                self._is_initialized = True

            def solve(self, problem):
                return OptimizationResult(
                    status=SolverStatus.SUCCESS,
                    solution={"custom": True},
                    objective_value=42.0,
                    metadata={},
                    computation_time=0.001,
                )

            def get_problem_types(self):
                return [ProblemType.QAP]

            def estimate_complexity(self, problem):
                return "low"

        factory = OptimizerFactory()
        factory.register_solver(ProblemType.QAP, CustomSolver, {"custom_config": True})

        problem = OptimizationProblem(
            problem_type=ProblemType.QAP,
            data={"distance_matrix": [[0]], "flow_matrix": [[0]]},
        )

        optimizer = factory.create_optimizer(problem)
        assert isinstance(optimizer, CustomSolver)

    def test_get_registered_solvers(self):
        """Test getting list of registered solvers"""
        factory = OptimizerFactory()
        solvers = factory.get_registered_solvers()

        assert isinstance(solvers, dict)
        # Should have heuristic fallback registered for all types
        assert len(solvers) > 0

    def test_update_feature_flags(self):
        """Test dynamically updating feature flags"""
        factory = OptimizerFactory(
            config={"feature_flags": {"enable_qap_libria": False}}
        )

        assert factory.feature_flags.get("enable_qap_libria") == False

        factory.update_feature_flags({"enable_qap_libria": True})

        assert factory.feature_flags.get("enable_qap_libria") == True

    def test_singleton_factory(self):
        """Test global factory singleton"""
        factory1 = get_optimizer_factory()
        factory2 = get_optimizer_factory()

        assert factory1 is factory2  # Same instance

        reset_optimizer_factory()

        factory3 = get_optimizer_factory()
        assert factory3 is not factory1  # New instance after reset


class TestOptimizationResult:
    """Test OptimizationResult data class"""

    def test_result_validity(self):
        """Test result validity check"""
        # Valid result
        valid_result = OptimizationResult(
            status=SolverStatus.SUCCESS,
            solution={"assignment": [0, 1, 2]},
            objective_value=123.45,
            metadata={},
            computation_time=0.5,
        )

        assert valid_result.is_valid()

        # Invalid: no solution
        invalid_result = OptimizationResult(
            status=SolverStatus.FAILED,
            solution=None,
            objective_value=None,
            metadata={"error": "Something went wrong"},
            computation_time=0.1,
        )

        assert not invalid_result.is_valid()

    def test_result_serialization(self):
        """Test result to_dict()"""
        result = OptimizationResult(
            status=SolverStatus.SUCCESS,
            solution={"test": 123},
            objective_value=456.0,
            metadata={"solver": "test"},
            computation_time=0.789,
            iterations=100,
            improvement_over_baseline=25.5,
        )

        result_dict = result.to_dict()

        assert result_dict["status"] == "success"
        assert result_dict["solution"] == {"test": 123}
        assert result_dict["objective_value"] == 456.0
        assert result_dict["computation_time"] == 0.789
        assert result_dict["iterations"] == 100
        assert result_dict["improvement_over_baseline"] == 25.5


class TestEndToEndIntegration:
    """End-to-end integration tests"""

    def setup_method(self):
        reset_optimizer_factory()

    def test_qap_problem_end_to_end(self):
        """Test full QAP problem solving flow"""
        # Create factory with heuristic fallback
        factory = OptimizerFactory(
            config={"feature_flags": {"force_heuristic": True}}
        )

        # Create problem
        problem = OptimizationProblem(
            problem_type=ProblemType.QAP,
            data={
                "distance_matrix": [
                    [0, 10, 20],
                    [10, 0, 15],
                    [20, 15, 0],
                ],
                "flow_matrix": [
                    [0, 5, 3],
                    [5, 0, 2],
                    [3, 2, 0],
                ],
            },
        )

        # Validate problem
        is_valid, error = problem.validate()
        assert is_valid

        # Create optimizer
        optimizer = factory.create_optimizer(problem)
        assert optimizer is not None

        # Solve
        optimizer.initialize()
        result = optimizer.solve(problem)

        # Check result
        assert result.status == SolverStatus.FALLBACK
        assert result.is_valid()
        assert len(result.solution["assignment"]) == 3
        assert result.computation_time > 0

    def test_multiple_problem_types(self):
        """Test factory handles multiple problem types"""
        factory = OptimizerFactory()

        # QAP problem
        qap_problem = OptimizationProblem(
            problem_type=ProblemType.QAP,
            data={"distance_matrix": [[0]], "flow_matrix": [[0]]},
        )

        qap_optimizer = factory.create_optimizer(qap_problem)
        assert qap_optimizer is not None

        # FLOW problem
        flow_problem = OptimizationProblem(
            problem_type=ProblemType.FLOW,
            data={
                "workflow_graph": {"nodes": ["a"], "edges": []},
                "confidence_scores": {"a": 1.0},
            },
        )

        flow_optimizer = factory.create_optimizer(flow_problem)
        assert flow_optimizer is not None

        # ALLOC problem
        alloc_problem = OptimizationProblem(
            problem_type=ProblemType.ALLOC,
            data={
                "resource_demands": [("agent1", 10.0)],
                "budget_constraint": 100.0,
            },
        )

        alloc_optimizer = factory.create_optimizer(alloc_problem)
        assert alloc_optimizer is not None

    def test_error_handling(self):
        """Test error handling for invalid problems"""
        factory = OptimizerFactory()

        # Missing required data
        invalid_problem = OptimizationProblem(
            problem_type=ProblemType.QAP,
            data={},  # Missing distance and flow matrices
        )

        optimizer = factory.create_optimizer(invalid_problem)
        result = optimizer.solve(invalid_problem)

        assert result.status == SolverStatus.FAILED
        assert "error" in result.metadata
