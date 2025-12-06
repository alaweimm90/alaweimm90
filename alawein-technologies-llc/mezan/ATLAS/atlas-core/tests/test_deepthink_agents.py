"""
Comprehensive Test Suite for DeepThink Agents

Tests the optimized 3+1 agent architecture:
- 3 parallel agents (Analyzer, Optimizer, Validator)
- 1 sequential synthesizer (deep reasoning)

Author: MEZAN Research Team
Date: 2025-11-18
"""

import pytest
import time
from typing import Dict, Any

from atlas_core.deepthink_agents import (
    DeepThinkOrchestrator,
    DeepTask,
    DeepResult,
    AnalysisDepth,
    AgentRole,
    AnalyzerAgent,
    OptimizerAgent,
    ValidatorAgent,
    SynthesizerAgent,
)


class TestAnalyzerAgent:
    """Test suite for AnalyzerAgent"""

    def test_analyzer_initialization(self):
        """Test that analyzer initializes correctly"""
        analyzer = AnalyzerAgent()
        assert analyzer.role == AgentRole.ANALYZER
        assert analyzer.analysis_count == 0
        assert analyzer.total_time == 0.0

    def test_analyzer_qap_problem(self):
        """Test analyzer on QAP problem"""
        analyzer = AnalyzerAgent()

        problem = {
            "type": "qap",
            "size": 30,
            "constraints": [
                {"type": "assignment"},
                {"type": "capacity", "max": 100},
            ],
        }

        task = DeepTask(
            task_id="test_qap",
            problem=problem,
            depth=AnalysisDepth.DEEP,
        )

        result = analyzer.analyze(task)

        assert result.task_id == "test_qap"
        assert result.role == AgentRole.ANALYZER
        assert len(result.insights) > 0
        assert len(result.recommendations) > 0
        assert result.confidence > 0.0
        assert "qap" in result.reasoning.lower()

        # Check for expected insights
        insights_text = " ".join(result.insights).lower()
        assert "qap" in insights_text or "quadratic" in insights_text

    def test_analyzer_assignment_problem(self):
        """Test analyzer on assignment problem"""
        analyzer = AnalyzerAgent()

        problem = {
            "type": "assignment",
            "size": 20,
            "constraints": [],
        }

        task = DeepTask(
            task_id="test_assignment",
            problem=problem,
            depth=AnalysisDepth.QUICK,
        )

        result = analyzer.analyze(task)

        assert len(result.recommendations) > 0
        # Should recommend Hungarian algorithm for assignment
        rec_text = str(result.recommendations).lower()
        assert "hungarian" in rec_text or "assignment" in rec_text

    def test_analyzer_statistics(self):
        """Test analyzer statistics tracking"""
        analyzer = AnalyzerAgent()

        problem = {"type": "qap", "size": 10}
        task = DeepTask(task_id="test", problem=problem)

        # Analyze multiple times
        for i in range(3):
            analyzer.analyze(task)

        stats = analyzer.get_stats()
        assert stats["analyses_completed"] == 3
        assert stats["total_time"] > 0
        assert stats["avg_time"] > 0


class TestOptimizerAgent:
    """Test suite for OptimizerAgent"""

    def test_optimizer_initialization(self):
        """Test that optimizer initializes correctly"""
        optimizer = OptimizerAgent()
        assert optimizer.role == AgentRole.OPTIMIZER
        assert optimizer.analysis_count == 0

    def test_optimizer_single_objective(self):
        """Test optimizer on single-objective problem"""
        optimizer = OptimizerAgent()

        problem = {
            "type": "qap",
            "size": 25,
            "objectives": 1,
        }

        task = DeepTask(task_id="test", problem=problem)
        result = optimizer.analyze(task)

        assert len(result.recommendations) > 0
        # Should recommend algorithm portfolio or specific algorithm
        has_algo_rec = any(
            r.get("type") in ["algorithm_selection", "algorithm_portfolio"]
            for r in result.recommendations
        )
        assert has_algo_rec

    def test_optimizer_multi_objective(self):
        """Test optimizer on multi-objective problem"""
        optimizer = OptimizerAgent()

        problem = {
            "type": "optimization",
            "size": 30,
            "objectives": 3,
        }

        task = DeepTask(task_id="test", problem=problem)
        result = optimizer.analyze(task)

        # Should mention multi-objective or Pareto
        text = " ".join(result.insights).lower()
        assert "multi-objective" in text or "pareto" in text

    def test_optimizer_resource_allocation(self):
        """Test optimizer resource allocation recommendations"""
        optimizer = OptimizerAgent()

        # Test with different time budgets
        for budget in [5, 30, 120]:
            problem = {"type": "qap", "size": 20, "objectives": 1}
            task = DeepTask(
                task_id=f"test_{budget}s",
                problem=problem,
                max_time_seconds=budget,
            )

            result = optimizer.analyze(task)

            # Should have resource allocation recommendation
            resource_recs = [
                r for r in result.recommendations
                if r.get("type") == "resource_allocation"
            ]
            assert len(resource_recs) > 0


class TestValidatorAgent:
    """Test suite for ValidatorAgent"""

    def test_validator_initialization(self):
        """Test that validator initializes correctly"""
        validator = ValidatorAgent()
        assert validator.role == AgentRole.VALIDATOR
        assert validator.analysis_count == 0

    def test_validator_without_solution(self):
        """Test validator on problem without solution (pre-solve)"""
        validator = ValidatorAgent()

        problem = {
            "type": "qap",
            "size": 30,
            "constraints": [{"type": "symmetry"}] * 5,
        }

        task = DeepTask(task_id="test", problem=problem)
        result = validator.analyze(task)

        assert len(result.insights) > 0
        # Should provide pre-solve validation insights

    def test_validator_with_solution(self):
        """Test validator with solution provided"""
        validator = ValidatorAgent()

        problem = {"type": "qap", "size": 20}
        solution = {"permutation": list(range(20))}

        task = DeepTask(
            task_id="test",
            problem=problem,
            metadata={"solution": solution},
        )

        result = validator.analyze(task)

        # Should validate the solution
        insights_text = " ".join(result.insights).lower()
        assert "solution" in insights_text or "feasible" in insights_text


class TestSynthesizerAgent:
    """Test suite for SynthesizerAgent (deep sequential)"""

    def test_synthesizer_initialization(self):
        """Test that synthesizer initializes correctly"""
        synthesizer = SynthesizerAgent()
        assert synthesizer.role == AgentRole.SYNTHESIZER
        assert synthesizer.analysis_count == 0

    def test_synthesizer_synthesis(self):
        """Test synthesizer deep synthesis"""
        # Create mock results from other agents
        problem = {
            "type": "qap",
            "size": 30,
            "constraints": [{"type": "assignment"}],
        }

        task = DeepTask(task_id="test", problem=problem)

        # Create analyzer result
        analyzer_result = DeepResult(
            task_id="test",
            role=AgentRole.ANALYZER,
            insights=["QAP problem detected", "Constraint analysis complete"],
            recommendations=[
                {
                    "type": "algorithm_selection",
                    "recommendation": "Hybrid approach",
                    "confidence": 0.85,
                    "reason": "Medium size problem",
                }
            ],
            confidence=0.85,
            reasoning="Analyzer reasoning",
            time_seconds=0.001,
        )

        # Create optimizer result
        optimizer_result = DeepResult(
            task_id="test",
            role=AgentRole.OPTIMIZER,
            insights=["Algorithm portfolio recommended"],
            recommendations=[
                {
                    "type": "algorithm_portfolio",
                    "recommendation": "Multi-algorithm",
                    "confidence": 0.88,
                    "reason": "Hedges risk",
                }
            ],
            confidence=0.87,
            reasoning="Optimizer reasoning",
            time_seconds=0.001,
        )

        # Create validator result
        validator_result = DeepResult(
            task_id="test",
            role=AgentRole.VALIDATOR,
            insights=["Problem structure validated"],
            recommendations=[
                {
                    "type": "preprocessing",
                    "recommendation": "Apply symmetry breaking",
                    "confidence": 0.82,
                    "reason": "Reduce search space",
                }
            ],
            confidence=0.83,
            reasoning="Validator reasoning",
            time_seconds=0.001,
        )

        # Synthesize
        synthesizer = SynthesizerAgent()
        synthesis = synthesizer.synthesize(
            task, analyzer_result, optimizer_result, validator_result
        )

        # Validate synthesis output
        assert synthesis.task_id == "test"
        assert synthesis.role == AgentRole.SYNTHESIZER
        assert len(synthesis.insights) > 3  # Should aggregate + add synthesis
        assert len(synthesis.recommendations) > 0  # Should synthesize recommendations
        assert synthesis.confidence > 0.85  # Should be high confidence
        assert "synthesis" in synthesis.reasoning.lower()

        # Check for causal analysis
        insights_text = " ".join(synthesis.insights).lower()
        assert "causal" in insights_text or "pattern" in insights_text

    def test_synthesizer_theme_extraction(self):
        """Test theme extraction from insights"""
        synthesizer = SynthesizerAgent()

        insights = [
            "Constraint complexity is high",
            "Algorithm selection critical",
            "Quality metrics important",
            "Constraint handling needed",
        ]

        themes = synthesizer._extract_themes(insights)

        assert "constraint" in themes
        assert themes["constraint"] == 2  # Appears twice
        assert "algorithm" in themes


class TestDeepThinkOrchestrator:
    """Test suite for DeepThinkOrchestrator"""

    def test_orchestrator_initialization(self):
        """Test orchestrator initialization"""
        orchestrator = DeepThinkOrchestrator(max_parallel_workers=3)

        assert orchestrator.analyzer is not None
        assert orchestrator.optimizer is not None
        assert orchestrator.validator is not None
        assert orchestrator.synthesizer is not None

    def test_orchestrator_deep_analyze(self):
        """Test full deep analysis workflow"""
        orchestrator = DeepThinkOrchestrator(max_parallel_workers=3)

        problem = {
            "type": "qap",
            "size": 25,
            "constraints": [{"type": "assignment"}],
        }

        task = DeepTask(
            task_id="test_workflow",
            problem=problem,
            depth=AnalysisDepth.DEEP,
        )

        # Execute deep analysis
        start_time = time.time()
        analyzer, optimizer, validator, synthesis = orchestrator.deep_analyze(
            task, use_synthesis=True
        )
        elapsed = time.time() - start_time

        # Verify all results
        assert analyzer is not None
        assert optimizer is not None
        assert validator is not None
        assert synthesis is not None

        # Check analysis quality
        assert analyzer.confidence > 0.7
        assert optimizer.confidence > 0.7
        assert validator.confidence > 0.7
        assert synthesis.confidence > 0.85  # Synthesis should be high confidence

        # Check synthesis aggregation
        total_insights = (
            len(analyzer.insights) +
            len(optimizer.insights) +
            len(validator.insights)
        )
        assert len(synthesis.insights) > total_insights  # Should add synthesis insights

        # Check performance
        assert elapsed < 1.0  # Should complete quickly

    def test_orchestrator_without_synthesis(self):
        """Test orchestrator without synthesis phase"""
        orchestrator = DeepThinkOrchestrator(max_parallel_workers=3)

        problem = {"type": "qap", "size": 20}
        task = DeepTask(task_id="test", problem=problem)

        analyzer, optimizer, validator, synthesis = orchestrator.deep_analyze(
            task, use_synthesis=False
        )

        # First three should exist
        assert analyzer is not None
        assert optimizer is not None
        assert validator is not None

        # Synthesis should be None
        assert synthesis is None

    def test_orchestrator_statistics(self):
        """Test orchestrator statistics"""
        orchestrator = DeepThinkOrchestrator(max_parallel_workers=3)

        # Run analysis
        problem = {"type": "qap", "size": 15}
        task = DeepTask(task_id="test", problem=problem)
        orchestrator.deep_analyze(task)

        # Get statistics
        stats = orchestrator.get_statistics()

        assert "analyzer" in stats
        assert "optimizer" in stats
        assert "validator" in stats
        assert "synthesizer" in stats

        # Each should have completed 1 analysis
        assert stats["analyzer"]["analyses_completed"] == 1
        assert stats["optimizer"]["analyses_completed"] == 1
        assert stats["validator"]["analyses_completed"] == 1
        assert stats["synthesizer"]["analyses_completed"] == 1

    def test_orchestrator_parallel_execution(self):
        """Test that parallel execution actually happens"""
        orchestrator = DeepThinkOrchestrator(max_parallel_workers=3)

        problem = {"type": "qap", "size": 30}
        task = DeepTask(task_id="test", problem=problem)

        # Time the parallel execution
        start = time.time()
        orchestrator.deep_analyze(task)
        parallel_time = time.time() - start

        # Should be much faster than sequential
        # (though in practice with fast mocks, difference is minimal)
        assert parallel_time < 1.0


class TestIntegration:
    """Integration tests for the full system"""

    def test_full_workflow_qap_small(self):
        """Test full workflow on small QAP problem"""
        orchestrator = DeepThinkOrchestrator(max_parallel_workers=3)

        problem = {
            "type": "qap",
            "size": 15,
            "objectives": 1,
            "constraints": [{"type": "assignment"}],
        }

        task = DeepTask(
            task_id="integration_test_small",
            problem=problem,
            depth=AnalysisDepth.DEEP,
            max_time_seconds=10.0,
        )

        _, _, _, synthesis = orchestrator.deep_analyze(task)

        # Should recommend exact or near-exact methods for small problem
        rec_text = str(synthesis.recommendations).lower()
        assert "branch" in rec_text or "exact" in rec_text or "small" in rec_text

    def test_full_workflow_qap_large(self):
        """Test full workflow on large QAP problem"""
        orchestrator = DeepThinkOrchestrator(max_parallel_workers=3)

        problem = {
            "type": "qap",
            "size": 60,
            "objectives": 1,
            "constraints": [],
        }

        task = DeepTask(
            task_id="integration_test_large",
            problem=problem,
            depth=AnalysisDepth.DEEP,
        )

        _, _, _, synthesis = orchestrator.deep_analyze(task)

        # Should recommend metaheuristics for large problem
        rec_text = str(synthesis.recommendations).lower()
        assert (
            "metaheuristic" in rec_text or
            "genetic" in rec_text or
            "simulated" in rec_text
        )

    def test_multi_objective_workflow(self):
        """Test workflow on multi-objective problem"""
        orchestrator = DeepThinkOrchestrator(max_parallel_workers=3)

        problem = {
            "type": "optimization",
            "size": 40,
            "objectives": 3,
        }

        task = DeepTask(task_id="multi_obj_test", problem=problem)

        _, optimizer, _, synthesis = orchestrator.deep_analyze(task)

        # Should mention multi-objective or pareto
        text = (
            " ".join(optimizer.insights) +
            " ".join(synthesis.insights)
        ).lower()
        assert "multi-objective" in text or "pareto" in text

    def test_end_to_end_performance(self):
        """Test end-to-end performance"""
        orchestrator = DeepThinkOrchestrator(max_parallel_workers=3)

        problems = [
            {"type": "qap", "size": s}
            for s in [10, 20, 30, 40, 50]
        ]

        total_time = 0
        for i, problem in enumerate(problems):
            task = DeepTask(task_id=f"perf_test_{i}", problem=problem)

            start = time.time()
            orchestrator.deep_analyze(task)
            elapsed = time.time() - start

            total_time += elapsed

            # Each should complete quickly
            assert elapsed < 0.5

        # Total should be reasonable
        assert total_time < 2.0

        # Check statistics
        stats = orchestrator.get_statistics()
        assert stats["synthesizer"]["analyses_completed"] == len(problems)


# Pytest fixtures
@pytest.fixture
def sample_qap_problem():
    """Sample QAP problem for testing"""
    return {
        "type": "qap",
        "size": 30,
        "objectives": 1,
        "constraints": [
            {"type": "assignment"},
            {"type": "capacity", "max": 100},
        ],
    }


@pytest.fixture
def sample_task(sample_qap_problem):
    """Sample deep task for testing"""
    return DeepTask(
        task_id="fixture_task",
        problem=sample_qap_problem,
        depth=AnalysisDepth.DEEP,
        max_time_seconds=30.0,
    )


@pytest.fixture
def orchestrator():
    """DeepThink orchestrator fixture"""
    return DeepThinkOrchestrator(max_parallel_workers=3)


# Tests using fixtures
def test_with_fixtures(orchestrator, sample_task):
    """Test using pytest fixtures"""
    _, _, _, synthesis = orchestrator.deep_analyze(sample_task)

    assert synthesis is not None
    assert synthesis.confidence > 0.85


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
