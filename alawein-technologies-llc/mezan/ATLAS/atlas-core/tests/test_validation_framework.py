"""Comprehensive tests for the validation and A/B testing framework.

Author: Meshal Alawein
Date: 2025-11-18
"""

import json
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest
from scipy import stats

# Import all validation modules
import sys
sys.path.insert(0, '/home/user/AlaweinOS/MEZAN/ORCHEX/ORCHEX-core')

from atlas_core.validation import (
    StatisticalValidator, ValidationResult, TestType,
    MultipleComparisonMethod, quick_t_test, validate_improvement
)
from atlas_core.ab_testing import (
    ABTestingFramework, ABTestExperiment, ExperimentConfig,
    Variant, Metric, MetricType, SplitMethod, ExperimentStatus,
    quick_ab_test
)
from atlas_core.formal_verification import (
    WorkflowVerifier, WorkflowState, WorkflowTransition,
    Invariant, TemporalProperty, TemporalOperator, ResourceRequirement,
    VerificationStatus, verify_mezan_workflow, check_workflow_safety
)
from atlas_core.cross_validation import (
    CrossValidator, BaselineSystem, BenchmarkCase, PerformanceMetrics,
    ComparisonMethod, FoldStrategy, quick_cross_validation, compare_systems_quick
)


class TestStatisticalValidator:
    """Tests for the StatisticalValidator class."""

    def test_t_test_independent(self):
        """Test independent samples t-test."""
        validator = StatisticalValidator(confidence_level=0.95)

        # Create two groups with different means
        group1 = np.random.normal(10, 2, 100)
        group2 = np.random.normal(12, 2, 100)

        result = validator.t_test(group1, group2, paired=False)

        assert isinstance(result, ValidationResult)
        assert result.test_type == TestType.TTEST_IND
        assert result.p_value < 0.05  # Should be significant
        assert result.is_significant is True
        assert result.effect_size is not None
        assert result.confidence_interval is not None
        assert result.sample_size == 200

    def test_t_test_paired(self):
        """Test paired samples t-test."""
        validator = StatisticalValidator()

        # Create paired data with improvement
        baseline = np.random.normal(10, 2, 50)
        improved = baseline + np.random.normal(2, 1, 50)

        result = validator.t_test(improved, baseline, paired=True)

        assert result.test_type == TestType.TTEST_PAIRED
        assert result.is_significant is True
        assert result.effect_size > 0

    def test_chi_square_test(self):
        """Test chi-square test of independence."""
        validator = StatisticalValidator()

        # Create contingency table
        contingency = np.array([
            [10, 20, 30],
            [6, 9, 17]
        ])

        result = validator.chi_square_test(contingency)

        assert result.test_type == TestType.CHI_SQUARE
        assert result.p_value is not None
        assert result.effect_size is not None  # Cramér's V
        assert 'expected_frequencies' in result.additional_metrics

    def test_mann_whitney_u_test(self):
        """Test Mann-Whitney U test."""
        validator = StatisticalValidator()

        # Create non-normal data
        group1 = np.random.exponential(2, 50)
        group2 = np.random.exponential(3, 50)

        result = validator.mann_whitney_u_test(group1, group2)

        assert result.test_type == TestType.MANN_WHITNEY
        assert result.p_value is not None
        assert result.effect_size is not None  # Rank-biserial correlation

    def test_anova_test(self):
        """Test one-way ANOVA."""
        validator = StatisticalValidator()

        # Create three groups
        group1 = np.random.normal(10, 2, 30)
        group2 = np.random.normal(12, 2, 30)
        group3 = np.random.normal(14, 2, 30)

        result = validator.anova_test(group1, group2, group3)

        assert result.test_type == TestType.ANOVA
        assert result.is_significant is True
        assert result.effect_size is not None  # Eta-squared
        assert 'group_means' in result.additional_metrics

    def test_multiple_comparison_correction(self):
        """Test multiple comparison correction methods."""
        validator = StatisticalValidator()

        # Multiple p-values
        p_values = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06]

        # Test different methods
        for method in MultipleComparisonMethod:
            rejected, adjusted = validator.multiple_comparison_correction(
                p_values, method
            )

            assert len(rejected) == len(p_values)
            assert len(adjusted) == len(p_values)
            assert all(adj >= orig for adj, orig in zip(adjusted, p_values))

    def test_proportion_test(self):
        """Test proportion comparison."""
        validator = StatisticalValidator()

        result = validator.proportion_test(
            successes1=45, trials1=100,
            successes2=30, trials2=100
        )

        assert result.is_significant is True
        assert 'proportion1' in result.additional_metrics
        assert 'proportion2' in result.additional_metrics
        assert 'cohens_h' in result.additional_metrics

    def test_sample_size_calculation(self):
        """Test sample size calculation."""
        validator = StatisticalValidator()

        # T-test sample size
        n_ttest = validator.calculate_sample_size(
            effect_size=0.5, power=0.8, test_type='t-test'
        )
        assert n_ttest > 0
        assert isinstance(n_ttest, int)

        # Chi-square sample size
        n_chi = validator.calculate_sample_size(
            effect_size=0.3, power=0.8, test_type='chi-square'
        )
        assert n_chi > 0

    def test_quick_functions(self):
        """Test convenience functions."""
        result = quick_t_test([1, 2, 3, 4, 5], [2, 3, 4, 5, 6])
        assert 'p_value' in result
        assert 'effect_size' in result

        is_improved = validate_improvement(
            baseline=[1, 2, 3], improved=[2, 3, 4], min_effect_size=0.1
        )
        assert isinstance(is_improved, bool)


class TestABTestingFramework:
    """Tests for the A/B testing framework."""

    def test_experiment_creation(self):
        """Test creating an A/B test experiment."""
        config = ExperimentConfig(
            name="Test Experiment",
            description="Testing the framework",
            variants=[
                Variant("control", "Control group"),
                Variant("treatment", "Treatment group")
            ],
            metrics=[
                Metric("conversion", MetricType.CONVERSION, "Conversion rate", is_primary=True)
            ]
        )

        experiment = ABTestExperiment(config)

        assert experiment.status == ExperimentStatus.DRAFT
        assert len(experiment.config.variants) == 2
        assert len(experiment.config.metrics) == 1

    def test_variant_assignment(self):
        """Test variant assignment methods."""
        config = ExperimentConfig(
            name="Assignment Test",
            description="Testing assignment",
            variants=[
                Variant("A", "Variant A", weight=0.5),
                Variant("B", "Variant B", weight=0.5)
            ],
            metrics=[],
            split_method=SplitMethod.USER_ID
        )

        experiment = ABTestExperiment(config)
        experiment.start()

        # Test deterministic assignment
        user_id = "test_user_1"
        variant1 = experiment.assign_variant(user_id)
        variant2 = experiment.assign_variant(user_id)

        assert variant1 == variant2  # Same user should get same variant
        assert variant1 in ["A", "B"]

    def test_metric_tracking(self):
        """Test metric tracking."""
        config = ExperimentConfig(
            name="Metric Test",
            description="Testing metrics",
            variants=[
                Variant("control", "Control"),
                Variant("treatment", "Treatment")
            ],
            metrics=[
                Metric("clicks", MetricType.COUNT, "Click count")
            ]
        )

        experiment = ABTestExperiment(config)
        experiment.start()

        # Assign users and track metrics
        experiment.assignments["user1"] = "control"
        experiment.assignments["user2"] = "treatment"

        experiment.track_metric("user1", "clicks", 5)
        experiment.track_metric("user2", "clicks", 7)

        assert len(experiment.data) == 2
        assert "clicks" in experiment.metric_aggregates

    def test_experiment_analysis(self):
        """Test experiment analysis."""
        config = ExperimentConfig(
            name="Analysis Test",
            description="Testing analysis",
            variants=[
                Variant("control", "Control"),
                Variant("treatment", "Treatment")
            ],
            metrics=[
                Metric("conversion", MetricType.CONVERSION, "Conversion", is_primary=True)
            ],
            enable_bayesian=True
        )

        experiment = ABTestExperiment(config)
        experiment.start()

        # Simulate data
        np.random.seed(42)
        for i in range(100):
            user_id = f"user_{i}"
            variant = "control" if i < 50 else "treatment"
            experiment.assignments[user_id] = variant

            # Treatment has higher conversion rate
            if variant == "control":
                converted = np.random.random() < 0.1
            else:
                converted = np.random.random() < 0.15

            experiment.track_metric(user_id, "conversion", 1 if converted else 0)

        results = experiment.analyze_results()

        assert 'metrics' in results
        assert 'conversion' in results['metrics']
        assert 'frequentist' in results['metrics']['conversion']
        assert 'bayesian' in results['metrics']['conversion']

    def test_early_stopping(self):
        """Test early stopping mechanism."""
        config = ExperimentConfig(
            name="Early Stopping Test",
            description="Testing early stopping",
            variants=[
                Variant("control", "Control"),
                Variant("treatment", "Treatment")
            ],
            metrics=[
                Metric("conversion", MetricType.CONVERSION, "Conversion", is_primary=True)
            ],
            enable_early_stopping=True,
            early_stopping_threshold=0.05,
            target_sample_size=1000
        )

        experiment = ABTestExperiment(config)
        experiment.start()

        # Simulate strong effect for early stopping
        for i in range(200):
            user_id = f"user_{i}"
            variant = "control" if i < 100 else "treatment"
            experiment.assignments[user_id] = variant

            if variant == "control":
                converted = np.random.random() < 0.05
            else:
                converted = np.random.random() < 0.20  # Large effect

            experiment.track_metric(user_id, "conversion", 1 if converted else 0)

        results = experiment.analyze_results()

        # Check if early stopping would be triggered
        # (actual triggering depends on sequential testing boundaries)
        assert 'metrics' in results

    def test_ab_testing_framework(self, tmp_path):
        """Test the main A/B testing framework."""
        framework = ABTestingFramework(storage_path=tmp_path)

        config = ExperimentConfig(
            name="Framework Test",
            description="Testing framework",
            variants=[
                Variant("control", "Control"),
                Variant("treatment", "Treatment")
            ],
            metrics=[
                Metric("success", MetricType.CONVERSION, "Success rate")
            ]
        )

        experiment = framework.create_experiment(config)

        assert experiment.experiment_id in framework.experiments
        assert len(framework.list_experiments()) == 1

        # Test traffic routing
        experiment.start()
        variant = framework.route_traffic(experiment.experiment_id, "user123")
        assert variant in ["control", "treatment"]

    def test_sample_size_calculation(self):
        """Test sample size calculation for A/B tests."""
        framework = ABTestingFramework()

        n = framework.calculate_sample_size(
            baseline_rate=0.1,
            minimum_detectable_effect=0.2,  # 20% relative improvement
            power=0.8,
            confidence_level=0.95
        )

        assert n > 0
        assert isinstance(n, int)

    def test_quick_ab_test(self):
        """Test quick A/B test function."""
        control = [0, 1, 0, 1, 0, 1, 0, 0, 1, 0]  # 40% conversion
        treatment = [1, 1, 0, 1, 1, 1, 0, 1, 1, 0]  # 70% conversion

        results = quick_ab_test(control, treatment, "conversion")

        assert 'metrics' in results
        assert 'conversion' in results['metrics']
        assert results['metrics']['conversion']['frequentist']['is_significant']


class TestFormalVerification:
    """Tests for the formal verification framework."""

    def test_workflow_creation(self):
        """Test creating a workflow for verification."""
        verifier = WorkflowVerifier()

        # Add states
        verifier.add_state(WorkflowState("start", "Start", is_initial=True))
        verifier.add_state(WorkflowState("process", "Processing"))
        verifier.add_state(WorkflowState("end", "End", is_final=True))

        # Add transitions
        verifier.add_transition(WorkflowTransition("start", "process", "begin"))
        verifier.add_transition(WorkflowTransition("process", "end", "complete"))

        graph = verifier.build_graph()

        assert len(graph.nodes()) == 3
        assert len(graph.edges()) == 2

    def test_invariant_checking(self):
        """Test invariant verification."""
        verifier = WorkflowVerifier()

        # Create simple workflow
        verifier.add_state(WorkflowState("s1", "State 1", properties={"value": 5}))
        verifier.add_state(WorkflowState("s2", "State 2", properties={"value": 10}))

        # Add invariant
        verifier.add_invariant(Invariant(
            name="positive_value",
            description="Value must be positive",
            predicate=lambda s: s.properties.get("value", 0) > 0
        ))

        results = verifier.verify_invariants()

        assert len(results) == 1
        assert results[0].status == VerificationStatus.PASSED

    def test_deadlock_detection(self):
        """Test deadlock detection."""
        verifier = WorkflowVerifier()

        # Create workflow with potential deadlock
        verifier.add_state(WorkflowState("s1", "State 1", is_initial=True))
        verifier.add_state(WorkflowState("s2", "State 2"))
        verifier.add_state(WorkflowState("s3", "State 3"))  # No outgoing transitions
        verifier.add_state(WorkflowState("end", "End", is_final=True))

        verifier.add_transition(WorkflowTransition("s1", "s2", "go"))
        verifier.add_transition(WorkflowTransition("s2", "s3", "proceed"))
        # s3 has no outgoing transition - deadlock!

        result = verifier.detect_deadlocks()

        assert result.status == VerificationStatus.FAILED
        assert "s3" in result.details['terminal_deadlocks']

    def test_reachability_verification(self):
        """Test state reachability verification."""
        verifier = WorkflowVerifier()

        # Create workflow with unreachable state
        verifier.add_state(WorkflowState("start", "Start", is_initial=True))
        verifier.add_state(WorkflowState("middle", "Middle"))
        verifier.add_state(WorkflowState("unreachable", "Unreachable"))
        verifier.add_state(WorkflowState("end", "End", is_final=True))

        verifier.add_transition(WorkflowTransition("start", "middle", "go"))
        verifier.add_transition(WorkflowTransition("middle", "end", "finish"))
        # "unreachable" has no incoming transitions

        result = verifier.verify_reachability()

        assert result.status == VerificationStatus.WARNING
        assert "unreachable" in result.details['unreachable_states']

    def test_termination_verification(self):
        """Test termination verification."""
        verifier = WorkflowVerifier()

        # Create properly terminating workflow
        verifier.add_state(WorkflowState("start", "Start", is_initial=True))
        verifier.add_state(WorkflowState("process", "Process"))
        verifier.add_state(WorkflowState("end", "End", is_final=True))

        verifier.add_transition(WorkflowTransition("start", "process", "begin"))
        verifier.add_transition(WorkflowTransition("process", "end", "complete"))

        result = verifier.verify_termination()

        assert result.status == VerificationStatus.PASSED

    def test_resource_allocation(self):
        """Test resource allocation verification."""
        verifier = WorkflowVerifier()

        verifier.add_state(WorkflowState("s1", "State 1"))
        verifier.add_state(WorkflowState("s2", "State 2"))

        # Add resource requirements
        verifier.add_resource_requirement("s1", ResourceRequirement("cpu", 50, is_exclusive=True))
        verifier.add_resource_requirement("s1", ResourceRequirement("cpu", 30, is_exclusive=True))

        result = verifier.verify_resource_allocation()

        # Should fail due to multiple exclusive requests
        assert result.status == VerificationStatus.FAILED
        assert len(result.details['violations']) > 0

    def test_temporal_logic_ctl(self):
        """Test CTL temporal logic verification."""
        verifier = WorkflowVerifier()

        # Create workflow
        verifier.add_state(WorkflowState("s1", "State 1", is_initial=True,
                                        properties={"safe": True}))
        verifier.add_state(WorkflowState("s2", "State 2",
                                        properties={"safe": True}))
        verifier.add_state(WorkflowState("s3", "State 3", is_final=True,
                                        properties={"safe": True}))

        verifier.add_transition(WorkflowTransition("s1", "s2", "go"))
        verifier.add_transition(WorkflowTransition("s2", "s3", "finish"))

        # Add AG property (always globally safe)
        verifier.add_temporal_property(TemporalProperty(
            name="always_safe",
            formula="AG(safe)",
            operator=TemporalOperator.AG,
            predicate=lambda s: s.properties.get("safe", False),
            description="System is always safe"
        ))

        results = verifier.verify_temporal_logic()

        assert len(results) == 1
        assert results[0].status == VerificationStatus.PASSED

    def test_convenience_functions(self):
        """Test convenience verification functions."""
        workflow_def = {
            'states': [
                {'id': 's1', 'name': 'Start', 'is_initial': True},
                {'id': 's2', 'name': 'End', 'is_final': True}
            ],
            'transitions': [
                {'source': 's1', 'target': 's2', 'action': 'complete'}
            ]
        }

        report = verify_mezan_workflow(workflow_def)

        assert 'verification_summary' in report
        assert report['verification_summary']['failed'] == 0

        # Test quick safety check
        is_safe = check_workflow_safety(
            states=[
                {'id': 'a', 'is_initial': True},
                {'id': 'b', 'is_final': True}
            ],
            transitions=[
                {'source': 'a', 'target': 'b'}
            ]
        )

        assert is_safe is True


class TestCrossValidation:
    """Tests for the cross-validation framework."""

    def test_cross_validator_creation(self):
        """Test creating a cross-validator."""
        mezan_executor = Mock(return_value="result")
        validator = CrossValidator(mezan_executor)

        assert validator.mezan_executor is not None
        assert len(validator.baselines) == 0
        assert len(validator.benchmarks) == 0

    def test_baseline_addition(self):
        """Test adding baseline systems."""
        validator = CrossValidator(Mock())

        baseline = BaselineSystem(
            name="baseline1",
            version="1.0",
            executor=Mock()
        )

        validator.add_baseline(baseline)

        assert "baseline1" in validator.baselines
        assert validator.baselines["baseline1"] == baseline

    def test_benchmark_addition(self):
        """Test adding benchmarks."""
        validator = CrossValidator(Mock())

        benchmark = BenchmarkCase(
            id="bench1",
            name="Test Benchmark",
            category="test",
            input_data={"data": [1, 2, 3]},
            expected_output={"result": 6}
        )

        validator.add_benchmark(benchmark)

        assert len(validator.benchmarks) == 1
        assert validator.benchmarks[0] == benchmark

    def test_cross_validation_k_fold(self):
        """Test k-fold cross-validation."""
        validator = CrossValidator(Mock())

        # Simple predictor that returns first element
        solver_selector = lambda x: x[0] if isinstance(x, (list, np.ndarray)) else str(x)

        problems = [
            ['solver1', 1, 2],
            ['solver2', 2, 3],
            ['solver1', 3, 4],
            ['solver2', 4, 5]
        ]
        labels = ['solver1', 'solver2', 'solver1', 'solver2']

        result = validator.cross_validate_solver_selection(
            solver_selector,
            problems,
            labels,
            strategy=FoldStrategy.K_FOLD,
            n_folds=2
        )

        assert result.method == FoldStrategy.K_FOLD
        assert result.n_folds == 2
        assert len(result.scores) == 2
        assert result.mean_score >= 0
        assert result.std_score >= 0

    def test_system_comparison(self, tmp_path):
        """Test comparing against baseline."""
        # Create mock executors
        mezan_executor = Mock(return_value={"result": 10})
        baseline_executor = Mock(return_value={"result": 8})

        validator = CrossValidator(mezan_executor, storage_path=tmp_path)

        baseline = BaselineSystem(
            name="baseline",
            version="1.0",
            executor=baseline_executor
        )
        validator.add_baseline(baseline)

        # Add benchmark
        validator.add_benchmark(BenchmarkCase(
            id="test1",
            name="Test",
            category="test",
            input_data={"x": 5},
            expected_output=10
        ))

        with patch.object(validator, '_run_system') as mock_run:
            # Mock performance metrics
            mezan_metrics = PerformanceMetrics(execution_time=1.0, accuracy=0.9)
            baseline_metrics = PerformanceMetrics(execution_time=1.5, accuracy=0.8)

            mock_run.side_effect = [mezan_metrics, baseline_metrics]

            results = validator.compare_against_baseline("baseline")

            assert len(results) == 1
            assert results[0].is_better is True
            assert results[0].improvement_percentage > 0

    def test_performance_regression_detection(self):
        """Test regression detection."""
        validator = CrossValidator(Mock())

        historical = [
            {"benchmark_id": "b1", "execution_time": 1.0},
            {"benchmark_id": "b1", "execution_time": 1.1},
            {"benchmark_id": "b1", "execution_time": 0.9}
        ]

        current = {
            "b1": {"execution_time": 1.5}  # 50% slower - regression!
        }

        analysis = validator.detect_performance_regression(
            historical, current, threshold=0.1
        )

        assert analysis['overall_trend'] == 'regression'
        assert len(analysis['regressions']) == 1
        assert analysis['regressions'][0]['benchmark_id'] == 'b1'

    def test_quick_comparison(self):
        """Test quick comparison function."""
        system1_results = [1.0, 1.1, 0.9, 1.2, 0.8]
        system2_results = [1.5, 1.6, 1.4, 1.7, 1.3]

        comparison = compare_systems_quick(
            system1_results, system2_results,
            "MEZAN", "Baseline"
        )

        assert comparison['better_system'] == "MEZAN"
        assert comparison['is_significant'] is True
        assert comparison['p_value'] < 0.05


class TestIntegration:
    """Integration tests across modules."""

    def test_full_validation_pipeline(self, tmp_path):
        """Test complete validation pipeline."""
        # Create validator
        validator = StatisticalValidator()

        # Generate synthetic data
        np.random.seed(42)
        control_data = np.random.normal(10, 2, 100)
        treatment_data = np.random.normal(11, 2, 100)

        # Run validation
        result = validator.t_test(treatment_data, control_data)

        # Generate report
        report = validator.generate_validation_report(tmp_path / "report.json")

        assert report['summary']['total_tests'] == 1
        assert result.is_significant is True

    def test_ab_test_with_validation(self):
        """Test A/B test with statistical validation."""
        # Create experiment
        config = ExperimentConfig(
            name="Integration Test",
            description="Testing integration",
            variants=[
                Variant("control", "Control"),
                Variant("treatment", "Treatment")
            ],
            metrics=[
                Metric("score", MetricType.CONTINUOUS, "Performance score", is_primary=True)
            ]
        )

        experiment = ABTestExperiment(config)
        experiment.start()

        # Generate and track data
        np.random.seed(42)
        for i in range(200):
            user_id = f"user_{i}"
            variant = "control" if i < 100 else "treatment"
            experiment.assignments[user_id] = variant

            if variant == "control":
                score = np.random.normal(50, 10)
            else:
                score = np.random.normal(55, 10)  # Treatment is better

            experiment.track_metric(user_id, "score", score)

        # Analyze with validation
        results = experiment.analyze_results()

        assert results['metrics']['score']['frequentist']['is_significant']
        assert results['recommendation']['action'] == 'deploy_treatment'

    def test_workflow_verification_with_resources(self):
        """Test workflow verification with resource constraints."""
        verifier = WorkflowVerifier()

        # Create complex workflow
        states = ["init", "load", "process1", "process2", "merge", "output"]
        for i, state_id in enumerate(states):
            is_initial = i == 0
            is_final = i == len(states) - 1
            verifier.add_state(WorkflowState(
                state_id, state_id.capitalize(),
                is_initial=is_initial, is_final=is_final
            ))

        # Add transitions
        transitions = [
            ("init", "load"), ("load", "process1"),
            ("load", "process2"), ("process1", "merge"),
            ("process2", "merge"), ("merge", "output")
        ]
        for src, dst in transitions:
            verifier.add_transition(WorkflowTransition(src, dst, f"{src}_to_{dst}"))

        # Add resources
        verifier.add_resource_requirement("process1", ResourceRequirement("gpu", 1, is_exclusive=True))
        verifier.add_resource_requirement("process2", ResourceRequirement("gpu", 1, is_exclusive=True))

        # Run full verification
        report = verifier.generate_verification_report()

        assert report['verification_summary']['failed'] == 0
        assert report['workflow_stats']['total_states'] == 6
        assert report['workflow_stats']['total_transitions'] == 6

    def test_cross_validation_with_real_workflow(self, tmp_path):
        """Test cross-validation with realistic workflow."""
        # Create MEZAN executor mock
        def mezan_executor(input_data):
            # Simulate MEZAN processing
            time.sleep(0.01)  # Simulate work
            return {"result": sum(input_data.get("values", []))}

        # Create baseline executor
        def baseline_executor(input_data):
            # Simulate baseline processing (slower)
            time.sleep(0.02)
            return {"result": sum(input_data.get("values", []))}

        validator = CrossValidator(mezan_executor, storage_path=tmp_path)

        # Add baseline
        validator.add_baseline(BaselineSystem(
            name="SimpleSum",
            version="1.0",
            executor=baseline_executor
        ))

        # Add benchmarks
        for i in range(5):
            validator.add_benchmark(BenchmarkCase(
                id=f"sum_{i}",
                name=f"Sum Test {i}",
                category="arithmetic",
                input_data={"values": list(range(i, i+10))},
                expected_output=sum(range(i, i+10))
            ))

        # Run comparison
        results = validator.compare_against_baseline("SimpleSum")

        assert len(results) == 5
        # MEZAN should be faster (shorter sleep)
        assert all(r.is_better for r in results)

        # Generate report
        report = validator.generate_comparison_report("SimpleSum", tmp_path / "comparison.json")

        assert report['summary']['win_rate'] == 100.0


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])