"""
Unit tests for HELIOS Validation Module.

Tests for:
- TuringValidator
- Falsification strategies
- Interrogator
"""

import pytest


class TestTuringValidator:
    """Test TuringValidator class."""

    @pytest.mark.unit
    def test_validation_result_structure(self, mock_validation_result):
        """Test that validation result has required fields."""
        required_fields = [
            'hypothesis_id',
            'logical_score',
            'empirical_score',
            'analogical_score',
            'boundary_score',
            'mechanism_score',
            'overall_score',
        ]
        for field in required_fields:
            assert field in mock_validation_result

    @pytest.mark.unit
    def test_scores_in_valid_range(self, mock_validation_result):
        """Test that all scores are in valid range [0, 100]."""
        score_fields = [
            'logical_score',
            'empirical_score',
            'analogical_score',
            'boundary_score',
            'mechanism_score',
            'overall_score',
        ]
        for field in score_fields:
            score = mock_validation_result[field]
            assert 0 <= score <= 100, f"{field} = {score} out of range"

    @pytest.mark.unit
    def test_overall_score_weighted_avg(self, mock_validation_result):
        """Test that overall score is reasonable weighted average."""
        # Overall should be between min and max of component scores
        scores = [
            mock_validation_result['logical_score'],
            mock_validation_result['empirical_score'],
            mock_validation_result['analogical_score'],
            mock_validation_result['boundary_score'],
            mock_validation_result['mechanism_score'],
        ]
        overall = mock_validation_result['overall_score']
        assert min(scores) <= overall <= max(scores)

    @pytest.mark.unit
    def test_validation_weaknesses(self, mock_validation_result):
        """Test that validation includes weaknesses."""
        assert 'weaknesses' in mock_validation_result
        assert isinstance(mock_validation_result['weaknesses'], list)

    @pytest.mark.unit
    def test_interrogation_results(self, mock_validation_result):
        """Test that interrogation results are present."""
        assert 'interrogation_results' in mock_validation_result
        ir = mock_validation_result['interrogation_results']
        assert 'num_questions' in ir
        assert ir['num_questions'] > 0

    @pytest.mark.unit
    @pytest.mark.requires_api
    def test_validate_basic(self, mock_hypothesis):
        """Test basic validation (requires API)."""
        try:
            from helios.core.validation.turing import TuringValidator

            validator = TuringValidator()
            result = validator.validate([mock_hypothesis])

            assert isinstance(result, list)
            assert len(result) > 0
            assert result[0]['overall_score'] > 0
        except ImportError:
            pytest.skip("TuringValidator not available")

    @pytest.mark.unit
    def test_multiple_hypotheses_validation(self, mock_hypotheses, mock_validation_results):
        """Test validation of multiple hypotheses."""
        assert len(mock_hypotheses) == len(mock_validation_results)

        for h, r in zip(mock_hypotheses, mock_validation_results):
            assert h['id'] == r['hypothesis_id']
            assert isinstance(r['overall_score'], (int, float))

    @pytest.mark.unit
    def test_validation_timestamp(self, mock_validation_result):
        """Test that validation includes timestamp."""
        assert 'validation_timestamp' in mock_validation_result


class TestFalsificationStrategies:
    """Test falsification strategies."""

    @pytest.mark.unit
    def test_logical_contradiction_strategy(self):
        """Test logical contradiction detection."""
        try:
            from helios.core.validation.turing.logical_contradiction import (
                LogicalContradictionStrategy
            )
            assert LogicalContradictionStrategy is not None
        except ImportError:
            pytest.skip("LogicalContradictionStrategy not available")

    @pytest.mark.unit
    def test_empirical_counter_example_strategy(self):
        """Test empirical counter-example strategy."""
        try:
            from helios.core.validation.turing.empirical_counter_example import (
                EmpiricalCounterExampleStrategy
            )
            assert EmpiricalCounterExampleStrategy is not None
        except ImportError:
            pytest.skip("EmpiricalCounterExampleStrategy not available")

    @pytest.mark.unit
    def test_analogical_falsification_strategy(self):
        """Test analogical falsification strategy."""
        try:
            from helios.core.validation.turing.analogical_falsification import (
                AnalogicalFalsificationStrategy
            )
            assert AnalogicalFalsificationStrategy is not None
        except ImportError:
            pytest.skip("AnalogicalFalsificationStrategy not available")

    @pytest.mark.unit
    def test_boundary_violation_strategy(self):
        """Test boundary violation strategy."""
        try:
            from helios.core.validation.turing.boundary_violation import (
                BoundaryViolationStrategy
            )
            assert BoundaryViolationStrategy is not None
        except ImportError:
            pytest.skip("BoundaryViolationStrategy not available")

    @pytest.mark.unit
    def test_mechanism_implausibility_strategy(self):
        """Test mechanism implausibility strategy."""
        try:
            from helios.core.validation.turing.mechanism_implausibility import (
                MechanismImplausibilityStrategy
            )
            assert MechanismImplausibilityStrategy is not None
        except ImportError:
            pytest.skip("MechanismImplausibilityStrategy not available")


class TestInterrogator:
    """Test 200-question interrogation framework."""

    @pytest.mark.unit
    def test_interrogator_exists(self):
        """Test that Interrogator can be imported."""
        try:
            from helios.core.validation.turing.interrogator import Interrogator
            assert Interrogator is not None
        except ImportError:
            pytest.skip("Interrogator not available")

    @pytest.mark.unit
    @pytest.mark.requires_api
    def test_interrogator_questions(self):
        """Test that interrogator generates critical questions."""
        try:
            from helios.core.validation.turing.interrogator import Interrogator

            interrogator = Interrogator()
            questions = interrogator.generate_questions("test hypothesis")

            assert isinstance(questions, list)
            assert len(questions) > 0
        except ImportError:
            pytest.skip("Interrogator not available")
