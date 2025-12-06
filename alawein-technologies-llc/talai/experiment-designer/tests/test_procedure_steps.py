"""Tests for experimental procedure step generation."""

import pytest
from experiment_designer.main import Step


class TestStepGeneration:
    """Test generation of experimental procedure steps."""

    def test_generates_multiple_steps(self, designer, sample_hypothesis):
        """Test that multiple procedure steps are generated."""
        protocol = designer.design_protocol(sample_hypothesis.hypothesis_id)

        assert len(protocol.steps) >= 5
        assert all(isinstance(step, Step) for step in protocol.steps)

    def test_steps_are_numbered_sequentially(self, designer, sample_hypothesis):
        """Test that steps have sequential numbering."""
        protocol = designer.design_protocol(sample_hypothesis.hypothesis_id)

        for i, step in enumerate(protocol.steps, 1):
            assert step.step_number == i

    def test_first_step_is_recruitment(self, designer, sample_hypothesis):
        """Test that first step is participant recruitment."""
        protocol = designer.design_protocol(sample_hypothesis.hypothesis_id)

        first_step = protocol.steps[0]
        assert 'Recruit' in first_step.action or 'recruit' in first_step.action

    def test_steps_include_baseline_measurement(self, designer, sample_hypothesis):
        """Test that steps include baseline measurement."""
        protocol = designer.design_protocol(sample_hypothesis.hypothesis_id)

        actions = [step.action for step in protocol.steps]
        assert any('baseline' in action.lower() for action in actions)

    def test_steps_include_intervention(self, designer, sample_hypothesis):
        """Test that steps include intervention/manipulation."""
        protocol = designer.design_protocol(sample_hypothesis.hypothesis_id)

        actions = [step.action for step in protocol.steps]
        assert any('intervention' in action.lower() or 'manipulate' in action.lower()
                   for action in actions)

    def test_steps_include_outcome_measurement(self, designer, sample_hypothesis):
        """Test that steps include outcome measurement."""
        protocol = designer.design_protocol(sample_hypothesis.hypothesis_id)

        actions = [step.action for step in protocol.steps]
        assert any('outcome' in action.lower() or 'measure' in action.lower()
                   for action in actions)

    def test_steps_include_analysis(self, designer, sample_hypothesis):
        """Test that steps include data analysis."""
        protocol = designer.design_protocol(sample_hypothesis.hypothesis_id)

        actions = [step.action for step in protocol.steps]
        assert any('analysis' in action.lower() for action in actions)

    def test_randomized_controlled_includes_randomization_step(self, designer, medicine_hypothesis_data):
        """Test that RCT design includes randomization step."""
        h = designer.submit_hypothesis(**medicine_hypothesis_data)
        protocol = designer.design_protocol(h.hypothesis_id)

        actions = [step.action for step in protocol.steps]
        assert any('randomize' in action.lower() for action in actions)

    def test_factorial_includes_randomization_step(self, designer, sample_hypothesis_data):
        """Test that factorial design includes randomization step."""
        h = designer.submit_hypothesis(**sample_hypothesis_data)
        protocol = designer.design_protocol(h.hypothesis_id)

        actions = [step.action for step in protocol.steps]
        assert any('randomize' in action.lower() for action in actions)

    def test_observational_may_not_include_randomization(self, designer, social_science_hypothesis_data):
        """Test that observational design may not include randomization."""
        h = designer.submit_hypothesis(**social_science_hypothesis_data)
        protocol = designer.design_protocol(h.hypothesis_id)

        # Observational designs don't require randomization step
        actions = [step.action for step in protocol.steps]
        # This is acceptable - observational studies don't randomize


class TestStepDetails:
    """Test details of individual steps."""

    def test_steps_have_duration(self, designer, sample_hypothesis):
        """Test that all steps have duration specified."""
        protocol = designer.design_protocol(sample_hypothesis.hypothesis_id)

        for step in protocol.steps:
            assert step.duration is not None
            assert len(step.duration) > 0

    def test_steps_have_equipment_needed(self, designer, sample_hypothesis):
        """Test that steps specify required equipment."""
        protocol = designer.design_protocol(sample_hypothesis.hypothesis_id)

        for step in protocol.steps:
            assert isinstance(step.equipment_needed, list)
            assert len(step.equipment_needed) > 0

    def test_steps_have_safety_notes(self, designer, sample_hypothesis):
        """Test that steps include safety notes."""
        protocol = designer.design_protocol(sample_hypothesis.hypothesis_id)

        for step in protocol.steps:
            assert isinstance(step.safety_notes, list)
            assert len(step.safety_notes) > 0

    def test_steps_have_quality_checks(self, designer, sample_hypothesis):
        """Test that steps include quality checks."""
        protocol = designer.design_protocol(sample_hypothesis.hypothesis_id)

        for step in protocol.steps:
            assert isinstance(step.quality_checks, list)
            assert len(step.quality_checks) > 0

    def test_steps_have_expected_output(self, designer, sample_hypothesis):
        """Test that steps specify expected output."""
        protocol = designer.design_protocol(sample_hypothesis.hypothesis_id)

        for step in protocol.steps:
            assert step.expected_output is not None
            assert len(step.expected_output) > 0

    def test_step_actions_reference_hypothesis_variables(self, designer, sample_hypothesis):
        """Test that steps reference hypothesis variables."""
        protocol = designer.design_protocol(sample_hypothesis.hypothesis_id)

        all_step_text = ' '.join([step.action for step in protocol.steps])
        # Should reference the dependent variable
        assert sample_hypothesis.variables['dependent'] in all_step_text
