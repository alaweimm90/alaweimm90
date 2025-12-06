"""Tests for control variable identification and management."""

import pytest
from experiment_designer.main import ControlVariable


class TestControlVariableIdentification:
    """Test control variable identification."""

    def test_identifies_all_control_variables(self, designer, sample_hypothesis):
        """Test that all control variables are identified."""
        protocol = designer.design_protocol(sample_hypothesis.hypothesis_id)

        control_names = [cv.name for cv in protocol.control_vars]
        assert 'Age' in control_names
        assert 'Sleep duration' in control_names
        assert 'Time of day' in control_names

    def test_control_variables_have_valid_types(self, designer, sample_hypothesis):
        """Test that control variables have valid types."""
        protocol = designer.design_protocol(sample_hypothesis.hypothesis_id)

        valid_types = ['randomize', 'block', 'measure', 'hold_constant']
        for cv in protocol.control_vars:
            assert cv.type in valid_types

    def test_control_variables_have_justification(self, designer, sample_hypothesis):
        """Test that each control variable has justification."""
        protocol = designer.design_protocol(sample_hypothesis.hypothesis_id)

        for cv in protocol.control_vars:
            assert cv.justification is not None
            assert len(cv.justification) > 0
            assert cv.name in cv.justification

    def test_control_variables_have_method(self, designer, sample_hypothesis):
        """Test that each control variable has a method."""
        protocol = designer.design_protocol(sample_hypothesis.hypothesis_id)

        for cv in protocol.control_vars:
            assert cv.method is not None
            assert len(cv.method) > 0
            assert cv.name in cv.method

    def test_empty_control_vars_handled(self, designer):
        """Test that empty control variables are handled gracefully."""
        h = designer.submit_hypothesis(
            statement='Test',
            domain='psychology',
            independent_var='A',
            dependent_var='B',
            control_vars=[''],
            expected_effect_size='medium',
            mechanism='Test',
            prior_evidence='Test'
        )
        protocol = designer.design_protocol(h.hypothesis_id)

        # Should handle empty strings gracefully
        assert isinstance(protocol.control_vars, list)

    def test_single_control_variable(self, designer):
        """Test handling of single control variable."""
        h = designer.submit_hypothesis(
            statement='Test',
            domain='psychology',
            independent_var='A',
            dependent_var='B',
            control_vars=['Age'],
            expected_effect_size='medium',
            mechanism='Test',
            prior_evidence='Test'
        )
        protocol = designer.design_protocol(h.hypothesis_id)

        assert len(protocol.control_vars) == 1
        assert protocol.control_vars[0].name == 'Age'

    def test_multiple_control_variables(self, designer):
        """Test handling of multiple control variables."""
        h = designer.submit_hypothesis(
            statement='Test',
            domain='psychology',
            independent_var='A',
            dependent_var='B',
            control_vars=['Age', 'Gender', 'Education', 'Income'],
            expected_effect_size='medium',
            mechanism='Test',
            prior_evidence='Test'
        )
        protocol = designer.design_protocol(h.hypothesis_id)

        assert len(protocol.control_vars) == 4
        control_names = [cv.name for cv in protocol.control_vars]
        assert 'Age' in control_names
        assert 'Gender' in control_names
        assert 'Education' in control_names
        assert 'Income' in control_names


class TestControlVariableTypes:
    """Test different control variable types and their justifications."""

    def test_randomize_type_has_appropriate_justification(self, designer, sample_hypothesis):
        """Test that randomize type has correct justification."""
        protocol = designer.design_protocol(sample_hypothesis.hypothesis_id)

        randomize_controls = [cv for cv in protocol.control_vars if cv.type == 'randomize']
        for cv in randomize_controls:
            assert 'Randomization eliminates systematic bias' in cv.justification

    def test_block_type_has_appropriate_justification(self, designer, sample_hypothesis):
        """Test that block type has correct justification."""
        protocol = designer.design_protocol(sample_hypothesis.hypothesis_id)

        block_controls = [cv for cv in protocol.control_vars if cv.type == 'block']
        for cv in block_controls:
            assert 'Blocking' in cv.justification
            assert 'reduces within-group variance' in cv.justification

    def test_measure_type_has_appropriate_justification(self, designer, sample_hypothesis):
        """Test that measure type has correct justification."""
        protocol = designer.design_protocol(sample_hypothesis.hypothesis_id)

        measure_controls = [cv for cv in protocol.control_vars if cv.type == 'measure']
        for cv in measure_controls:
            assert 'Measuring' in cv.justification
            assert 'allows statistical adjustment' in cv.justification

    def test_hold_constant_type_has_appropriate_justification(self, designer, sample_hypothesis):
        """Test that hold_constant type has correct justification."""
        protocol = designer.design_protocol(sample_hypothesis.hypothesis_id)

        constant_controls = [cv for cv in protocol.control_vars if cv.type == 'hold_constant']
        for cv in constant_controls:
            assert 'Holding' in cv.justification
            assert 'constant isolates effect of interest' in cv.justification
