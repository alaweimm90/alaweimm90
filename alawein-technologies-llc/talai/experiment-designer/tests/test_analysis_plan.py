"""Tests for analysis plan generation."""

import pytest


class TestOutcomePlanning:
    """Test outcome planning."""

    def test_has_primary_outcome(self, designer, sample_hypothesis):
        """Test that protocol has primary outcome."""
        protocol = designer.design_protocol(sample_hypothesis.hypothesis_id)

        assert protocol.primary_outcome is not None
        assert len(protocol.primary_outcome) > 0

    def test_primary_outcome_references_dependent_variable(self, designer, sample_hypothesis):
        """Test that primary outcome references dependent variable."""
        protocol = designer.design_protocol(sample_hypothesis.hypothesis_id)

        assert sample_hypothesis.variables['dependent'] in protocol.primary_outcome

    def test_has_secondary_outcomes(self, designer, sample_hypothesis):
        """Test that protocol has secondary outcomes."""
        protocol = designer.design_protocol(sample_hypothesis.hypothesis_id)

        assert len(protocol.secondary_outcomes) > 0
        assert all(isinstance(outcome, str) for outcome in protocol.secondary_outcomes)

    def test_secondary_outcomes_include_subgroup_analysis(self, designer, sample_hypothesis):
        """Test that secondary outcomes include subgroup analysis."""
        protocol = designer.design_protocol(sample_hypothesis.hypothesis_id)

        outcomes_text = ' '.join(protocol.secondary_outcomes)
        assert 'Subgroup' in outcomes_text or 'subgroup' in outcomes_text

    def test_secondary_outcomes_diverse(self, designer, sample_hypothesis):
        """Test that secondary outcomes cover different aspects."""
        protocol = designer.design_protocol(sample_hypothesis.hypothesis_id)

        # Should have multiple different types of secondary outcomes
        assert len(protocol.secondary_outcomes) >= 3


class TestStatisticalMethods:
    """Test statistical method selection."""

    def test_has_statistical_methods(self, designer, sample_hypothesis):
        """Test that protocol has statistical methods."""
        protocol = designer.design_protocol(sample_hypothesis.hypothesis_id)

        assert len(protocol.statistical_methods) > 0

    def test_randomized_controlled_has_appropriate_methods(self, designer, medicine_hypothesis_data):
        """Test that RCT design has appropriate statistical methods."""
        h = designer.submit_hypothesis(**medicine_hypothesis_data)
        protocol = designer.design_protocol(h.hypothesis_id)

        methods_text = ' '.join(protocol.statistical_methods)
        assert 'Intention-to-treat' in methods_text

    def test_factorial_has_appropriate_methods(self, designer, sample_hypothesis_data):
        """Test that factorial design has appropriate statistical methods."""
        h = designer.submit_hypothesis(**sample_hypothesis_data)
        protocol = designer.design_protocol(h.hypothesis_id)

        methods_text = ' '.join(protocol.statistical_methods)
        assert 'Factorial ANOVA' in methods_text or 'factorial' in methods_text.lower()

    def test_observational_has_appropriate_methods(self, designer, social_science_hypothesis_data):
        """Test that observational design has appropriate methods."""
        h = designer.submit_hypothesis(**social_science_hypothesis_data)
        protocol = designer.design_protocol(h.hypothesis_id)

        methods_text = ' '.join(protocol.statistical_methods)
        assert 'regression' in methods_text.lower() or 'Propensity' in methods_text

    def test_includes_multiple_correction(self, designer, sample_hypothesis):
        """Test that methods include multiple comparison correction."""
        protocol = designer.design_protocol(sample_hypothesis.hypothesis_id)

        methods_text = ' '.join(protocol.statistical_methods)
        # Should mention correction for multiple outcomes
        assert 'correction' in methods_text.lower() or 'false discovery rate' in methods_text.lower()


class TestDataCollectionPlan:
    """Test data collection planning."""

    def test_has_data_collection_plan(self, designer, sample_hypothesis):
        """Test that protocol has data collection plan."""
        protocol = designer.design_protocol(sample_hypothesis.hypothesis_id)

        assert protocol.data_collection_plan is not None
        assert len(protocol.data_collection_plan) > 0

    def test_data_plan_references_sample_size(self, designer, sample_hypothesis):
        """Test that data plan references required sample size."""
        protocol = designer.design_protocol(sample_hypothesis.hypothesis_id)

        assert str(protocol.power_analysis.required_sample_size) in protocol.data_collection_plan

    def test_data_plan_references_dependent_variable(self, designer, sample_hypothesis):
        """Test that data plan references dependent variable."""
        protocol = designer.design_protocol(sample_hypothesis.hypothesis_id)

        assert sample_hypothesis.variables['dependent'] in protocol.data_collection_plan

    def test_data_plan_mentions_data_quality(self, designer, sample_hypothesis):
        """Test that data plan mentions data quality measures."""
        protocol = designer.design_protocol(sample_hypothesis.hypothesis_id)

        plan_lower = protocol.data_collection_plan.lower()
        # Should mention quality measures
        assert 'duplicate' in plan_lower or 'quality' in plan_lower or 'checks' in plan_lower

    def test_data_plan_mentions_missing_data(self, designer, sample_hypothesis):
        """Test that data plan addresses missing data."""
        protocol = designer.design_protocol(sample_hypothesis.hypothesis_id)

        plan_lower = protocol.data_collection_plan.lower()
        assert 'missing' in plan_lower or 'imputation' in plan_lower

    def test_data_plan_mentions_data_security(self, designer, sample_hypothesis):
        """Test that data plan mentions data security."""
        protocol = designer.design_protocol(sample_hypothesis.hypothesis_id)

        plan_lower = protocol.data_collection_plan.lower()
        assert 'encrypted' in plan_lower or 'security' in plan_lower or 'privacy' in plan_lower
