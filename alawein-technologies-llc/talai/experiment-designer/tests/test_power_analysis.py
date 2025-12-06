"""Tests for statistical power analysis calculations."""

import pytest
from experiment_designer.main import PowerAnalysis


class TestPowerCalculation:
    """Test power analysis calculations."""

    def test_small_effect_size_calculation(self, designer):
        """Test power calculation for small effect size."""
        h = designer.submit_hypothesis(
            statement='Test',
            domain='psychology',
            independent_var='A',
            dependent_var='B',
            control_vars=['C'],
            expected_effect_size='small',
            mechanism='Test',
            prior_evidence='Test'
        )
        protocol = designer.design_protocol(h.hypothesis_id)

        assert protocol.power_analysis.effect_size == 0.2
        # Small effects need larger samples
        assert protocol.power_analysis.required_sample_size > 100

    def test_medium_effect_size_calculation(self, designer):
        """Test power calculation for medium effect size."""
        h = designer.submit_hypothesis(
            statement='Test',
            domain='psychology',
            independent_var='A',
            dependent_var='B',
            control_vars=['C'],
            expected_effect_size='medium',
            mechanism='Test',
            prior_evidence='Test'
        )
        protocol = designer.design_protocol(h.hypothesis_id)

        assert protocol.power_analysis.effect_size == 0.5
        assert 50 < protocol.power_analysis.required_sample_size < 200

    def test_large_effect_size_calculation(self, designer):
        """Test power calculation for large effect size."""
        h = designer.submit_hypothesis(
            statement='Test',
            domain='psychology',
            independent_var='A',
            dependent_var='B',
            control_vars=['C'],
            expected_effect_size='large',
            mechanism='Test',
            prior_evidence='Test'
        )
        protocol = designer.design_protocol(h.hypothesis_id)

        assert protocol.power_analysis.effect_size == 0.8
        # Large effects need smaller samples
        assert protocol.power_analysis.required_sample_size < 100

    def test_power_analysis_has_standard_alpha(self, designer, sample_hypothesis):
        """Test that alpha is set to standard 0.05."""
        protocol = designer.design_protocol(sample_hypothesis.hypothesis_id)

        assert protocol.power_analysis.alpha == 0.05

    def test_power_analysis_has_standard_power(self, designer, sample_hypothesis):
        """Test that power is set to standard 0.80."""
        protocol = designer.design_protocol(sample_hypothesis.hypothesis_id)

        assert protocol.power_analysis.power == 0.80

    def test_power_analysis_has_two_groups(self, designer, sample_hypothesis):
        """Test that power analysis assumes 2 groups."""
        protocol = designer.design_protocol(sample_hypothesis.hypothesis_id)

        assert protocol.power_analysis.groups == 2

    def test_sample_size_equals_groups_times_per_group(self, designer, sample_hypothesis):
        """Test that total sample size calculation is correct."""
        protocol = designer.design_protocol(sample_hypothesis.hypothesis_id)

        expected_total = protocol.power_analysis.observations_per_group * protocol.power_analysis.groups
        assert protocol.power_analysis.required_sample_size == expected_total

    def test_power_analysis_has_assumptions(self, designer, sample_hypothesis):
        """Test that power analysis includes statistical assumptions."""
        protocol = designer.design_protocol(sample_hypothesis.hypothesis_id)

        assumptions = protocol.power_analysis.assumptions
        assert len(assumptions) > 0
        assert any('Normal distribution' in a for a in assumptions)
        assert any('variance' in a for a in assumptions)
        assert any('Independence' in a for a in assumptions)

    def test_power_analysis_has_sensitivity_analysis(self, designer, sample_hypothesis):
        """Test that sensitivity analysis is included."""
        protocol = designer.design_protocol(sample_hypothesis.hypothesis_id)

        sensitivity = protocol.power_analysis.sensitivity_analysis
        assert len(sensitivity) == 3
        assert 'small (d=0.2)' in sensitivity
        assert 'medium (d=0.5)' in sensitivity
        assert 'large (d=0.8)' in sensitivity


class TestStatisticalTestSelection:
    """Test selection of appropriate statistical tests."""

    def test_medicine_gets_ttest_or_anova(self, designer, medicine_hypothesis_data):
        """Test that medicine domain gets appropriate statistical test."""
        h = designer.submit_hypothesis(**medicine_hypothesis_data)
        protocol = designer.design_protocol(h.hypothesis_id)

        assert protocol.power_analysis.statistical_test == 't-test or ANOVA'

    def test_psychology_gets_mixed_effects(self, designer, sample_hypothesis_data):
        """Test that psychology domain gets mixed-effects model."""
        h = designer.submit_hypothesis(**sample_hypothesis_data)
        protocol = designer.design_protocol(h.hypothesis_id)

        assert protocol.power_analysis.statistical_test == 'Mixed-effects model'

    def test_biology_gets_ttest(self, designer, biology_hypothesis_data):
        """Test that biology domain gets t-test."""
        h = designer.submit_hypothesis(**biology_hypothesis_data)
        protocol = designer.design_protocol(h.hypothesis_id)

        assert protocol.power_analysis.statistical_test == 't-test'

    def test_physics_gets_linear_regression(self, designer, physics_hypothesis_data):
        """Test that physics domain gets linear regression."""
        h = designer.submit_hypothesis(**physics_hypothesis_data)
        protocol = designer.design_protocol(h.hypothesis_id)

        assert protocol.power_analysis.statistical_test == 'Linear regression'

    def test_social_science_gets_chisquare_or_regression(self, designer, social_science_hypothesis_data):
        """Test that social science gets chi-square or regression."""
        h = designer.submit_hypothesis(**social_science_hypothesis_data)
        protocol = designer.design_protocol(h.hypothesis_id)

        assert protocol.power_analysis.statistical_test == 'Chi-square or regression'

    def test_unknown_domain_defaults_to_ttest(self, designer):
        """Test that unknown domain defaults to t-test."""
        h = designer.submit_hypothesis(
            statement='Test',
            domain='unknown',
            independent_var='A',
            dependent_var='B',
            control_vars=['C'],
            expected_effect_size='medium',
            mechanism='Test',
            prior_evidence='Test'
        )
        protocol = designer.design_protocol(h.hypothesis_id)

        assert protocol.power_analysis.statistical_test == 't-test'
