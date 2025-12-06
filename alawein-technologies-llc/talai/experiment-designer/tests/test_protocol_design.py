"""Tests for protocol design and generation."""

import pytest
from experiment_designer.main import ExperimentProtocol, PowerAnalysis


class TestProtocolGeneration:
    """Test protocol generation from hypotheses."""

    def test_design_protocol_creates_protocol(self, designer, sample_hypothesis):
        """Test that design_protocol creates ExperimentProtocol object."""
        protocol = designer.design_protocol(sample_hypothesis.hypothesis_id)

        assert isinstance(protocol, ExperimentProtocol)
        assert protocol.hypothesis_id == sample_hypothesis.hypothesis_id
        assert protocol.protocol_id == 1

    def test_design_protocol_increments_id(self, designer, sample_hypothesis):
        """Test that protocol IDs increment correctly."""
        p1 = designer.design_protocol(sample_hypothesis.hypothesis_id)
        p2 = designer.design_protocol(sample_hypothesis.hypothesis_id)
        p3 = designer.design_protocol(sample_hypothesis.hypothesis_id)

        assert p1.protocol_id == 1
        assert p2.protocol_id == 2
        assert p3.protocol_id == 3

    def test_design_protocol_nonexistent_hypothesis_raises_error(self, designer):
        """Test that designing protocol for nonexistent hypothesis raises error."""
        with pytest.raises(ValueError, match="Hypothesis 999 not found"):
            designer.design_protocol(999)

    def test_protocol_has_all_required_fields(self, designer, sample_hypothesis):
        """Test that protocol contains all required fields."""
        protocol = designer.design_protocol(sample_hypothesis.hypothesis_id)

        # Core design
        assert protocol.design_type is not None
        assert protocol.blind_level is not None
        assert protocol.randomization_method is not None

        # Power analysis
        assert isinstance(protocol.power_analysis, PowerAnalysis)

        # Variables
        assert len(protocol.independent_vars) > 0
        assert len(protocol.dependent_vars) > 0
        assert len(protocol.control_vars) >= 0

        # Procedure
        assert len(protocol.steps) > 0

        # Resources
        assert len(protocol.equipment_list) > 0
        assert len(protocol.personnel_needed) > 0

        # Timeline
        assert len(protocol.timeline) > 0
        assert protocol.total_duration_days > 0

        # Budget
        assert protocol.total_cost > 0
        assert 'equipment' in protocol.cost_breakdown
        assert 'personnel' in protocol.cost_breakdown
        assert 'overhead' in protocol.cost_breakdown

        # Analysis
        assert protocol.primary_outcome is not None
        assert len(protocol.secondary_outcomes) > 0
        assert len(protocol.statistical_methods) > 0

        # Quality
        assert len(protocol.quality_assurance) > 0
        assert len(protocol.risks) > 0
        assert len(protocol.ethics_considerations) > 0

        # Metadata
        assert 0 <= protocol.confidence_score <= 1
        assert len(protocol.limitations) > 0
        assert protocol.created_at is not None

    def test_protocol_persists_to_file(self, designer, sample_hypothesis, temp_data_file):
        """Test that protocol is saved to JSON file."""
        import json

        protocol = designer.design_protocol(sample_hypothesis.hypothesis_id)

        with open(temp_data_file, 'r') as f:
            data = json.load(f)

        assert str(protocol.protocol_id) in data['protocols']

    def test_protocol_loads_from_file(self, temp_data_file, sample_hypothesis_data):
        """Test that protocols are loaded from file."""
        from experiment_designer.main import ExperimentDesigner

        # Create and save protocol
        designer1 = ExperimentDesigner(data_file=temp_data_file)
        h = designer1.submit_hypothesis(**sample_hypothesis_data)
        p = designer1.design_protocol(h.hypothesis_id)

        # Load from file
        designer2 = ExperimentDesigner(data_file=temp_data_file)

        assert p.protocol_id in designer2.protocols
        loaded_p = designer2.protocols[p.protocol_id]
        assert loaded_p.hypothesis_id == p.hypothesis_id
        assert loaded_p.design_type == p.design_type


class TestDesignTypeSelection:
    """Test experimental design type selection."""

    def test_medicine_domain_gets_randomized_controlled(self, designer, medicine_hypothesis_data):
        """Test that medicine domain gets randomized controlled design."""
        h = designer.submit_hypothesis(**medicine_hypothesis_data)
        protocol = designer.design_protocol(h.hypothesis_id)

        assert protocol.design_type == 'randomized_controlled'

    def test_psychology_domain_gets_factorial(self, designer, sample_hypothesis_data):
        """Test that psychology domain gets factorial design."""
        h = designer.submit_hypothesis(**sample_hypothesis_data)
        protocol = designer.design_protocol(h.hypothesis_id)

        assert protocol.design_type == 'factorial'

    def test_biology_domain_gets_randomized_controlled(self, designer, biology_hypothesis_data):
        """Test that biology domain gets randomized controlled design."""
        h = designer.submit_hypothesis(**biology_hypothesis_data)
        protocol = designer.design_protocol(h.hypothesis_id)

        assert protocol.design_type == 'randomized_controlled'

    def test_physics_domain_gets_factorial(self, designer, physics_hypothesis_data):
        """Test that physics domain gets factorial design."""
        h = designer.submit_hypothesis(**physics_hypothesis_data)
        protocol = designer.design_protocol(h.hypothesis_id)

        assert protocol.design_type == 'factorial'

    def test_social_science_domain_gets_observational(self, designer, social_science_hypothesis_data):
        """Test that social science domain gets observational design."""
        h = designer.submit_hypothesis(**social_science_hypothesis_data)
        protocol = designer.design_protocol(h.hypothesis_id)

        assert protocol.design_type == 'observational'

    def test_unknown_domain_defaults_to_randomized_controlled(self, designer):
        """Test that unknown domain defaults to randomized controlled."""
        h = designer.submit_hypothesis(
            statement='Test',
            domain='unknown_domain',
            independent_var='A',
            dependent_var='B',
            control_vars=['C'],
            expected_effect_size='medium',
            mechanism='Test',
            prior_evidence='Test'
        )
        protocol = designer.design_protocol(h.hypothesis_id)

        assert protocol.design_type == 'randomized_controlled'


class TestBlindingSelection:
    """Test blinding level selection."""

    def test_medicine_gets_double_blind(self, designer, medicine_hypothesis_data):
        """Test that medicine domain gets double blinding."""
        h = designer.submit_hypothesis(**medicine_hypothesis_data)
        protocol = designer.design_protocol(h.hypothesis_id)

        assert protocol.blind_level == 'double'

    def test_psychology_gets_double_blind(self, designer, sample_hypothesis_data):
        """Test that psychology domain gets double blinding."""
        h = designer.submit_hypothesis(**sample_hypothesis_data)
        protocol = designer.design_protocol(h.hypothesis_id)

        assert protocol.blind_level == 'double'

    def test_biology_gets_single_blind(self, designer, biology_hypothesis_data):
        """Test that biology domain gets single blinding."""
        h = designer.submit_hypothesis(**biology_hypothesis_data)
        protocol = designer.design_protocol(h.hypothesis_id)

        assert protocol.blind_level == 'single'

    def test_physics_gets_no_blinding(self, designer, physics_hypothesis_data):
        """Test that physics domain gets no blinding."""
        h = designer.submit_hypothesis(**physics_hypothesis_data)
        protocol = designer.design_protocol(h.hypothesis_id)

        assert protocol.blind_level == 'none'


class TestRandomizationMethod:
    """Test randomization method selection."""

    def test_randomized_controlled_gets_block_randomization(self, designer, medicine_hypothesis_data):
        """Test randomization method for RCT design."""
        h = designer.submit_hypothesis(**medicine_hypothesis_data)
        protocol = designer.design_protocol(h.hypothesis_id)

        assert protocol.randomization_method == 'Block randomization with stratification'

    def test_factorial_gets_complete_randomization(self, designer, sample_hypothesis_data):
        """Test randomization method for factorial design."""
        h = designer.submit_hypothesis(**sample_hypothesis_data)
        protocol = designer.design_protocol(h.hypothesis_id)

        assert protocol.randomization_method == 'Complete randomization across all factors'

    def test_observational_gets_propensity_matching(self, designer, social_science_hypothesis_data):
        """Test randomization method for observational design."""
        h = designer.submit_hypothesis(**social_science_hypothesis_data)
        protocol = designer.design_protocol(h.hypothesis_id)

        assert protocol.randomization_method == 'Propensity score matching'
