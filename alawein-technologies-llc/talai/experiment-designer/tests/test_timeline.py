"""Tests for project timeline generation."""

import pytest
from experiment_designer.main import Timeline


class TestTimelineGeneration:
    """Test timeline generation."""

    def test_generates_timeline(self, designer, sample_hypothesis):
        """Test that timeline is generated."""
        protocol = designer.design_protocol(sample_hypothesis.hypothesis_id)

        assert len(protocol.timeline) > 0
        assert all(isinstance(phase, Timeline) for phase in protocol.timeline)

    def test_timeline_has_planning_phase(self, designer, sample_hypothesis):
        """Test that timeline includes planning phase."""
        protocol = designer.design_protocol(sample_hypothesis.hypothesis_id)

        phases = [t.phase for t in protocol.timeline]
        assert any('Planning' in phase for phase in phases)

    def test_timeline_has_recruitment_phase(self, designer, sample_hypothesis):
        """Test that timeline includes recruitment phase."""
        protocol = designer.design_protocol(sample_hypothesis.hypothesis_id)

        phases = [t.phase for t in protocol.timeline]
        assert any('Recruitment' in phase for phase in phases)

    def test_timeline_has_data_collection_phase(self, designer, sample_hypothesis):
        """Test that timeline includes data collection phase."""
        protocol = designer.design_protocol(sample_hypothesis.hypothesis_id)

        phases = [t.phase for t in protocol.timeline]
        assert any('Data Collection' in phase for phase in phases)

    def test_timeline_has_analysis_phase(self, designer, sample_hypothesis):
        """Test that timeline includes analysis phase."""
        protocol = designer.design_protocol(sample_hypothesis.hypothesis_id)

        phases = [t.phase for t in protocol.timeline]
        assert any('Analysis' in phase for phase in phases)

    def test_timeline_has_dissemination_phase(self, designer, sample_hypothesis):
        """Test that timeline includes dissemination phase."""
        protocol = designer.design_protocol(sample_hypothesis.hypothesis_id)

        phases = [t.phase for t in protocol.timeline]
        assert any('Dissemination' in phase for phase in phases)

    def test_all_phases_have_duration(self, designer, sample_hypothesis):
        """Test that all phases have duration specified."""
        protocol = designer.design_protocol(sample_hypothesis.hypothesis_id)

        for phase in protocol.timeline:
            assert phase.duration_days > 0

    def test_all_phases_have_description(self, designer, sample_hypothesis):
        """Test that all phases have description."""
        protocol = designer.design_protocol(sample_hypothesis.hypothesis_id)

        for phase in protocol.timeline:
            assert phase.description is not None
            assert len(phase.description) > 0

    def test_all_phases_have_milestones(self, designer, sample_hypothesis):
        """Test that all phases have milestones."""
        protocol = designer.design_protocol(sample_hypothesis.hypothesis_id)

        for phase in protocol.timeline:
            assert isinstance(phase.milestones, list)
            assert len(phase.milestones) > 0

    def test_phases_have_dependencies(self, designer, sample_hypothesis):
        """Test that phases have dependency information."""
        protocol = designer.design_protocol(sample_hypothesis.hypothesis_id)

        for phase in protocol.timeline:
            assert isinstance(phase.dependencies, list)

    def test_total_duration_equals_sum_of_phases(self, designer, sample_hypothesis):
        """Test that total duration equals sum of phase durations."""
        protocol = designer.design_protocol(sample_hypothesis.hypothesis_id)

        phase_sum = sum(t.duration_days for t in protocol.timeline)
        assert protocol.total_duration_days == phase_sum

    def test_recruitment_duration_scales_with_sample_size(self, designer):
        """Test that recruitment duration increases with sample size."""
        h1 = designer.submit_hypothesis(
            statement='Test',
            domain='psychology',
            independent_var='A',
            dependent_var='B',
            control_vars=['C'],
            expected_effect_size='large',  # Small sample
            mechanism='Test',
            prior_evidence='Test'
        )
        protocol1 = designer.design_protocol(h1.hypothesis_id)

        h2 = designer.submit_hypothesis(
            statement='Test',
            domain='psychology',
            independent_var='A',
            dependent_var='B',
            control_vars=['C'],
            expected_effect_size='small',  # Large sample
            mechanism='Test',
            prior_evidence='Test'
        )
        protocol2 = designer.design_protocol(h2.hypothesis_id)

        # Find recruitment phases
        recruitment1 = [t for t in protocol1.timeline if 'Recruitment' in t.phase][0]
        recruitment2 = [t for t in protocol2.timeline if 'Recruitment' in t.phase][0]

        # Larger sample should have longer recruitment
        assert recruitment2.duration_days >= recruitment1.duration_days

    def test_data_collection_duration_scales_with_sample_size(self, designer):
        """Test that data collection duration increases with sample size."""
        h1 = designer.submit_hypothesis(
            statement='Test',
            domain='psychology',
            independent_var='A',
            dependent_var='B',
            control_vars=['C'],
            expected_effect_size='large',  # Small sample
            mechanism='Test',
            prior_evidence='Test'
        )
        protocol1 = designer.design_protocol(h1.hypothesis_id)

        h2 = designer.submit_hypothesis(
            statement='Test',
            domain='psychology',
            independent_var='A',
            dependent_var='B',
            control_vars=['C'],
            expected_effect_size='small',  # Large sample
            mechanism='Test',
            prior_evidence='Test'
        )
        protocol2 = designer.design_protocol(h2.hypothesis_id)

        # Find data collection phases
        data_coll1 = [t for t in protocol1.timeline if 'Data Collection' in t.phase][0]
        data_coll2 = [t for t in protocol2.timeline if 'Data Collection' in t.phase][0]

        # Larger sample should have longer data collection
        assert data_coll2.duration_days >= data_coll1.duration_days
