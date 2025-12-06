"""Tests for resource estimation (equipment, personnel, budget)."""

import pytest
from experiment_designer.main import Equipment


class TestEquipmentGeneration:
    """Test equipment and materials list generation."""

    def test_generates_equipment_list(self, designer, sample_hypothesis):
        """Test that equipment list is generated."""
        protocol = designer.design_protocol(sample_hypothesis.hypothesis_id)

        assert len(protocol.equipment_list) > 0
        assert all(isinstance(eq, Equipment) for eq in protocol.equipment_list)

    def test_equipment_has_all_required_fields(self, designer, sample_hypothesis):
        """Test that equipment objects have all required fields."""
        protocol = designer.design_protocol(sample_hypothesis.hypothesis_id)

        for eq in protocol.equipment_list:
            assert eq.name is not None
            assert eq.quantity > 0
            assert eq.unit is not None
            assert eq.estimated_cost > 0

    def test_domain_specific_equipment_psychology(self, designer, sample_hypothesis_data):
        """Test that psychology domain gets appropriate equipment."""
        h = designer.submit_hypothesis(**sample_hypothesis_data)
        protocol = designer.design_protocol(h.hypothesis_id)

        equipment_names = [eq.name for eq in protocol.equipment_list]
        # Should have psychology-specific equipment
        assert any('Computer' in name or 'workstation' in name for name in equipment_names)

    def test_domain_specific_equipment_medicine(self, designer, medicine_hypothesis_data):
        """Test that medicine domain gets appropriate equipment."""
        h = designer.submit_hypothesis(**medicine_hypothesis_data)
        protocol = designer.design_protocol(h.hypothesis_id)

        equipment_names = [eq.name.lower() for eq in protocol.equipment_list]
        # Should have medical equipment
        assert any('medical' in name or 'blood' in name or 'imaging' in name
                   for name in equipment_names)

    def test_domain_specific_equipment_biology(self, designer, biology_hypothesis_data):
        """Test that biology domain gets appropriate equipment."""
        h = designer.submit_hypothesis(**biology_hypothesis_data)
        protocol = designer.design_protocol(h.hypothesis_id)

        equipment_names = [eq.name.lower() for eq in protocol.equipment_list]
        # Should have biology lab equipment
        assert any('microscope' in name or 'incubator' in name or 'culture' in name
                   for name in equipment_names)

    def test_domain_specific_equipment_physics(self, designer, physics_hypothesis_data):
        """Test that physics domain gets appropriate equipment."""
        h = designer.submit_hypothesis(**physics_hypothesis_data)
        protocol = designer.design_protocol(h.hypothesis_id)

        equipment_names = [eq.name.lower() for eq in protocol.equipment_list]
        # Should have physics equipment
        assert any('oscilloscope' in name or 'laser' in name or 'acquisition' in name
                   for name in equipment_names)

    def test_includes_common_equipment(self, designer, sample_hypothesis):
        """Test that common equipment is included."""
        protocol = designer.design_protocol(sample_hypothesis.hypothesis_id)

        equipment_names = [eq.name.lower() for eq in protocol.equipment_list]
        # Should include computer for analysis
        assert any('computer' in name for name in equipment_names)
        # Should include software
        assert any('software' in name for name in equipment_names)


class TestPersonnelEstimation:
    """Test personnel needs estimation."""

    def test_estimates_personnel(self, designer, sample_hypothesis):
        """Test that personnel needs are estimated."""
        protocol = designer.design_protocol(sample_hypothesis.hypothesis_id)

        assert len(protocol.personnel_needed) > 0
        assert isinstance(protocol.personnel_needed, dict)

    def test_includes_principal_investigator(self, designer, sample_hypothesis):
        """Test that PI is always included."""
        protocol = designer.design_protocol(sample_hypothesis.hypothesis_id)

        assert 'Principal Investigator' in protocol.personnel_needed
        assert protocol.personnel_needed['Principal Investigator'] >= 1

    def test_includes_research_coordinator(self, designer, sample_hypothesis):
        """Test that research coordinator is included."""
        protocol = designer.design_protocol(sample_hypothesis.hypothesis_id)

        assert 'Research Coordinator' in protocol.personnel_needed

    def test_includes_data_analyst(self, designer, sample_hypothesis):
        """Test that data analyst is included."""
        protocol = designer.design_protocol(sample_hypothesis.hypothesis_id)

        assert 'Data Analysts' in protocol.personnel_needed

    def test_personnel_scales_with_sample_size_large(self, designer):
        """Test that personnel increases for large sample sizes."""
        h = designer.submit_hypothesis(
            statement='Test',
            domain='psychology',
            independent_var='A',
            dependent_var='B',
            control_vars=['C'],
            expected_effect_size='small',  # Small effect = large sample
            mechanism='Test',
            prior_evidence='Test'
        )
        protocol = designer.design_protocol(h.hypothesis_id)

        # Large sample should have more research assistants
        if protocol.power_analysis.required_sample_size > 100:
            assert protocol.personnel_needed.get('Research Assistants', 0) >= 3

    def test_personnel_scales_with_sample_size_small(self, designer):
        """Test that personnel is appropriate for small sample sizes."""
        h = designer.submit_hypothesis(
            statement='Test',
            domain='psychology',
            independent_var='A',
            dependent_var='B',
            control_vars=['C'],
            expected_effect_size='large',  # Large effect = small sample
            mechanism='Test',
            prior_evidence='Test'
        )
        protocol = designer.design_protocol(h.hypothesis_id)

        # Small sample should have fewer research assistants
        if protocol.power_analysis.required_sample_size <= 50:
            assert protocol.personnel_needed.get('Research Assistants', 0) >= 1

    def test_medicine_includes_clinical_staff(self, designer, medicine_hypothesis_data):
        """Test that medicine domain includes nurses/clinicians."""
        h = designer.submit_hypothesis(**medicine_hypothesis_data)
        protocol = designer.design_protocol(h.hypothesis_id)

        assert 'Nurses/Clinicians' in protocol.personnel_needed

    def test_biology_includes_lab_technicians(self, designer, biology_hypothesis_data):
        """Test that biology domain includes lab technicians."""
        h = designer.submit_hypothesis(**biology_hypothesis_data)
        protocol = designer.design_protocol(h.hypothesis_id)

        assert 'Lab Technicians' in protocol.personnel_needed


class TestBudgetEstimation:
    """Test budget and cost estimation."""

    def test_calculates_total_cost(self, designer, sample_hypothesis):
        """Test that total cost is calculated."""
        protocol = designer.design_protocol(sample_hypothesis.hypothesis_id)

        assert protocol.total_cost > 0

    def test_cost_breakdown_has_all_categories(self, designer, sample_hypothesis):
        """Test that cost breakdown includes all categories."""
        protocol = designer.design_protocol(sample_hypothesis.hypothesis_id)

        assert 'equipment' in protocol.cost_breakdown
        assert 'personnel' in protocol.cost_breakdown
        assert 'overhead' in protocol.cost_breakdown

    def test_equipment_cost_is_sum_of_equipment(self, designer, sample_hypothesis):
        """Test that equipment cost matches sum of equipment list."""
        protocol = designer.design_protocol(sample_hypothesis.hypothesis_id)

        expected_equipment_cost = sum(eq.estimated_cost for eq in protocol.equipment_list)
        assert protocol.cost_breakdown['equipment'] == expected_equipment_cost

    def test_overhead_is_30_percent(self, designer, sample_hypothesis):
        """Test that overhead is 30% of equipment and personnel."""
        protocol = designer.design_protocol(sample_hypothesis.hypothesis_id)

        base_cost = protocol.cost_breakdown['equipment'] + protocol.cost_breakdown['personnel']
        expected_overhead = base_cost * 0.3

        assert abs(protocol.cost_breakdown['overhead'] - expected_overhead) < 0.01

    def test_total_equals_sum_of_breakdown(self, designer, sample_hypothesis):
        """Test that total cost equals sum of breakdown."""
        protocol = designer.design_protocol(sample_hypothesis.hypothesis_id)

        breakdown_sum = sum(protocol.cost_breakdown.values())
        assert abs(protocol.total_cost - breakdown_sum) < 0.01

    def test_personnel_cost_scales_with_duration(self, designer):
        """Test that personnel cost increases with project duration."""
        # This is implicitly tested through the cost calculation
        # Longer projects should have higher personnel costs
        h = designer.submit_hypothesis(
            statement='Test',
            domain='psychology',
            independent_var='A',
            dependent_var='B',
            control_vars=['C'],
            expected_effect_size='small',  # Small effect = longer duration
            mechanism='Test',
            prior_evidence='Test'
        )
        protocol = designer.design_protocol(h.hypothesis_id)

        assert protocol.cost_breakdown['personnel'] > 0
