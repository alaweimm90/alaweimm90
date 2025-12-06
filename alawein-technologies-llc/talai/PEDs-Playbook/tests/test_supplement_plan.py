"""
Tests for SupplementPlan class and related functionality.
"""

import pytest
from PEDs_Playbook.main import SupplementPlan, Supplement, SupplementCategory


class TestSupplementPlanCreation:
    """Test supplement plan creation."""

    def test_create_empty_plan(self, empty_supplement_plan):
        """Test creating an empty supplement plan."""
        assert empty_supplement_plan.plan_id == "plan001"
        assert empty_supplement_plan.user_id == "user123"
        assert len(empty_supplement_plan.supplements) == 0
        assert empty_supplement_plan.total_cost == 0.0

    def test_plan_has_created_at_timestamp(self, empty_supplement_plan):
        """Test that plan has a created_at timestamp."""
        assert empty_supplement_plan.created_at is not None
        assert isinstance(empty_supplement_plan.created_at, str)


class TestSupplementPlanMethods:
    """Test supplement plan methods."""

    def test_add_supplement_to_plan(self, empty_supplement_plan, sample_protein_supplement):
        """Test adding a supplement to the plan."""
        empty_supplement_plan.add_supplement(sample_protein_supplement)
        assert len(empty_supplement_plan.supplements) == 1
        assert empty_supplement_plan.total_cost == 29.99

    def test_add_multiple_supplements_updates_cost(self, empty_supplement_plan,
                                                   sample_protein_supplement,
                                                   sample_vitamin_supplement):
        """Test that adding multiple supplements updates total cost."""
        empty_supplement_plan.add_supplement(sample_protein_supplement)
        empty_supplement_plan.add_supplement(sample_vitamin_supplement)
        expected_cost = 29.99 + 12.99
        assert empty_supplement_plan.total_cost == expected_cost

    def test_add_invalid_type_raises_error(self, empty_supplement_plan):
        """Test that adding non-Supplement raises TypeError."""
        with pytest.raises(TypeError, match="Must be a Supplement instance"):
            empty_supplement_plan.add_supplement("not a supplement")

    def test_remove_supplement_by_name(self, populated_supplement_plan):
        """Test removing a supplement by name."""
        result = populated_supplement_plan.remove_supplement("Whey Protein")
        assert result is True
        assert len(populated_supplement_plan.supplements) == 1

    def test_remove_updates_total_cost(self, populated_supplement_plan):
        """Test that removing a supplement updates total cost."""
        initial_cost = populated_supplement_plan.total_cost
        populated_supplement_plan.remove_supplement("Whey Protein")
        assert populated_supplement_plan.total_cost < initial_cost
        assert populated_supplement_plan.total_cost == 12.99

    def test_remove_nonexistent_supplement_returns_false(self, populated_supplement_plan):
        """Test that removing non-existent supplement returns False."""
        result = populated_supplement_plan.remove_supplement("Nonexistent")
        assert result is False
        assert len(populated_supplement_plan.supplements) == 2

    def test_get_supplements_by_category(self, populated_supplement_plan):
        """Test getting supplements by category."""
        proteins = populated_supplement_plan.get_supplements_by_category(
            SupplementCategory.PROTEIN
        )
        assert len(proteins) == 1
        assert proteins[0].name == "Whey Protein"

    def test_get_supplements_by_category_empty_result(self, populated_supplement_plan):
        """Test getting supplements by category with no matches."""
        energy = populated_supplement_plan.get_supplements_by_category(
            SupplementCategory.ENERGY
        )
        assert len(energy) == 0

    def test_plan_to_dict(self, populated_supplement_plan):
        """Test converting plan to dictionary."""
        result = populated_supplement_plan.to_dict()
        assert isinstance(result, dict)
        assert result['plan_id'] == "plan002"
        assert result['user_id'] == "user123"
        assert len(result['supplements']) == 2
        assert 'total_cost' in result
        assert 'created_at' in result
