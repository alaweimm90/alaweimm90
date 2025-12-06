"""
Tests for Supplement class and related functionality.
"""

import pytest
from PEDs_Playbook.main import Supplement, SupplementCategory


class TestSupplementCreation:
    """Test supplement creation and validation."""

    def test_create_valid_supplement(self, sample_protein_supplement):
        """Test creating a valid supplement."""
        assert sample_protein_supplement.name == "Whey Protein"
        assert sample_protein_supplement.category == SupplementCategory.PROTEIN
        assert sample_protein_supplement.dosage == "25g"
        assert sample_protein_supplement.timing == "Post-workout"
        assert sample_protein_supplement.price == 29.99

    def test_supplement_with_benefits(self, sample_protein_supplement):
        """Test supplement benefits are stored correctly."""
        assert len(sample_protein_supplement.benefits) == 2
        assert "Muscle growth" in sample_protein_supplement.benefits

    def test_supplement_with_warnings(self, sample_protein_supplement):
        """Test supplement warnings are stored correctly."""
        assert len(sample_protein_supplement.warnings) == 1
        assert "Contains dairy" in sample_protein_supplement.warnings

    def test_supplement_empty_name_raises_error(self):
        """Test that empty name raises ValueError."""
        with pytest.raises(ValueError, match="name cannot be empty"):
            Supplement(
                name="",
                category=SupplementCategory.PROTEIN,
                dosage="25g",
                timing="Morning",
                price=10.0
            )

    def test_supplement_whitespace_name_raises_error(self):
        """Test that whitespace-only name raises ValueError."""
        with pytest.raises(ValueError, match="name cannot be empty"):
            Supplement(
                name="   ",
                category=SupplementCategory.PROTEIN,
                dosage="25g",
                timing="Morning",
                price=10.0
            )

    def test_supplement_negative_price_raises_error(self):
        """Test that negative price raises ValueError."""
        with pytest.raises(ValueError, match="Price cannot be negative"):
            Supplement(
                name="Test",
                category=SupplementCategory.PROTEIN,
                dosage="25g",
                timing="Morning",
                price=-10.0
            )

    def test_supplement_empty_dosage_raises_error(self):
        """Test that empty dosage raises ValueError."""
        with pytest.raises(ValueError, match="Dosage cannot be empty"):
            Supplement(
                name="Test",
                category=SupplementCategory.PROTEIN,
                dosage="",
                timing="Morning",
                price=10.0
            )


class TestSupplementMethods:
    """Test supplement methods."""

    def test_supplement_to_dict(self, sample_protein_supplement):
        """Test converting supplement to dictionary."""
        result = sample_protein_supplement.to_dict()
        assert isinstance(result, dict)
        assert result['name'] == "Whey Protein"
        assert result['category'] == "protein"
        assert result['price'] == 29.99

    def test_supplement_to_dict_includes_all_fields(self, sample_vitamin_supplement):
        """Test that to_dict includes all fields."""
        result = sample_vitamin_supplement.to_dict()
        assert 'name' in result
        assert 'category' in result
        assert 'dosage' in result
        assert 'timing' in result
        assert 'price' in result
        assert 'benefits' in result
        assert 'warnings' in result
