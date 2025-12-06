"""
Tests for SupplementDatabase class and related functionality.
"""

import pytest
from PEDs_Playbook.main import SupplementDatabase, Supplement, SupplementCategory


class TestDatabaseCreation:
    """Test database creation."""

    def test_create_empty_database(self, empty_database):
        """Test creating an empty database."""
        assert empty_database.count() == 0
        assert len(empty_database.get_all_supplements()) == 0

    def test_create_populated_database(self, supplement_database):
        """Test creating a populated database."""
        assert supplement_database.count() == 5


class TestDatabaseOperations:
    """Test database CRUD operations."""

    def test_add_supplement_to_database(self, empty_database, sample_protein_supplement):
        """Test adding a supplement to the database."""
        empty_database.add_supplement(sample_protein_supplement)
        assert empty_database.count() == 1

    def test_add_invalid_type_raises_error(self, empty_database):
        """Test that adding non-Supplement raises TypeError."""
        with pytest.raises(TypeError, match="Must be a Supplement instance"):
            empty_database.add_supplement("not a supplement")

    def test_get_supplement_by_name(self, supplement_database):
        """Test retrieving a supplement by name."""
        supplement = supplement_database.get_supplement("Whey Protein")
        assert supplement is not None
        assert supplement.name == "Whey Protein"

    def test_get_nonexistent_supplement_returns_none(self, supplement_database):
        """Test that getting non-existent supplement returns None."""
        supplement = supplement_database.get_supplement("Nonexistent")
        assert supplement is None

    def test_get_all_supplements(self, supplement_database):
        """Test getting all supplements."""
        all_supplements = supplement_database.get_all_supplements()
        assert len(all_supplements) == 5

    def test_add_duplicate_name_overwrites(self, supplement_database):
        """Test that adding supplement with same name overwrites."""
        new_protein = Supplement(
            name="Whey Protein",
            category=SupplementCategory.PROTEIN,
            dosage="30g",
            timing="Morning",
            price=35.00
        )
        supplement_database.add_supplement(new_protein)
        assert supplement_database.count() == 5  # Still 5
        retrieved = supplement_database.get_supplement("Whey Protein")
        assert retrieved.dosage == "30g"  # Updated


class TestDatabaseSearch:
    """Test database search functionality."""

    def test_search_by_category(self, supplement_database):
        """Test searching supplements by category."""
        proteins = supplement_database.search_by_category(SupplementCategory.PROTEIN)
        assert len(proteins) == 1
        assert proteins[0].category == SupplementCategory.PROTEIN

    def test_search_by_category_no_results(self, supplement_database):
        """Test searching for category with no supplements."""
        minerals = supplement_database.search_by_category(SupplementCategory.MINERALS)
        assert len(minerals) == 0

    def test_search_by_price_range(self, supplement_database):
        """Test searching supplements by price range."""
        affordable = supplement_database.search_by_price_range(0, 20)
        assert len(affordable) > 0
        for supplement in affordable:
            assert supplement.price <= 20

    def test_search_by_price_range_exact_bounds(self, supplement_database):
        """Test price range search includes exact bounds."""
        results = supplement_database.search_by_price_range(12.99, 29.99)
        prices = [s.price for s in results]
        assert 12.99 in prices
        assert 29.99 in prices

    def test_search_price_range_negative_min_raises_error(self, supplement_database):
        """Test that negative min price raises ValueError."""
        with pytest.raises(ValueError, match="Prices cannot be negative"):
            supplement_database.search_by_price_range(-10, 50)

    def test_search_price_range_negative_max_raises_error(self, supplement_database):
        """Test that negative max price raises ValueError."""
        with pytest.raises(ValueError, match="Prices cannot be negative"):
            supplement_database.search_by_price_range(0, -50)

    def test_search_price_range_min_exceeds_max_raises_error(self, supplement_database):
        """Test that min > max raises ValueError."""
        with pytest.raises(ValueError, match="Min price cannot exceed max price"):
            supplement_database.search_by_price_range(100, 50)
