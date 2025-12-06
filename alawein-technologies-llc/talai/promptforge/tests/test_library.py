"""Tests for prompt library management"""

import pytest
from promptforge.library import PromptLibrary
from promptforge.models import PromptPattern


class TestPromptLibrary:
    """Test PromptLibrary class"""

    def test_create_library(self, library):
        """Test creating an empty PromptLibrary"""
        assert library is not None
        assert len(library.patterns) == 0

    def test_add_pattern(self, library, sample_pattern):
        """Test adding a single pattern to library"""
        library.add_pattern(sample_pattern)
        assert len(library.patterns) == 1
        assert library.patterns[0] == sample_pattern

    def test_add_patterns(self, library, sample_patterns):
        """Test adding multiple patterns to library"""
        library.add_patterns(sample_patterns)
        assert len(library.patterns) == len(sample_patterns)

    def test_filter_by_type(self, populated_library):
        """Test filtering patterns by type"""
        instruction_patterns = populated_library.filter_by_type("instruction")
        assert len(instruction_patterns) == 2
        assert all(p.pattern_type == "instruction" for p in instruction_patterns)

    def test_filter_by_type_no_match(self, populated_library):
        """Test filtering by type with no matches"""
        patterns = populated_library.filter_by_type("nonexistent_type")
        assert len(patterns) == 0

    def test_filter_by_tag(self, populated_library):
        """Test filtering patterns by tag"""
        coding_patterns = populated_library.filter_by_tag("coding")
        assert len(coding_patterns) >= 2
        assert all("coding" in p.tags for p in coding_patterns)

    def test_filter_by_tag_no_match(self, populated_library):
        """Test filtering by tag with no matches"""
        patterns = populated_library.filter_by_tag("nonexistent_tag")
        assert len(patterns) == 0

    def test_filter_by_confidence(self, populated_library):
        """Test filtering patterns by minimum confidence"""
        high_confidence = populated_library.filter_by_confidence(0.8)
        assert all(p.confidence >= 0.8 for p in high_confidence)

    def test_filter_by_confidence_threshold(self, populated_library):
        """Test filtering with different confidence thresholds"""
        all_patterns = populated_library.filter_by_confidence(0.0)
        some_patterns = populated_library.filter_by_confidence(0.7)
        assert len(all_patterns) >= len(some_patterns)

    def test_get_template_by_name(self, populated_library):
        """Test retrieving a pattern by name"""
        pattern = populated_library.get_template("instruction_task_one")
        assert pattern is not None
        assert pattern.pattern_name == "instruction_task_one"

    def test_get_template_not_found(self, populated_library):
        """Test retrieving a non-existent pattern"""
        pattern = populated_library.get_template("nonexistent_pattern")
        assert pattern is None

    def test_deduplicate(self, library):
        """Test removing duplicate patterns"""
        # Add duplicate patterns
        pattern1 = PromptPattern(
            pattern_type="instruction",
            pattern_name="test_1",
            template="Same template",
            variables=[],
            example="example",
            source_file="test.md",
            line_number=1,
            confidence=0.7,
            tags=[]
        )
        pattern2 = PromptPattern(
            pattern_type="instruction",
            pattern_name="test_2",
            template="Same template",  # Duplicate
            variables=[],
            example="example",
            source_file="test.md",
            line_number=5,
            confidence=0.8,
            tags=[]
        )
        pattern3 = PromptPattern(
            pattern_type="instruction",
            pattern_name="test_3",
            template="Different template",
            variables=[],
            example="example",
            source_file="test.md",
            line_number=10,
            confidence=0.7,
            tags=[]
        )

        library.add_patterns([pattern1, pattern2, pattern3])
        assert len(library.patterns) == 3

        library.deduplicate()
        assert len(library.patterns) == 2  # One duplicate removed

    def test_deduplicate_case_insensitive(self, library):
        """Test that deduplication is case-insensitive"""
        pattern1 = PromptPattern(
            pattern_type="instruction",
            pattern_name="test_1",
            template="Template Text",
            variables=[],
            example="example",
            source_file="test.md",
            line_number=1,
            confidence=0.7,
            tags=[]
        )
        pattern2 = PromptPattern(
            pattern_type="instruction",
            pattern_name="test_2",
            template="template text",  # Same but lowercase
            variables=[],
            example="example",
            source_file="test.md",
            line_number=5,
            confidence=0.8,
            tags=[]
        )

        library.add_patterns([pattern1, pattern2])
        library.deduplicate()
        assert len(library.patterns) == 1

    def test_rank_by_reusability(self, populated_library):
        """Test ranking patterns by reusability"""
        ranked = populated_library.rank_by_reusability()
        assert len(ranked) == len(populated_library.patterns)
        # Should be sorted in descending order by reusability
        for i in range(len(ranked) - 1):
            score1 = ranked[i].confidence * (1 + len(ranked[i].variables))
            score2 = ranked[i + 1].confidence * (1 + len(ranked[i + 1].variables))
            assert score1 >= score2

    def test_rank_by_reusability_empty(self, library):
        """Test ranking with empty library"""
        ranked = library.rank_by_reusability()
        assert len(ranked) == 0

    def test_get_pattern_types(self, populated_library):
        """Test getting list of unique pattern types"""
        types = populated_library.get_pattern_types()
        assert "instruction" in types
        assert "role_play" in types
        assert "format" in types
        assert len(types) == 3  # Should be unique

    def test_get_pattern_types_empty(self, library):
        """Test getting pattern types from empty library"""
        types = library.get_pattern_types()
        assert len(types) == 0

    def test_get_all_tags(self, populated_library):
        """Test getting all unique tags"""
        tags = populated_library.get_all_tags()
        assert "coding" in tags
        assert "role" in tags
        assert "output" in tags

    def test_get_all_tags_empty(self, library):
        """Test getting tags from empty library"""
        tags = library.get_all_tags()
        assert len(tags) == 0

    def test_count(self, populated_library):
        """Test counting patterns in library"""
        assert populated_library.count() == 4

    def test_count_empty(self, library):
        """Test counting patterns in empty library"""
        assert library.count() == 0

    def test_clear(self, populated_library):
        """Test clearing all patterns from library"""
        assert populated_library.count() > 0
        populated_library.clear()
        assert populated_library.count() == 0
        assert len(populated_library.patterns) == 0

    def test_multiple_operations(self, library, sample_patterns):
        """Test performing multiple operations in sequence"""
        # Add patterns
        library.add_patterns(sample_patterns)
        assert library.count() == 4

        # Filter by type
        instruction_patterns = library.filter_by_type("instruction")
        assert len(instruction_patterns) == 2

        # Deduplicate
        library.deduplicate()
        assert library.count() == 3  # One duplicate removed

        # Filter by confidence
        high_confidence = library.filter_by_confidence(0.8)
        assert len(high_confidence) >= 1

        # Get pattern types
        types = library.get_pattern_types()
        assert len(types) > 0

    def test_reusability_scoring_with_variables(self, library):
        """Test that patterns with more variables rank higher"""
        pattern_no_vars = PromptPattern(
            pattern_type="instruction",
            pattern_name="test_no_vars",
            template="Simple template",
            variables=[],
            example="example",
            source_file="test.md",
            line_number=1,
            confidence=0.8,
            tags=[]
        )
        pattern_with_vars = PromptPattern(
            pattern_type="instruction",
            pattern_name="test_with_vars",
            template="Template with {var1} and {var2}",
            variables=["var1", "var2"],
            example="example",
            source_file="test.md",
            line_number=5,
            confidence=0.8,
            tags=[]
        )

        library.add_patterns([pattern_no_vars, pattern_with_vars])
        ranked = library.rank_by_reusability()

        # Pattern with variables should rank higher (same confidence)
        assert ranked[0].pattern_name == "test_with_vars"
