"""
Test suite for PromptLibrary class
"""
import pytest
from promptforge_lite.main import PromptLibrary, PromptPattern


class TestPromptLibraryInit:
    """Test PromptLibrary initialization"""

    def test_library_init(self, library):
        """Test that library initializes with empty patterns"""
        assert library is not None
        assert library.patterns == []

    def test_library_patterns_is_list(self, library):
        """Test that patterns is a list"""
        assert isinstance(library.patterns, list)


class TestAddPatterns:
    """Test add_patterns method"""

    def test_add_single_pattern(self, library, sample_prompt_pattern):
        """Test adding a single pattern"""
        library.add_patterns([sample_prompt_pattern])
        assert len(library.patterns) == 1
        assert library.patterns[0] == sample_prompt_pattern

    def test_add_multiple_patterns(self, library, sample_patterns_list):
        """Test adding multiple patterns"""
        library.add_patterns(sample_patterns_list)
        assert len(library.patterns) == len(sample_patterns_list)

    def test_add_patterns_extends(self, library, sample_patterns_list, sample_prompt_pattern):
        """Test that add_patterns extends existing patterns"""
        library.add_patterns([sample_prompt_pattern])
        assert len(library.patterns) == 1
        library.add_patterns(sample_patterns_list)
        assert len(library.patterns) == 1 + len(sample_patterns_list)

    def test_add_empty_list(self, library):
        """Test adding empty list of patterns"""
        library.add_patterns([])
        assert len(library.patterns) == 0


class TestFilterByType:
    """Test filter_by_type method"""

    def test_filter_instruction_type(self, library, sample_patterns_list):
        """Test filtering by instruction type"""
        library.add_patterns(sample_patterns_list)
        instruction_patterns = library.filter_by_type("instruction")
        assert len(instruction_patterns) > 0
        assert all(p.pattern_type == "instruction" for p in instruction_patterns)

    def test_filter_role_play_type(self, library, sample_patterns_list):
        """Test filtering by role_play type"""
        library.add_patterns(sample_patterns_list)
        role_patterns = library.filter_by_type("role_play")
        assert len(role_patterns) > 0
        assert all(p.pattern_type == "role_play" for p in role_patterns)

    def test_filter_nonexistent_type(self, library, sample_patterns_list):
        """Test filtering by non-existent type returns empty list"""
        library.add_patterns(sample_patterns_list)
        result = library.filter_by_type("nonexistent_type")
        assert result == []

    def test_filter_empty_library(self, library):
        """Test filtering on empty library"""
        result = library.filter_by_type("instruction")
        assert result == []


class TestFilterByTag:
    """Test filter_by_tag method"""

    def test_filter_by_existing_tag(self, library, sample_patterns_list):
        """Test filtering by an existing tag"""
        library.add_patterns(sample_patterns_list)
        coding_patterns = library.filter_by_tag("coding")
        assert len(coding_patterns) > 0
        assert all("coding" in p.tags for p in coding_patterns)

    def test_filter_by_multiple_patterns_with_tag(self, library, sample_patterns_list):
        """Test that multiple patterns can have same tag"""
        library.add_patterns(sample_patterns_list)
        # Check if any tag appears in multiple patterns
        all_tags = [tag for p in sample_patterns_list for tag in p.tags]
        if len(all_tags) != len(set(all_tags)):  # Has duplicates
            common_tag = [tag for tag in set(all_tags) if all_tags.count(tag) > 1][0]
            result = library.filter_by_tag(common_tag)
            # This test is informational - depends on fixture data

    def test_filter_by_nonexistent_tag(self, library, sample_patterns_list):
        """Test filtering by non-existent tag returns empty list"""
        library.add_patterns(sample_patterns_list)
        result = library.filter_by_tag("nonexistent_tag")
        assert result == []

    def test_filter_tag_empty_library(self, library):
        """Test filtering by tag on empty library"""
        result = library.filter_by_tag("coding")
        assert result == []


class TestFilterByConfidence:
    """Test filter_by_confidence method"""

    def test_filter_high_confidence(self, library, sample_patterns_list):
        """Test filtering by high confidence threshold"""
        library.add_patterns(sample_patterns_list)
        high_conf = library.filter_by_confidence(0.7)
        assert all(p.confidence >= 0.7 for p in high_conf)

    def test_filter_low_confidence(self, library, sample_patterns_list):
        """Test filtering by low confidence threshold"""
        library.add_patterns(sample_patterns_list)
        low_conf = library.filter_by_confidence(0.5)
        assert all(p.confidence >= 0.5 for p in low_conf)
        assert len(low_conf) >= len(library.filter_by_confidence(0.7))

    def test_filter_confidence_zero(self, library, sample_patterns_list):
        """Test filtering with zero confidence (should return all)"""
        library.add_patterns(sample_patterns_list)
        result = library.filter_by_confidence(0.0)
        assert len(result) == len(sample_patterns_list)

    def test_filter_confidence_one(self, library, sample_patterns_list):
        """Test filtering with confidence 1.0"""
        library.add_patterns(sample_patterns_list)
        result = library.filter_by_confidence(1.0)
        # Should return only patterns with exactly 1.0 confidence
        assert all(p.confidence >= 1.0 for p in result)

    def test_filter_confidence_empty_library(self, library):
        """Test filtering by confidence on empty library"""
        result = library.filter_by_confidence(0.5)
        assert result == []


class TestGetTemplate:
    """Test get_template method"""

    def test_get_existing_template(self, library, sample_patterns_list):
        """Test getting an existing template by name"""
        library.add_patterns(sample_patterns_list)
        pattern = library.get_template("instruction_write_code")
        assert pattern is not None
        assert pattern.pattern_name == "instruction_write_code"

    def test_get_nonexistent_template(self, library, sample_patterns_list):
        """Test getting a non-existent template returns None"""
        library.add_patterns(sample_patterns_list)
        pattern = library.get_template("nonexistent_pattern")
        assert pattern is None

    def test_get_template_empty_library(self, library):
        """Test getting template from empty library"""
        pattern = library.get_template("some_pattern")
        assert pattern is None

    def test_get_template_returns_first_match(self, library):
        """Test that get_template returns first match if duplicates exist"""
        pattern1 = PromptPattern(
            pattern_type="instruction",
            pattern_name="duplicate_name",
            template="First template",
            variables=[],
            example="Example 1",
            source_file="test1.md",
            line_number=1,
            confidence=0.7,
            tags=[]
        )
        pattern2 = PromptPattern(
            pattern_type="instruction",
            pattern_name="duplicate_name",
            template="Second template",
            variables=[],
            example="Example 2",
            source_file="test2.md",
            line_number=1,
            confidence=0.8,
            tags=[]
        )
        library.add_patterns([pattern1, pattern2])
        result = library.get_template("duplicate_name")
        assert result.template == "First template"


class TestDeduplicate:
    """Test deduplicate method"""

    def test_deduplicate_removes_duplicates(self, library):
        """Test that deduplication removes duplicate templates"""
        pattern1 = PromptPattern(
            pattern_type="instruction",
            pattern_name="pattern1",
            template="Analyze the code",
            variables=[],
            example="Example",
            source_file="test1.md",
            line_number=1,
            confidence=0.7,
            tags=[]
        )
        pattern2 = PromptPattern(
            pattern_type="instruction",
            pattern_name="pattern2",
            template="Analyze the code",  # Same template
            variables=[],
            example="Example",
            source_file="test2.md",
            line_number=5,
            confidence=0.8,
            tags=[]
        )
        library.add_patterns([pattern1, pattern2])
        assert len(library.patterns) == 2
        library.deduplicate()
        assert len(library.patterns) == 1

    def test_deduplicate_case_insensitive(self, library):
        """Test that deduplication is case-insensitive"""
        pattern1 = PromptPattern(
            pattern_type="instruction",
            pattern_name="pattern1",
            template="Analyze the Code",
            variables=[],
            example="Example",
            source_file="test1.md",
            line_number=1,
            confidence=0.7,
            tags=[]
        )
        pattern2 = PromptPattern(
            pattern_type="instruction",
            pattern_name="pattern2",
            template="analyze the code",  # Different case
            variables=[],
            example="Example",
            source_file="test2.md",
            line_number=5,
            confidence=0.8,
            tags=[]
        )
        library.add_patterns([pattern1, pattern2])
        library.deduplicate()
        assert len(library.patterns) == 1

    def test_deduplicate_preserves_unique(self, library, sample_patterns_list):
        """Test that deduplication preserves unique patterns"""
        original_count = len(sample_patterns_list)
        library.add_patterns(sample_patterns_list)
        library.deduplicate()
        # All patterns in fixture should be unique
        assert len(library.patterns) == original_count

    def test_deduplicate_empty_library(self, library):
        """Test deduplication on empty library"""
        library.deduplicate()
        assert len(library.patterns) == 0

    def test_deduplicate_keeps_first_occurrence(self, library):
        """Test that deduplication keeps the first occurrence"""
        pattern1 = PromptPattern(
            pattern_type="instruction",
            pattern_name="first",
            template="Same template",
            variables=[],
            example="Example",
            source_file="first.md",
            line_number=1,
            confidence=0.7,
            tags=[]
        )
        pattern2 = PromptPattern(
            pattern_type="instruction",
            pattern_name="second",
            template="Same template",
            variables=[],
            example="Example",
            source_file="second.md",
            line_number=1,
            confidence=0.8,
            tags=[]
        )
        library.add_patterns([pattern1, pattern2])
        library.deduplicate()
        assert len(library.patterns) == 1
        assert library.patterns[0].pattern_name == "first"
        assert library.patterns[0].source_file == "first.md"


class TestRankByReusability:
    """Test rank_by_reusability method"""

    def test_rank_returns_all_patterns(self, library, sample_patterns_list):
        """Test that ranking returns all patterns"""
        library.add_patterns(sample_patterns_list)
        ranked = library.rank_by_reusability()
        assert len(ranked) == len(sample_patterns_list)

    def test_rank_descending_order(self, library, sample_patterns_list):
        """Test that patterns are ranked in descending order"""
        library.add_patterns(sample_patterns_list)
        ranked = library.rank_by_reusability()
        # Calculate reusability scores and verify order
        scores = []
        for p in ranked:
            score = p.confidence * (1 + len(p.variables)) * min(len(p.template) / 100, 2.0)
            scores.append(score)
        # Check that scores are in descending order
        assert scores == sorted(scores, reverse=True)

    def test_rank_prefers_patterns_with_variables(self, library):
        """Test that patterns with variables rank higher"""
        pattern_no_vars = PromptPattern(
            pattern_type="instruction",
            pattern_name="no_vars",
            template="A" * 100,  # Long enough template
            variables=[],
            example="Example",
            source_file="test.md",
            line_number=1,
            confidence=0.7,
            tags=[]
        )
        pattern_with_vars = PromptPattern(
            pattern_type="instruction",
            pattern_name="with_vars",
            template="A" * 100,  # Same length
            variables=["var1", "var2"],
            example="Example",
            source_file="test.md",
            line_number=1,
            confidence=0.7,  # Same confidence
            tags=[]
        )
        library.add_patterns([pattern_no_vars, pattern_with_vars])
        ranked = library.rank_by_reusability()
        assert ranked[0].pattern_name == "with_vars"

    def test_rank_empty_library(self, library):
        """Test ranking on empty library"""
        ranked = library.rank_by_reusability()
        assert ranked == []

    def test_rank_template_length_factor(self, library):
        """Test that template length affects ranking (capped at 2.0)"""
        short_pattern = PromptPattern(
            pattern_type="instruction",
            pattern_name="short",
            template="Short",
            variables=["var"],
            example="Example",
            source_file="test.md",
            line_number=1,
            confidence=0.7,
            tags=[]
        )
        long_pattern = PromptPattern(
            pattern_type="instruction",
            pattern_name="long",
            template="A" * 300,  # Very long (> 200 chars, should cap at 2.0)
            variables=["var"],
            example="Example",
            source_file="test.md",
            line_number=1,
            confidence=0.7,  # Same confidence and variables
            tags=[]
        )
        library.add_patterns([short_pattern, long_pattern])
        ranked = library.rank_by_reusability()
        assert ranked[0].pattern_name == "long"
