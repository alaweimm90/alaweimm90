"""
Integration tests for PromptForge Lite
Tests end-to-end workflows and component interactions
"""
import pytest
import json
from pathlib import Path
from promptforge_lite.main import (
    PatternExtractor,
    PromptLibrary,
    extract_from_file,
    extract_from_directory,
    ExtractionResult,
    PromptPattern
)


class TestEndToEndExtraction:
    """Test complete extraction workflows"""

    def test_file_extraction_workflow(self, temp_markdown_file, extractor):
        """Test complete workflow: extract -> filter -> save"""
        # Extract
        patterns = extract_from_file(temp_markdown_file, extractor)
        assert len(patterns) > 0

        # Filter by confidence
        high_conf_patterns = [p for p in patterns if p.confidence >= 0.5]

        # Create result
        result = ExtractionResult(
            total_patterns=len(high_conf_patterns),
            patterns=high_conf_patterns,
            files_processed=1,
            timestamp="2025-11-16T00:00:00"
        )

        assert result.total_patterns == len(high_conf_patterns)
        assert result.files_processed == 1

    def test_directory_scan_workflow(self, temp_multiple_files, extractor):
        """Test complete directory scanning workflow"""
        temp_dir, files = temp_multiple_files

        # Scan directory
        result = extract_from_directory(temp_dir, extractor)

        # Filter by confidence
        filtered = [p for p in result.patterns if p.confidence >= 0.4]

        # Use library to deduplicate
        library = PromptLibrary()
        library.add_patterns(filtered)
        library.deduplicate()

        assert len(library.patterns) > 0
        assert len(library.patterns) <= len(filtered)

    def test_library_management_workflow(self, sample_patterns_list):
        """Test complete library management workflow"""
        library = PromptLibrary()

        # Add patterns
        library.add_patterns(sample_patterns_list)

        # Filter by type
        instruction_patterns = library.filter_by_type("instruction")
        assert len(instruction_patterns) > 0

        # Filter by confidence
        high_conf = library.filter_by_confidence(0.7)

        # Get specific template
        if len(instruction_patterns) > 0:
            template = library.get_template(instruction_patterns[0].pattern_name)
            assert template is not None

        # Rank by reusability
        ranked = library.rank_by_reusability()
        assert len(ranked) == len(sample_patterns_list)


class TestLibraryOperations:
    """Test library operation combinations"""

    def test_filter_chain(self, library, sample_patterns_list):
        """Test chaining multiple filters"""
        library.add_patterns(sample_patterns_list)

        # Chain filters
        coding_patterns = library.filter_by_tag("coding")
        high_conf_coding = [p for p in coding_patterns if p.confidence >= 0.7]

        assert all(p.confidence >= 0.7 for p in high_conf_coding)
        assert all("coding" in p.tags for p in high_conf_coding)

    def test_deduplicate_then_filter(self, library):
        """Test deduplication followed by filtering"""
        # Add duplicates
        pattern1 = PromptPattern(
            pattern_type="instruction",
            pattern_name="pattern1",
            template="Analyze code",
            variables=["var"],
            example="Ex",
            source_file="test1.md",
            line_number=1,
            confidence=0.8,
            tags=["coding"]
        )
        pattern2 = PromptPattern(
            pattern_type="instruction",
            pattern_name="pattern2",
            template="Analyze code",  # Duplicate
            variables=["var"],
            example="Ex",
            source_file="test2.md",
            line_number=1,
            confidence=0.6,
            tags=["coding"]
        )
        pattern3 = PromptPattern(
            pattern_type="instruction",
            pattern_name="pattern3",
            template="Different template",
            variables=[],
            example="Ex",
            source_file="test3.md",
            line_number=1,
            confidence=0.5,
            tags=["other"]
        )

        library.add_patterns([pattern1, pattern2, pattern3])
        library.deduplicate()

        # Filter after deduplication
        high_conf = library.filter_by_confidence(0.7)
        assert len(high_conf) > 0

    def test_rank_after_filtering(self, library, sample_patterns_list):
        """Test ranking after filtering"""
        library.add_patterns(sample_patterns_list)

        # Filter first
        high_conf = library.filter_by_confidence(0.6)

        # Create new library with filtered patterns
        filtered_library = PromptLibrary()
        filtered_library.add_patterns(high_conf)

        # Rank
        ranked = filtered_library.rank_by_reusability()
        assert len(ranked) == len(high_conf)


class TestPatternQuality:
    """Test pattern extraction quality"""

    def test_variable_extraction_quality(self, extractor, variables_text):
        """Test that all variable formats are detected"""
        patterns = extractor.extract_from_text(variables_text, "test.md")

        # Should find patterns with variables
        patterns_with_vars = [p for p in patterns if len(p.variables) > 0]
        assert len(patterns_with_vars) > 0

        # Check variable extraction
        all_variables = set()
        for p in patterns_with_vars:
            all_variables.update(p.variables)

        # Should detect multiple variable formats
        assert len(all_variables) >= 2

    def test_confidence_scoring_consistency(self, extractor):
        """Test that confidence scoring is consistent"""
        # Same template should get same confidence
        template = "You must analyze the code and should provide feedback with {input}"
        confidence1 = extractor._calculate_confidence(template, ["input"])
        confidence2 = extractor._calculate_confidence(template, ["input"])

        assert confidence1 == confidence2

    def test_pattern_name_generation_quality(self, extractor):
        """Test that generated names are meaningful"""
        template = "Analyze the system architecture and provide recommendations"
        name = extractor._generate_name("instruction", template)

        # Name should contain meaningful words
        assert "analyze" in name.lower() or "system" in name.lower()
        assert name.startswith("instruction_")

    def test_tag_extraction_quality(self, extractor, tagged_text):
        """Test that tags are properly extracted"""
        patterns = extractor.extract_from_text(tagged_text, "test.md")

        # Should extract some tags
        all_tags = set()
        for p in patterns:
            all_tags.update(p.tags)

        assert len(all_tags) > 0


class TestEdgeCases:
    """Test edge cases and boundary conditions"""

    def test_very_large_file(self, temp_dir, extractor):
        """Test extraction from very large file"""
        large_file = temp_dir / "large.md"
        content = "\n".join([f"Task: Process item {i}" for i in range(1000)])
        large_file.write_text(content, encoding='utf-8')

        patterns = extract_from_file(large_file, extractor)
        assert isinstance(patterns, list)
        # Should have many patterns
        assert len(patterns) > 100

    def test_special_characters_in_text(self, temp_dir, extractor):
        """Test extraction with special characters"""
        special_file = temp_dir / "special.md"
        content = "Task: Process data with symbols !@#$%^&*() and {variables}"
        special_file.write_text(content, encoding='utf-8')

        patterns = extract_from_file(special_file, extractor)
        assert len(patterns) > 0

    def test_multiline_patterns(self, temp_dir, extractor):
        """Test extraction of multiline patterns"""
        multiline_file = temp_dir / "multiline.md"
        content = """
        Context: This is a multi-line context
        that spans several lines
        and contains important information
        """
        multiline_file.write_text(content, encoding='utf-8')

        patterns = extract_from_file(multiline_file, extractor)
        context_patterns = [p for p in patterns if p.pattern_type == "context"]
        assert len(context_patterns) > 0

    def test_patterns_at_file_boundaries(self, temp_dir, extractor):
        """Test patterns at beginning and end of file"""
        boundary_file = temp_dir / "boundary.md"
        content = "Task: First task\n\n" + "x" * 1000 + "\n\nTask: Last task"
        boundary_file.write_text(content, encoding='utf-8')

        patterns = extract_from_file(boundary_file, extractor)
        task_patterns = [p for p in patterns if p.pattern_type == "instruction"]
        assert len(task_patterns) >= 2

    def test_empty_variables_list(self, extractor):
        """Test patterns with no variables"""
        text = "Task: Simple task with no variables"
        patterns = extractor.extract_from_text(text, "test.md")

        assert len(patterns) > 0
        assert all(isinstance(p.variables, list) for p in patterns)

    def test_multiple_same_pattern_type(self, extractor):
        """Test multiple patterns of the same type"""
        text = """Task: First task
Task: Second task
Task: Third task
"""
        patterns = extractor.extract_from_text(text, "test.md")

        instruction_patterns = [p for p in patterns if p.pattern_type == "instruction"]
        assert len(instruction_patterns) >= 3


class TestDataIntegrity:
    """Test data integrity across operations"""

    def test_pattern_immutability_in_library(self, library, sample_prompt_pattern):
        """Test that patterns don't get modified in library"""
        original_template = sample_prompt_pattern.template
        original_confidence = sample_prompt_pattern.confidence

        library.add_patterns([sample_prompt_pattern])
        library.deduplicate()
        library.rank_by_reusability()

        # Original pattern should not be modified
        assert sample_prompt_pattern.template == original_template
        assert sample_prompt_pattern.confidence == original_confidence

    def test_filter_does_not_modify_library(self, library, sample_patterns_list):
        """Test that filtering doesn't modify original library"""
        library.add_patterns(sample_patterns_list)
        original_count = len(library.patterns)

        # Perform various filters
        library.filter_by_type("instruction")
        library.filter_by_confidence(0.7)
        library.filter_by_tag("coding")

        # Library should still have all patterns
        assert len(library.patterns) == original_count

    def test_extraction_preserves_source_info(self, temp_markdown_file, extractor):
        """Test that source file information is preserved"""
        patterns = extract_from_file(temp_markdown_file, extractor)

        for pattern in patterns:
            assert pattern.source_file == str(temp_markdown_file)
            assert pattern.line_number > 0
            assert isinstance(pattern.line_number, int)
