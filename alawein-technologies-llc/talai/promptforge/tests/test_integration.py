"""Integration tests for PromptForge"""

import pytest
import json
import tempfile
from pathlib import Path
from promptforge.extractor import PatternExtractor
from promptforge.library import PromptLibrary
from promptforge.models import ExtractionResult
from promptforge.main import extract_from_file, extract_from_directory


class TestFileExtraction:
    """Test file extraction integration"""

    def test_extract_from_markdown_file(self, sample_markdown_file, extractor):
        """Test extracting patterns from a markdown file"""
        patterns = extract_from_file(sample_markdown_file, extractor)
        assert len(patterns) > 0
        assert any(p.source_file == str(sample_markdown_file) for p in patterns)

    def test_extract_from_nonexistent_file(self, extractor):
        """Test extracting from a file that doesn't exist"""
        fake_path = Path("/nonexistent/file.md")
        patterns = extract_from_file(fake_path, extractor)
        assert len(patterns) == 0

    def test_extract_patterns_with_line_numbers(self, sample_markdown_file, extractor):
        """Test that extracted patterns have correct line numbers"""
        patterns = extract_from_file(sample_markdown_file, extractor)
        for pattern in patterns:
            assert pattern.line_number > 0


class TestDirectoryExtraction:
    """Test directory extraction integration"""

    def test_extract_from_directory(self, sample_directory_with_files, extractor):
        """Test extracting patterns from a directory"""
        result = extract_from_directory(sample_directory_with_files, extractor)
        assert result.files_processed >= 3
        assert result.total_patterns > 0
        assert len(result.patterns) > 0
        assert result.timestamp is not None

    def test_extract_from_directory_recursive(self, sample_directory_with_files, extractor):
        """Test that directory extraction is recursive"""
        result = extract_from_directory(sample_directory_with_files, extractor)
        # Should find files in subdirectory too
        assert result.files_processed >= 4

    def test_extract_from_empty_directory(self, tmp_path, extractor):
        """Test extracting from an empty directory"""
        result = extract_from_directory(tmp_path, extractor)
        assert result.files_processed == 0
        assert result.total_patterns == 0


class TestEndToEndWorkflow:
    """Test complete end-to-end workflows"""

    def test_extract_filter_save_workflow(self, sample_directory_with_files, extractor, tmp_path):
        """Test complete workflow: extract, filter, save"""
        # Extract patterns
        result = extract_from_directory(sample_directory_with_files, extractor)
        initial_count = result.total_patterns

        # Filter by confidence
        result.patterns = [p for p in result.patterns if p.confidence >= 0.6]
        result.total_patterns = len(result.patterns)

        # Should have filtered some out
        assert result.total_patterns <= initial_count

        # Save to JSON
        output_file = tmp_path / "output.json"
        with open(output_file, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)

        # Verify file was created and can be loaded
        assert output_file.exists()
        with open(output_file, 'r') as f:
            loaded_data = json.load(f)

        assert loaded_data['total_patterns'] == result.total_patterns
        assert len(loaded_data['patterns']) == result.total_patterns

    def test_extract_deduplicate_rank_workflow(self, sample_directory_with_files, extractor):
        """Test workflow: extract, deduplicate, rank"""
        # Extract patterns
        result = extract_from_directory(sample_directory_with_files, extractor)

        # Load into library
        library = PromptLibrary()
        library.add_patterns(result.patterns)

        initial_count = library.count()

        # Deduplicate
        library.deduplicate()
        deduped_count = library.count()

        assert deduped_count <= initial_count

        # Rank by reusability
        ranked = library.rank_by_reusability()
        assert len(ranked) == deduped_count

        # Top patterns should have higher scores
        if len(ranked) > 1:
            top_pattern = ranked[0]
            last_pattern = ranked[-1]
            top_score = top_pattern.confidence * (1 + len(top_pattern.variables))
            last_score = last_pattern.confidence * (1 + len(last_pattern.variables))
            assert top_score >= last_score

    def test_complex_text_extraction_workflow(self, sample_text_complex, extractor):
        """Test extraction from complex text with multiple pattern types"""
        patterns = extractor.extract_from_text(sample_text_complex, "complex.md")

        # Should extract multiple pattern types
        pattern_types = set(p.pattern_type for p in patterns)
        assert len(pattern_types) >= 4

        # Should extract variables
        patterns_with_vars = [p for p in patterns if len(p.variables) > 0]
        assert len(patterns_with_vars) > 0

        # Should extract tags
        all_tags = set()
        for p in patterns:
            all_tags.update(p.tags)
        assert len(all_tags) > 0

    def test_filter_by_multiple_criteria(self, sample_text_complex, extractor):
        """Test filtering patterns by multiple criteria"""
        patterns = extractor.extract_from_text(sample_text_complex, "complex.md")

        library = PromptLibrary()
        library.add_patterns(patterns)

        # Filter by type
        instruction_patterns = library.filter_by_type("instruction")

        # Further filter by confidence
        high_confidence_instructions = [p for p in instruction_patterns if p.confidence >= 0.6]

        # All should be instructions with high confidence
        assert all(p.pattern_type == "instruction" for p in high_confidence_instructions)
        assert all(p.confidence >= 0.6 for p in high_confidence_instructions)

    def test_save_and_load_patterns(self, sample_patterns, tmp_path):
        """Test saving patterns to file and loading them back"""
        # Create extraction result
        result = ExtractionResult(
            total_patterns=len(sample_patterns),
            patterns=sample_patterns,
            files_processed=1,
            timestamp="2024-01-01T12:00:00"
        )

        # Save to JSON
        output_file = tmp_path / "patterns.json"
        with open(output_file, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)

        # Load back
        with open(output_file, 'r') as f:
            loaded_data = json.load(f)

        # Reconstruct result
        loaded_result = ExtractionResult.from_dict(loaded_data)

        # Verify data integrity
        assert loaded_result.total_patterns == result.total_patterns
        assert loaded_result.files_processed == result.files_processed
        assert len(loaded_result.patterns) == len(result.patterns)

        # Load into library
        library = PromptLibrary()
        library.add_patterns(loaded_result.patterns)

        assert library.count() == len(sample_patterns)


class TestErrorHandling:
    """Test error handling and edge cases"""

    def test_extract_from_binary_file(self, tmp_path, extractor):
        """Test extracting from a binary file"""
        binary_file = tmp_path / "binary.dat"
        binary_file.write_bytes(b'\x00\x01\x02\x03\x04')

        patterns = extract_from_file(binary_file, extractor)
        # Should handle gracefully and return empty list
        assert isinstance(patterns, list)

    def test_extract_from_corrupted_json(self, tmp_path):
        """Test loading corrupted JSON file"""
        json_file = tmp_path / "corrupted.json"
        json_file.write_text("{ invalid json content }")

        library = PromptLibrary()
        # Should raise JSONDecodeError
        with pytest.raises(json.JSONDecodeError):
            with open(json_file, 'r') as f:
                data = json.load(f)

    def test_empty_library_operations(self, library):
        """Test operations on empty library"""
        assert library.count() == 0
        assert library.filter_by_type("any") == []
        assert library.filter_by_tag("any") == []
        assert library.filter_by_confidence(0.0) == []
        assert library.get_template("any") is None
        assert library.rank_by_reusability() == []
        assert library.get_pattern_types() == []
        assert library.get_all_tags() == []

    def test_very_large_text_extraction(self, extractor):
        """Test extraction from very large text"""
        # Create a large text with many patterns
        large_text = "\n".join([f"Task: Process item {i}" for i in range(1000)])
        patterns = extractor.extract_from_text(large_text, "large.md")

        # Should extract all or most patterns
        assert len(patterns) > 900

    def test_unicode_text_extraction(self, extractor):
        """Test extraction from text with unicode characters"""
        unicode_text = """
        Task: Process données with ñoño and 中文
        You are a développeur working on プロジェクト
        Format: UTF-8 encoded text with émojis 🚀
        """
        patterns = extractor.extract_from_text(unicode_text, "unicode.md")
        assert len(patterns) > 0

    def test_mixed_line_endings(self, extractor):
        """Test extraction from text with mixed line endings"""
        text_with_mixed_endings = "Task: First task\r\nGoal: Second goal\rObjective: Third objective\n"
        patterns = extractor.extract_from_text(text_with_mixed_endings, "mixed.md")
        assert len(patterns) > 0
