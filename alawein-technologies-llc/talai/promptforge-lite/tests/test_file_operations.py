"""
Test suite for file operation functions
"""
import pytest
import json
from pathlib import Path
from promptforge_lite.main import (
    extract_from_file,
    extract_from_directory,
    PatternExtractor,
    PromptPattern
)


class TestExtractFromFile:
    """Test extract_from_file function"""

    def test_extract_from_existing_file(self, temp_markdown_file, extractor):
        """Test extraction from an existing file"""
        patterns = extract_from_file(temp_markdown_file, extractor)
        assert isinstance(patterns, list)
        assert len(patterns) > 0

    def test_extract_from_file_correct_source(self, temp_markdown_file, extractor):
        """Test that source file is correctly set"""
        patterns = extract_from_file(temp_markdown_file, extractor)
        assert all(p.source_file == str(temp_markdown_file) for p in patterns)

    def test_extract_from_nonexistent_file(self, temp_dir, extractor):
        """Test extraction from non-existent file returns empty list"""
        nonexistent = temp_dir / "nonexistent.md"
        patterns = extract_from_file(nonexistent, extractor)
        assert patterns == []

    def test_extract_from_empty_file(self, temp_dir, extractor):
        """Test extraction from empty file"""
        empty_file = temp_dir / "empty.md"
        empty_file.write_text("", encoding='utf-8')
        patterns = extract_from_file(empty_file, extractor)
        assert patterns == []

    def test_extract_from_file_with_unicode(self, temp_dir, extractor):
        """Test extraction from file with unicode content"""
        unicode_file = temp_dir / "unicode.md"
        unicode_file.write_text("Task: Analyze données 数据 🎯", encoding='utf-8')
        patterns = extract_from_file(unicode_file, extractor)
        assert isinstance(patterns, list)


class TestExtractFromDirectory:
    """Test extract_from_directory function"""

    def test_extract_from_directory_basic(self, temp_multiple_files, extractor):
        """Test basic directory extraction"""
        temp_dir, files = temp_multiple_files
        result = extract_from_directory(temp_dir, extractor)

        assert result.total_patterns > 0
        assert result.files_processed == 3  # 2 md files + 1 txt file
        assert isinstance(result.patterns, list)
        assert result.timestamp is not None

    def test_extract_from_directory_finds_md_files(self, temp_multiple_files, extractor):
        """Test that .md files are processed"""
        temp_dir, files = temp_multiple_files
        result = extract_from_directory(temp_dir, extractor)

        # Check that patterns from .md files are present
        md_patterns = [p for p in result.patterns if p.source_file.endswith('.md')]
        assert len(md_patterns) > 0

    def test_extract_from_directory_finds_txt_files(self, temp_multiple_files, extractor):
        """Test that .txt files are processed"""
        temp_dir, files = temp_multiple_files
        result = extract_from_directory(temp_dir, extractor)

        # Check that patterns from .txt files are present
        txt_patterns = [p for p in result.patterns if p.source_file.endswith('.txt')]
        assert len(txt_patterns) > 0

    def test_extract_from_directory_recursive(self, temp_multiple_files, extractor):
        """Test that subdirectories are scanned recursively"""
        temp_dir, files = temp_multiple_files
        result = extract_from_directory(temp_dir, extractor)

        # Check that file from subdirectory was processed
        subdir_patterns = [p for p in result.patterns if 'subdir' in p.source_file]
        assert len(subdir_patterns) > 0

    def test_extract_from_empty_directory(self, temp_dir, extractor):
        """Test extraction from empty directory"""
        result = extract_from_directory(temp_dir, extractor)

        assert result.total_patterns == 0
        assert result.files_processed == 0
        assert result.patterns == []

    def test_extract_from_directory_timestamp_format(self, temp_multiple_files, extractor):
        """Test that timestamp is in ISO format"""
        temp_dir, files = temp_multiple_files
        result = extract_from_directory(temp_dir, extractor)

        # Check timestamp format (should be ISO format)
        assert 'T' in result.timestamp or '-' in result.timestamp

    def test_extract_from_directory_pattern_count_matches(self, temp_multiple_files, extractor):
        """Test that total_patterns matches length of patterns list"""
        temp_dir, files = temp_multiple_files
        result = extract_from_directory(temp_dir, extractor)

        assert result.total_patterns == len(result.patterns)


class TestDataClasses:
    """Test dataclass functionality"""

    def test_prompt_pattern_creation(self):
        """Test creating a PromptPattern instance"""
        pattern = PromptPattern(
            pattern_type="instruction",
            pattern_name="test_pattern",
            template="Test template",
            variables=["var1"],
            example="Example",
            source_file="test.md",
            line_number=1,
            confidence=0.7,
            tags=["test"]
        )

        assert pattern.pattern_type == "instruction"
        assert pattern.pattern_name == "test_pattern"
        assert pattern.template == "Test template"
        assert pattern.variables == ["var1"]
        assert pattern.confidence == 0.7

    def test_prompt_pattern_with_empty_variables(self):
        """Test PromptPattern with empty variables list"""
        pattern = PromptPattern(
            pattern_type="instruction",
            pattern_name="test",
            template="Test",
            variables=[],
            example="Ex",
            source_file="test.md",
            line_number=1,
            confidence=0.5,
            tags=[]
        )

        assert pattern.variables == []
        assert pattern.tags == []

    def test_extraction_result_creation(self):
        """Test creating an ExtractionResult instance"""
        from promptforge_lite.main import ExtractionResult

        patterns = [
            PromptPattern(
                pattern_type="instruction",
                pattern_name="test",
                template="Test",
                variables=[],
                example="Ex",
                source_file="test.md",
                line_number=1,
                confidence=0.5,
                tags=[]
            )
        ]

        result = ExtractionResult(
            total_patterns=1,
            patterns=patterns,
            files_processed=1,
            timestamp="2025-11-16T00:00:00"
        )

        assert result.total_patterns == 1
        assert len(result.patterns) == 1
        assert result.files_processed == 1


class TestJSONSerialization:
    """Test JSON serialization of patterns"""

    def test_pattern_to_dict(self, sample_prompt_pattern):
        """Test converting PromptPattern to dict"""
        from dataclasses import asdict

        pattern_dict = asdict(sample_prompt_pattern)
        assert isinstance(pattern_dict, dict)
        assert pattern_dict['pattern_type'] == "instruction"
        assert pattern_dict['pattern_name'] == "instruction_analyze_code"
        assert 'variables' in pattern_dict
        assert 'tags' in pattern_dict

    def test_extraction_result_to_json(self, temp_markdown_file, extractor):
        """Test converting ExtractionResult to JSON"""
        from dataclasses import asdict
        from promptforge_lite.main import ExtractionResult

        patterns = extract_from_file(temp_markdown_file, extractor)
        result = ExtractionResult(
            total_patterns=len(patterns),
            patterns=patterns,
            files_processed=1,
            timestamp="2025-11-16T00:00:00"
        )

        result_dict = asdict(result)
        json_str = json.dumps(result_dict, indent=2)

        assert isinstance(json_str, str)
        assert '"total_patterns"' in json_str
        assert '"patterns"' in json_str

    def test_pattern_from_dict(self, extraction_result_data):
        """Test creating PromptPattern from dict"""
        pattern_data = extraction_result_data['patterns'][0]
        pattern = PromptPattern(**pattern_data)

        assert pattern.pattern_type == "instruction"
        assert pattern.pattern_name == "instruction_test"
        assert pattern.confidence == 0.7

    def test_roundtrip_serialization(self, sample_prompt_pattern):
        """Test roundtrip JSON serialization"""
        from dataclasses import asdict

        # Convert to dict
        pattern_dict = asdict(sample_prompt_pattern)

        # Convert to JSON and back
        json_str = json.dumps(pattern_dict)
        loaded_dict = json.loads(json_str)

        # Create new pattern from loaded dict
        new_pattern = PromptPattern(**loaded_dict)

        assert new_pattern.pattern_type == sample_prompt_pattern.pattern_type
        assert new_pattern.pattern_name == sample_prompt_pattern.pattern_name
        assert new_pattern.template == sample_prompt_pattern.template
        assert new_pattern.variables == sample_prompt_pattern.variables
