"""Tests for data models"""

import pytest
from promptforge.models import PromptPattern, ExtractionResult


class TestPromptPattern:
    """Test PromptPattern model"""

    def test_create_pattern(self, sample_pattern):
        """Test creating a PromptPattern instance"""
        assert sample_pattern.pattern_type == "instruction"
        assert sample_pattern.pattern_name == "instruction_implement_user_auth"
        assert sample_pattern.template == "Implement user authentication"
        assert sample_pattern.variables == ["user", "auth"]
        assert sample_pattern.confidence == 0.7
        assert "coding" in sample_pattern.tags

    def test_to_dict(self, sample_pattern):
        """Test converting pattern to dictionary"""
        pattern_dict = sample_pattern.to_dict()
        assert isinstance(pattern_dict, dict)
        assert pattern_dict["pattern_type"] == "instruction"
        assert pattern_dict["pattern_name"] == "instruction_implement_user_auth"
        assert pattern_dict["confidence"] == 0.7
        assert pattern_dict["line_number"] == 1

    def test_from_dict(self):
        """Test creating pattern from dictionary"""
        data = {
            "pattern_type": "role_play",
            "pattern_name": "test_pattern",
            "template": "test template",
            "variables": ["var1", "var2"],
            "example": "test example",
            "source_file": "test.md",
            "line_number": 5,
            "confidence": 0.85,
            "tags": ["tag1", "tag2"]
        }
        pattern = PromptPattern.from_dict(data)
        assert pattern.pattern_type == "role_play"
        assert pattern.pattern_name == "test_pattern"
        assert pattern.confidence == 0.85
        assert len(pattern.variables) == 2

    def test_roundtrip_conversion(self, sample_pattern):
        """Test pattern survives dict conversion roundtrip"""
        pattern_dict = sample_pattern.to_dict()
        restored_pattern = PromptPattern.from_dict(pattern_dict)
        assert restored_pattern.pattern_type == sample_pattern.pattern_type
        assert restored_pattern.pattern_name == sample_pattern.pattern_name
        assert restored_pattern.template == sample_pattern.template
        assert restored_pattern.confidence == sample_pattern.confidence


class TestExtractionResult:
    """Test ExtractionResult model"""

    def test_create_extraction_result(self, sample_extraction_result):
        """Test creating an ExtractionResult instance"""
        assert sample_extraction_result.total_patterns == 4
        assert len(sample_extraction_result.patterns) == 4
        assert sample_extraction_result.files_processed == 3
        assert sample_extraction_result.timestamp == "2024-01-01T12:00:00"

    def test_to_dict(self, sample_extraction_result):
        """Test converting extraction result to dictionary"""
        result_dict = sample_extraction_result.to_dict()
        assert isinstance(result_dict, dict)
        assert result_dict["total_patterns"] == 4
        assert result_dict["files_processed"] == 3
        assert isinstance(result_dict["patterns"], list)
        assert len(result_dict["patterns"]) == 4
        assert isinstance(result_dict["patterns"][0], dict)

    def test_from_dict(self, sample_extraction_result):
        """Test creating extraction result from dictionary"""
        result_dict = sample_extraction_result.to_dict()
        restored_result = ExtractionResult.from_dict(result_dict)
        assert restored_result.total_patterns == sample_extraction_result.total_patterns
        assert restored_result.files_processed == sample_extraction_result.files_processed
        assert len(restored_result.patterns) == len(sample_extraction_result.patterns)

    def test_empty_extraction_result(self):
        """Test creating an empty extraction result"""
        result = ExtractionResult(
            total_patterns=0,
            patterns=[],
            files_processed=0,
            timestamp="2024-01-01T00:00:00"
        )
        assert result.total_patterns == 0
        assert len(result.patterns) == 0
        assert result.files_processed == 0

    def test_extraction_result_with_patterns(self, sample_patterns):
        """Test extraction result with multiple patterns"""
        result = ExtractionResult(
            total_patterns=len(sample_patterns),
            patterns=sample_patterns,
            files_processed=2,
            timestamp="2024-01-01T12:00:00"
        )
        assert result.total_patterns == 4
        assert len(result.patterns) == 4
        assert all(isinstance(p, PromptPattern) for p in result.patterns)
