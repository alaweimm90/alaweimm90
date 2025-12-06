"""Pytest fixtures for PromptForge tests"""

import pytest
import tempfile
from pathlib import Path
from promptforge.models import PromptPattern, ExtractionResult
from promptforge.extractor import PatternExtractor
from promptforge.library import PromptLibrary


@pytest.fixture
def sample_text_basic():
    """Basic sample text with simple patterns"""
    return """
# Project Notes

Task: Implement user authentication
Goal: Create a secure login system

You are a senior software architect.
Act as a helpful coding assistant.

Format: JSON output with fields id, name, email
Output: Return data in CSV format

Constraint: Must use HTTPS
Requirement: Password must be at least 8 characters

Example: user@example.com
Sample: john.doe@company.com

Step 1: Validate input
Step 2: Check credentials
First: Initialize database connection
Then: Execute query
Finally: Return results

If user is authenticated, then grant access.
When request arrives, do validate it.

Context: This is a web application for employee management
Background: The system handles sensitive data
"""


@pytest.fixture
def sample_text_with_variables():
    """Sample text with variable placeholders"""
    return """
Task: Process {filename} and extract {data_type}
Goal: Convert [input_format] to [output_format]

You are a {role} working on <project_name>.

Format: Output should be ${output_format}
Constraint: Must complete within {timeout} seconds

Example: e.g., convert PDF to JSON
"""


@pytest.fixture
def sample_text_with_tags():
    """Sample text with hashtags"""
    return """
#coding #python #automation

Task: Automate the deployment process
You are a DevOps engineer #devops

Write a script to generate documentation #documentation
Analyze the code for security issues #security #analysis
"""


@pytest.fixture
def sample_text_complex():
    """Complex sample text with multiple pattern types"""
    return """
# Advanced Prompt Engineering Guide

## System Design Pattern

You are an expert system architect specializing in {domain}.

Goal: Design a scalable {system_type} that handles {capacity} requests per second.

Constraints:
- Must use microservices architecture
- Should support horizontal scaling
- Required: 99.9% uptime SLA

Format: Provide output as architectural diagram with:
1. Component overview
2. Data flow
3. API specifications

Context: This system will replace the legacy {legacy_system} currently handling {current_load} users.

Step 1: Analyze current system bottlenecks
Step 2: Identify critical components for migration
Then: Design new architecture with fallback mechanisms
Finally: Create migration plan with rollback strategy

If traffic exceeds {threshold}, then activate auto-scaling.
When error rate > 5%, should trigger alerts and start investigation.

Example: For a social media platform with 1M daily active users, you would design...

#architecture #system-design #scalability
"""


@pytest.fixture
def sample_markdown_file(tmp_path):
    """Create a temporary markdown file"""
    file_path = tmp_path / "test_notes.md"
    content = """
Task: Extract data from documents
You are a data extraction specialist
Format: JSON with nested objects
"""
    file_path.write_text(content)
    return file_path


@pytest.fixture
def sample_directory_with_files(tmp_path):
    """Create a temporary directory with multiple files"""
    # Create markdown files
    (tmp_path / "file1.md").write_text("""
Task: First task
Goal: Complete objective one
""")

    (tmp_path / "file2.md").write_text("""
You are a project manager
Constraint: Must finish by deadline
""")

    (tmp_path / "file3.txt").write_text("""
Example: Sample implementation
Step 1: Initialize system
Step 2: Execute process
""")

    # Create a subdirectory
    subdir = tmp_path / "subdir"
    subdir.mkdir()
    (subdir / "file4.md").write_text("""
Format: XML output
Context: Enterprise application
""")

    return tmp_path


@pytest.fixture
def extractor():
    """Create a PatternExtractor instance"""
    return PatternExtractor()


@pytest.fixture
def library():
    """Create an empty PromptLibrary instance"""
    return PromptLibrary()


@pytest.fixture
def sample_pattern():
    """Create a sample PromptPattern"""
    return PromptPattern(
        pattern_type="instruction",
        pattern_name="instruction_implement_user_auth",
        template="Implement user authentication",
        variables=["user", "auth"],
        example="Implement user authentication",
        source_file="test.md",
        line_number=1,
        confidence=0.7,
        tags=["coding", "security"]
    )


@pytest.fixture
def sample_patterns():
    """Create multiple sample PromptPatterns"""
    return [
        PromptPattern(
            pattern_type="instruction",
            pattern_name="instruction_task_one",
            template="Complete task one with {parameter}",
            variables=["parameter"],
            example="Complete task one",
            source_file="test1.md",
            line_number=1,
            confidence=0.8,
            tags=["coding"]
        ),
        PromptPattern(
            pattern_type="role_play",
            pattern_name="role_play_software_engineer",
            template="software engineer",
            variables=[],
            example="You are a software engineer",
            source_file="test2.md",
            line_number=5,
            confidence=0.6,
            tags=["role"]
        ),
        PromptPattern(
            pattern_type="format",
            pattern_name="format_json_output",
            template="JSON output with fields {fields}",
            variables=["fields"],
            example="JSON output",
            source_file="test3.md",
            line_number=10,
            confidence=0.9,
            tags=["output", "coding"]
        ),
        PromptPattern(
            pattern_type="instruction",
            pattern_name="instruction_task_duplicate",
            template="Complete task one with {parameter}",  # Duplicate
            variables=["parameter"],
            example="Complete task one",
            source_file="test4.md",
            line_number=15,
            confidence=0.7,
            tags=["coding"]
        ),
    ]


@pytest.fixture
def sample_extraction_result(sample_patterns):
    """Create a sample ExtractionResult"""
    return ExtractionResult(
        total_patterns=len(sample_patterns),
        patterns=sample_patterns,
        files_processed=3,
        timestamp="2024-01-01T12:00:00"
    )


@pytest.fixture
def populated_library(library, sample_patterns):
    """Create a library populated with sample patterns"""
    library.add_patterns(sample_patterns)
    return library
