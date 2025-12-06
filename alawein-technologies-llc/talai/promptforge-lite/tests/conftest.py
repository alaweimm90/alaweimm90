"""
Pytest fixtures for PromptForge Lite test suite
"""
import pytest
from pathlib import Path
from tempfile import TemporaryDirectory
from promptforge_lite.main import (
    PatternExtractor,
    PromptLibrary,
    PromptPattern,
    ExtractionResult
)


@pytest.fixture
def extractor():
    """Create a PatternExtractor instance"""
    return PatternExtractor()


@pytest.fixture
def library():
    """Create an empty PromptLibrary instance"""
    return PromptLibrary()


@pytest.fixture
def sample_prompt_pattern():
    """Create a sample PromptPattern for testing"""
    return PromptPattern(
        pattern_type="instruction",
        pattern_name="instruction_analyze_code",
        template="Analyze the {code} and provide feedback",
        variables=["code"],
        example="Analyze the code and provide feedback",
        source_file="test.md",
        line_number=1,
        confidence=0.7,
        tags=["analysis", "coding"]
    )


@pytest.fixture
def sample_patterns_list():
    """Create a list of sample patterns for testing"""
    return [
        PromptPattern(
            pattern_type="instruction",
            pattern_name="instruction_write_code",
            template="Write a {language} function to {task}",
            variables=["language", "task"],
            example="Write a Python function",
            source_file="test1.md",
            line_number=1,
            confidence=0.8,
            tags=["coding", "generation"]
        ),
        PromptPattern(
            pattern_type="role_play",
            pattern_name="role_play_expert_developer",
            template="expert developer with years of experience",
            variables=[],
            example="You are an expert developer",
            source_file="test2.md",
            line_number=5,
            confidence=0.6,
            tags=["role"]
        ),
        PromptPattern(
            pattern_type="format",
            pattern_name="format_json_output",
            template="Return the output as JSON with {fields}",
            variables=["fields"],
            example="format: Return as JSON",
            source_file="test3.md",
            line_number=10,
            confidence=0.7,
            tags=["formatting"]
        ),
        PromptPattern(
            pattern_type="constraint",
            pattern_name="constraint_must_validate",
            template="Must validate all inputs before processing",
            variables=[],
            example="constraint: Must validate",
            source_file="test4.md",
            line_number=15,
            confidence=0.65,
            tags=["validation"]
        )
    ]


@pytest.fixture
def instruction_text():
    """Sample text with instruction pattern"""
    return """# Project Guidelines

Task: Create a comprehensive test suite for the application

This should include unit tests and integration tests.
"""


@pytest.fixture
def role_play_text():
    """Sample text with role play pattern"""
    return """
    You are a senior software architect with expertise in Python.

    Please review the following code.
    """


@pytest.fixture
def format_text():
    """Sample text with format pattern"""
    return """
    Output: Please return the results in the following format:

    - Name
    - Description
    - Status
    """


@pytest.fixture
def constraint_text():
    """Sample text with constraint pattern"""
    return """
    Requirements:
    - Must handle edge cases
    - Should include error handling

    Constraint: All functions must be documented
    """


@pytest.fixture
def step_by_step_text():
    """Sample text with step-by-step pattern"""
    return """
    Step 1: Initialize the environment
    Step 2: Load configuration files
    Step 3: Start the application
    Finally: Run tests
    """


@pytest.fixture
def conditional_text():
    """Sample text with conditional pattern"""
    return """
    If the user is authenticated, then grant access to the dashboard.

    When validation fails, show an error message.
    """


@pytest.fixture
def context_text():
    """Sample text with context pattern"""
    return """
    Context: This application is designed for data processing
    and analysis in research environments.

    It uses Python for backend processing.
    """


@pytest.fixture
def variables_text():
    """Sample text with various variable formats"""
    return """Task: Process {input_file} and save to [output_dir]

Use <template_name> with $config_file settings.

Apply ${environment} specific configurations.
"""


@pytest.fixture
def multi_pattern_text():
    """Sample text with multiple pattern types"""
    return """
    # System Design Prompt

    Role: You are a system design expert.

    Task: Design a scalable {system_type} architecture

    Format: Provide the design as a structured document

    Constraint: Must consider high availability

    Step 1: Analyze requirements
    Step 2: Design components

    If scalability is critical, then use microservices architecture.

    Context: This is for a financial services application

    #architecture #design #scalability

    Example: Design a payment processing system
    """


@pytest.fixture
def tagged_text():
    """Sample text with hashtags"""
    return """Task: Analyze the code quality

#code #review #quality #analysis

This should generate comprehensive reports.
"""


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing"""
    with TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_markdown_file(temp_dir):
    """Create a temporary markdown file with sample content"""
    file_path = temp_dir / "test.md"
    content = """
    # Test Document

    Task: Create a test suite

    You are an expert tester.

    Format: Return results as JSON
    """
    file_path.write_text(content, encoding='utf-8')
    return file_path


@pytest.fixture
def temp_multiple_files(temp_dir):
    """Create multiple temporary files for directory scanning"""
    files = []

    # File 1
    file1 = temp_dir / "doc1.md"
    file1.write_text("Task: Build the application\n#development", encoding='utf-8')
    files.append(file1)

    # File 2
    file2 = temp_dir / "doc2.txt"
    file2.write_text("You are a code reviewer.\nConstraint: Must be thorough", encoding='utf-8')
    files.append(file2)

    # File 3 in subdirectory
    subdir = temp_dir / "subdir"
    subdir.mkdir()
    file3 = subdir / "doc3.md"
    file3.write_text("Format: Output as CSV\nStep 1: Parse data", encoding='utf-8')
    files.append(file3)

    return temp_dir, files


@pytest.fixture
def extraction_result_data():
    """Sample extraction result data for JSON testing"""
    return {
        "total_patterns": 2,
        "patterns": [
            {
                "pattern_type": "instruction",
                "pattern_name": "instruction_test",
                "template": "Test template",
                "variables": ["var1"],
                "example": "Example",
                "source_file": "test.md",
                "line_number": 1,
                "confidence": 0.7,
                "tags": ["test"]
            },
            {
                "pattern_type": "role_play",
                "pattern_name": "role_play_expert",
                "template": "Expert template",
                "variables": [],
                "example": "Example",
                "source_file": "test.md",
                "line_number": 5,
                "confidence": 0.6,
                "tags": ["role"]
            }
        ],
        "files_processed": 1,
        "timestamp": "2025-11-16T00:00:00"
    }


@pytest.fixture
def short_template_text():
    """Very short template text for confidence scoring"""
    return "Task: Do it"


@pytest.fixture
def long_template_text():
    """Long template text for truncation testing"""
    return "Task: " + "A" * 600  # Will be truncated to 500 chars


@pytest.fixture
def duplicate_patterns_text():
    """Text that will generate duplicate patterns"""
    return """
    Task: Analyze the data

    Some other content here.

    Task: Analyze the data
    """
