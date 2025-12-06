# PromptForge Lite - Test Suite

## Quick Start

Run all tests with coverage:
```bash
PYTHONPATH=src pytest tests/ -v --cov=promptforge_lite --cov-report=term
```

## Test Execution Commands

### Run All Tests (Verbose)
```bash
PYTHONPATH=src pytest tests/ -v
```

### Run All Tests with Coverage
```bash
PYTHONPATH=src pytest tests/ -v --cov=promptforge_lite --cov-report=term
```

### Run with Coverage Report (Missing Lines)
```bash
PYTHONPATH=src pytest tests/ --cov=promptforge_lite --cov-report=term-missing
```

### Run Specific Test Module
```bash
PYTHONPATH=src pytest tests/test_pattern_extractor.py -v
PYTHONPATH=src pytest tests/test_prompt_library.py -v
PYTHONPATH=src pytest tests/test_file_operations.py -v
PYTHONPATH=src pytest tests/test_integration.py -v
```

### Run Specific Test Class
```bash
PYTHONPATH=src pytest tests/test_pattern_extractor.py::TestExtractFromText -v
PYTHONPATH=src pytest tests/test_prompt_library.py::TestFilterByType -v
```

### Run Specific Test Case
```bash
PYTHONPATH=src pytest tests/test_pattern_extractor.py::TestExtractFromText::test_extract_instruction_pattern -v
```

### Run Tests in Parallel (faster)
```bash
PYTHONPATH=src pytest tests/ -n auto
```

## Test Suite Overview

### Test Files

1. **conftest.py** - Pytest fixtures and test data
   - 20+ reusable fixtures
   - Sample patterns, text fixtures, file fixtures

2. **test_pattern_extractor.py** - PatternExtractor tests (35 tests)
   - Pattern extraction for all 8 types
   - Variable detection tests
   - Confidence scoring tests
   - Tag and example extraction tests

3. **test_prompt_library.py** - PromptLibrary tests (35 tests)
   - Library management tests
   - Filtering operations (type, tag, confidence)
   - Deduplication tests
   - Ranking tests

4. **test_file_operations.py** - File I/O tests (19 tests)
   - File extraction tests
   - Directory scanning tests
   - JSON serialization tests

5. **test_integration.py** - Integration tests (19 tests)
   - End-to-end workflows
   - Edge cases
   - Data integrity tests

### Total: 108 Tests

## Expected Results

```
================================ tests coverage ================================
Name                               Stmts   Miss  Cover
------------------------------------------------------
src/promptforge_lite/__init__.py       2      0   100%
src/promptforge_lite/main.py         225     75    67%
------------------------------------------------------
TOTAL                                227     75    67%
============================= 108 passed in ~1.5s ===========================
```

## Test Coverage

- **Overall Coverage**: 67%
- **Business Logic Coverage**: 90%+ (excluding CLI code)
- **Pass Rate**: 100% (108/108)

## Fixtures Available

### Component Fixtures
- `extractor` - PatternExtractor instance
- `library` - PromptLibrary instance
- `sample_prompt_pattern` - Single pattern
- `sample_patterns_list` - Multiple patterns

### Text Fixtures
- `instruction_text` - Instruction patterns
- `role_play_text` - Role play patterns
- `format_text` - Format patterns
- `constraint_text` - Constraint patterns
- `step_by_step_text` - Step-by-step patterns
- `conditional_text` - Conditional patterns
- `context_text` - Context patterns
- `variables_text` - Variable formats
- `multi_pattern_text` - Multiple patterns
- `tagged_text` - Tagged content

### File Fixtures
- `temp_dir` - Temporary directory
- `temp_markdown_file` - Single temp file
- `temp_multiple_files` - Multiple files

## Troubleshooting

### PYTHONPATH Issues
If tests can't find the module:
```bash
export PYTHONPATH=src
pytest tests/
```

Or use the full path in each command:
```bash
PYTHONPATH=src pytest tests/ -v
```

### Import Errors
Make sure you're in the project root:
```bash
cd /path/to/promptforge-lite
```

### Coverage Not Working
Install pytest-cov:
```bash
pip install pytest-cov
```

## Writing New Tests

### Test Structure
```python
class TestFeatureName:
    """Test description"""

    def test_specific_behavior(self, fixture_name):
        """Test what this validates"""
        # Arrange
        # Act
        # Assert
        assert expected == actual
```

### Using Fixtures
```python
def test_with_extractor(self, extractor):
    """Tests can use any fixture from conftest.py"""
    patterns = extractor.extract_from_text("Task: Do something", "test.md")
    assert len(patterns) > 0
```

### Adding New Fixtures
Add to `conftest.py`:
```python
@pytest.fixture
def my_fixture():
    """Fixture description"""
    return test_data
```

## CI/CD Integration

### GitHub Actions Example
```yaml
- name: Run tests
  run: |
    pip install pytest pytest-cov
    PYTHONPATH=src pytest tests/ --cov=promptforge_lite --cov-report=xml
```

### Pre-commit Hook
```bash
#!/bin/bash
PYTHONPATH=src pytest tests/ -v
```

## Additional Resources

- See `../TEST_REPORT.md` for detailed test analysis
- See `../README.md` for project documentation
- PyTest documentation: https://docs.pytest.org/
