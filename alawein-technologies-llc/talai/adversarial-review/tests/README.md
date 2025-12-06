# AdversarialReview Test Suite

## Quick Start

### Run All Tests
```bash
cd /mnt/c/Users/mesha/Documents/TalAI/adversarial-review
PYTHONPATH=src venv/bin/pytest tests/ -v
```

### Run with Coverage
```bash
PYTHONPATH=src venv/bin/pytest tests/ -v --cov=adversarial_review --cov-report=term
```

### Run Specific Test File
```bash
# Test only critics
PYTHONPATH=src venv/bin/pytest tests/test_critics.py -v

# Test only integration
PYTHONPATH=src venv/bin/pytest tests/test_adversarial_review.py -v
```

### Run Specific Test Class
```bash
# Test only StatisticalCritic
PYTHONPATH=src venv/bin/pytest tests/test_critics.py::TestStatisticalCritic -v

# Test only AdversarialReview main class
PYTHONPATH=src venv/bin/pytest tests/test_adversarial_review.py::TestAdversarialReview -v
```

### Run Specific Test
```bash
PYTHONPATH=src venv/bin/pytest tests/test_critics.py::TestStatisticalCritic::test_initialization -v
```

## Test Suite Structure

```
tests/
├── __init__.py                    # Package marker
├── conftest.py                    # Shared fixtures (103 lines)
├── test_critics.py                # Critic unit tests (75 tests, 477 lines)
├── test_adversarial_review.py     # Integration tests (28 tests, 301 lines)
└── README.md                      # This file
```

## Test Statistics

- **Total Tests**: 103
- **Pass Rate**: 100%
- **Coverage**: 87%
- **Execution Time**: ~1 second

## Test Categories

### Critic Tests (75 tests)
Tests for all 6 critics:
- StatisticalCritic (11 tests)
- MethodologicalCritic (7 tests)
- LogicalCritic (6 tests)
- HistoricalCritic (6 tests)
- EthicalCritic (7 tests)
- EconomicCritic (7 tests)
- Cross-Critic Behavior (30 tests)

### Integration Tests (28 tests)
- AdversarialReview workflow (23 tests)
- Edge cases (5 tests)

## Fixtures Available

From `conftest.py`:
- `sample_paper_text` - Standard sample paper
- `short_paper_text` - Brief text
- `long_paper_text` - Extended paper with multiple sections
- `paper_title` - Sample title
- `alternative_title` - Alternative title
- `nightmare_mode` - "nightmare" mode
- `normal_mode` - "standard" mode
- `brutal_mode` - "brutal" mode
- `expected_score_ranges` - Score validation ranges
- `valid_severity_levels` - Valid severities
- `valid_verdicts` - Valid verdict options
- `critic_dimensions` - Critic name to dimension mapping
- `minimum_issues_by_mode` - Expected minimum issues
- `maximum_issues_by_mode` - Expected maximum issues

## Coverage Report

```
Name                                 Stmts   Miss  Cover
--------------------------------------------------------
src/adversarial_review/__init__.py       2      0   100%
src/adversarial_review/main.py         185     24    87%
--------------------------------------------------------
TOTAL                                  187     24    87%
```

## Writing New Tests

### Example Test Structure
```python
import pytest
from adversarial_review.main import StatisticalCritic

class TestNewFeature:
    """Tests for new feature"""

    def test_feature_behavior(self, sample_paper_text, normal_mode):
        """Test feature works correctly"""
        critic = StatisticalCritic()
        result = critic.review(sample_paper_text, normal_mode)

        assert result.score >= 0.0
        assert result.score <= 10.0
```

### Using Fixtures
```python
def test_with_fixtures(self, sample_paper_text, nightmare_mode):
    """Fixtures are automatically available via conftest.py"""
    critic = StatisticalCritic()
    result = critic.review(sample_paper_text, nightmare_mode)
    assert len(result.issues) > 0
```

### Parameterized Tests
```python
@pytest.mark.parametrize("mode,expected_min_issues", [
    ("standard", 2),
    ("nightmare", 4),
])
def test_modes(self, sample_paper_text, mode, expected_min_issues):
    critic = StatisticalCritic()
    result = critic.review(sample_paper_text, mode)
    assert len(result.issues) >= expected_min_issues
```

## Best Practices

1. **Use fixtures** for common test data
2. **Test business logic**, not implementation details
3. **Keep tests fast** - current suite runs in ~1 second
4. **Use descriptive names** - `test_score_calculation_logic` not `test1`
5. **Test edge cases** - empty inputs, boundary values
6. **One assertion focus** per test when possible
7. **Use parametrize** for similar tests with different inputs

## Continuous Integration

To integrate with CI/CD:

```yaml
# Example GitHub Actions
- name: Run Tests
  run: |
    PYTHONPATH=src pytest tests/ -v --cov=adversarial_review --cov-report=xml

- name: Coverage Check
  run: |
    coverage report --fail-under=70
```

## Troubleshooting

### Import Errors
Make sure `PYTHONPATH=src` is set before running tests:
```bash
PYTHONPATH=src pytest tests/
```

### Virtual Environment
Ensure virtual environment is activated or use full path:
```bash
venv/bin/pytest tests/
```

### Coverage Not Working
Install pytest-cov:
```bash
venv/bin/pip install pytest-cov
```

## Additional Options

### Verbose Output
```bash
PYTHONPATH=src venv/bin/pytest tests/ -vv
```

### Stop on First Failure
```bash
PYTHONPATH=src venv/bin/pytest tests/ -x
```

### Show Local Variables on Failure
```bash
PYTHONPATH=src venv/bin/pytest tests/ -l
```

### Generate HTML Coverage Report
```bash
PYTHONPATH=src venv/bin/pytest tests/ --cov=adversarial_review --cov-report=html
# Open htmlcov/index.html in browser
```

## Support

For issues or questions about the test suite, refer to:
- Main documentation: `../README.md`
- Test report: `../TEST_REPORT.md`
- Source code: `../src/adversarial_review/main.py`
