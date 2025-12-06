# GhostResearcher Test Suite

Comprehensive test suite for the GhostResearcher project.

## Test Statistics

- **Total Tests**: 102
- **Pass Rate**: 100%
- **Code Coverage**: 99%

## Test Structure

### 1. `conftest.py`
Pytest configuration and fixtures:
- `temp_data_file`: Creates temporary JSON files for testing
- `sample_consultation_data`: Sample consultation data
- `mock_scientist_profile`: Mock scientist profile
- `prepopulated_data_file`: Pre-populated data file
- `all_scientist_names`: List of all scientists
- `valid_domains`: Valid problem domains
- `sample_problems`: Sample problems for testing

### 2. `test_ghost_researcher.py` (44 tests)
Core functionality tests:
- **TestGhostResearcherInit** (3 tests): Initialization with new/existing files
- **TestConsultMethod** (6 tests): Consultation creation and validation
- **TestConsultationStructure** (4 tests): Consultation data structure
- **TestDataPersistence** (3 tests): Data saving and loading
- **TestScientistInfo** (3 tests): Scientist information retrieval
- **TestPrivateMethods** (9 tests): Internal helper methods
- **TestConfidenceEstimation** (2 tests): Confidence calculation
- **TestLimitationsIdentification** (2 tests): Limitations identification
- **TestEdgeCases** (3 tests): Edge cases and error handling
- **TestApproachGeneration** (4 tests): Approach generation for different personalities

### 3. `test_scientists_data.py` (14 tests)
Data structure validation:
- **TestScientistsDataStructure** (4 tests): SCIENTISTS dict structure
- **TestIndividualScientists** (8 tests): Individual scientist data
- **TestLifeDates** (3 tests): Date format and validity
- **TestFieldsValidity** (2 tests): Field values
- **TestQuotesQuality** (2 tests): Quote quality

### 4. `test_consultation_dataclass.py` (4 tests)
Consultation dataclass tests:
- Creation and serialization
- Required fields
- Field type validation

### 5. `test_generation_methods.py` (25 tests)
Content generation methods:
- **TestInitialReactionGeneration** (3 tests): Initial reactions
- **TestAnalogiesGeneration** (2 tests): Historical analogies
- **TestExperimentalSuggestions** (2 tests): Experiment suggestions
- **TestTheoreticalFramework** (3 tests): Framework generation
- **TestCharacteristicQuotes** (3 tests): Quote generation
- **TestThoughtExperiments** (3 tests): Thought experiments
- **TestObstaclesPrediction** (2 tests): Obstacle prediction
- **TestInsightsGeneration** (3 tests): Key insights
- **TestGenerateApproachVariations** (2 tests): Approach variations
- **TestEraContextVariations** (1 test): Era context
- **TestDomainNatureMapping** (2 tests): Domain mapping

### 6. `test_main_function.py` (7 tests)
CLI interface tests:
- **TestMainFunctionDirect** (7 tests): Direct main() function testing
  - List command
  - Info command
  - Consult command
  - Output validation

### 7. `test_cli.py` (8 tests)
CLI subprocess tests:
- **TestCLIList** (1 test): List command via subprocess
- **TestCLIInfo** (2 tests): Info command validation
- **TestCLIConsult** (4 tests): Consult command execution
- **TestCLINoCommand** (1 test): Help display

## Coverage Details

```
Name                               Stmts   Miss  Cover
------------------------------------------------------
src/ghost_researcher/__init__.py       2      0   100%
src/ghost_researcher/main.py         214      3    99%
------------------------------------------------------
TOTAL                                216      3    99%
```

## Running Tests

### Run all tests:
```bash
PYTHONPATH=src pytest tests/ -v
```

### Run with coverage:
```bash
PYTHONPATH=src pytest tests/ -v --cov=ghost_researcher --cov-report=term
```

### Run specific test file:
```bash
PYTHONPATH=src pytest tests/test_ghost_researcher.py -v
```

### Run specific test class:
```bash
PYTHONPATH=src pytest tests/test_ghost_researcher.py::TestConsultMethod -v
```

### Run specific test:
```bash
PYTHONPATH=src pytest tests/test_ghost_researcher.py::TestConsultMethod::test_consult_valid_scientist -v
```

## Test Categories

### Business Logic Tests (70+ tests)
- Consultation creation and management
- Content generation (reactions, analogies, insights, etc.)
- Confidence estimation
- Domain-specific logic
- Era context determination

### Data Validation Tests (14 tests)
- SCIENTISTS database structure
- Data completeness and format
- Historical accuracy

### Integration Tests (15 tests)
- CLI interface
- Data persistence
- End-to-end workflows

### Edge Cases (3+ tests)
- Empty inputs
- Very long strings
- Special characters
- Invalid scientists

## Key Features Tested

1. **Consultation Creation**: All 8 scientists with various domains
2. **Data Persistence**: JSON serialization/deserialization
3. **Content Generation**: All generation methods produce valid output
4. **CLI Interface**: All three commands (list, info, consult)
5. **Error Handling**: Invalid inputs and edge cases
6. **Business Logic**: Confidence, limitations, approaches, insights
7. **Data Quality**: Scientists data completeness and validity

## Fixtures

The test suite uses extensive fixtures for:
- Temporary file management
- Sample data generation
- Mock profiles
- Prepopulated data files
- Scientist lists and domains

All fixtures are defined in `conftest.py` and reused across tests.
