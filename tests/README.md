# Test Suite

Comprehensive test coverage for the meta-governance repository.

## Structure

```
tests/
├── ai/                    # AI module tests
│   ├── cache.test.ts
│   ├── compliance.test.ts
│   ├── errors.test.ts
│   ├── monitor.test.ts
│   └── security.test.ts
├── ORCHEX/                 # ORCHEX framework tests
│   └── utils/
├── devops_*.test.ts       # DevOps CLI tests
├── test_*.py              # Python unit tests
└── conftest.py            # Pytest configuration
```

## Running Tests

### TypeScript Tests (Vitest)

```bash
# Run all tests
npm test

# Run with coverage
npm run test:coverage

# Run specific test file
npm test -- tests/ai/compliance.test.ts

# Run in watch mode
npm run test:watch
```

### Python Tests (Pytest)

```bash
# Run all Python tests
pytest

# Run with coverage
pytest --cov=automation

# Run specific test
pytest tests/test_validation.py
```

## Test Conventions

1. **Naming**: `*.test.ts` for TypeScript, `test_*.py` for Python
2. **Location**: Tests mirror source structure
3. **Isolation**: Each test file is self-contained
4. **Fixtures**: Use `conftest.py` for shared Python fixtures

## Coverage Requirements

- Minimum 70% line coverage for new code
- Critical paths require 90%+ coverage
- Integration tests for all CLI commands
