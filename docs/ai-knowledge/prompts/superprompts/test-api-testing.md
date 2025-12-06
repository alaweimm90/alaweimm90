# Write Comprehensive Tests For Rest

> **Comprehensive testing strategy**

## Purpose

write comprehensive tests for REST API.

## When to Use

- When you need to accomplish the stated goal
- When working on similar problems
- When you want to follow best practices

## Prompt

```markdown
Create comprehensive tests for:

### Test Coverage
- Unit tests: Core functionality
- Integration tests: Component interaction
- End-to-end tests: Full workflows
- Edge cases: Boundary conditions

### Test Strategy
1. Identify test scenarios
2. Write test cases
3. Implement tests
4. Run and validate
5. Maintain test suite

### Quality Gates
- Code coverage: > 80%
- All tests pass
- No flaky tests
- Fast execution (< 5 min)

### Test Structure
```python
def test_feature():
    # Arrange
    setup_test_data()
    
    # Act
    result = execute_feature()
    
    # Assert
    assert result == expected
```
```

## Examples

### Input
```
Example input based on requirement
```

### Output
```
Expected output
```

## Success Criteria

- [ ] 80%+ code coverage
- [ ] All tests pass
- [ ] Edge cases covered
- [ ] Tests are maintainable

## Related Prompts

- automated-test-generation
- testing-qa-strategy

---

**Domain**: testing  
**Generated**: Meta-Prompt Generator v1.0
