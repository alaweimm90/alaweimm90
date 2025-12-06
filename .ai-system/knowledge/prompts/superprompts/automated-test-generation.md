# Test Generation and Coverage Analysis

## Purpose

Automated test creation with coverage goals and quality standards.

---

## Test Generation Framework

### 1. ANALYSIS

```yaml
code_analysis:
  identify:
    - functions_and_methods
    - input_output_types
    - dependencies
    - side_effects

  determine:
    - happy_path_scenarios
    - edge_cases
    - error_conditions
    - boundary_values

  assess:
    - complexity_score
    - risk_level
    - test_priority
```

### 2. TEST TYPES

#### Unit Tests (Happy Path)

```typescript
describe('calculateTotal', () => {
  it('should calculate total for valid items', () => {
    const items = [
      { price: 10, quantity: 2 },
      { price: 5, quantity: 3 },
    ];

    const result = calculateTotal(items);

    expect(result).toBe(35);
  });
});
```

#### Edge Case Tests

```typescript
describe('calculateTotal edge cases', () => {
  it('should return 0 for empty array', () => {
    expect(calculateTotal([])).toBe(0);
  });

  it('should handle single item', () => {
    expect(calculateTotal([{ price: 10, quantity: 1 }])).toBe(10);
  });

  it('should handle zero quantity', () => {
    expect(calculateTotal([{ price: 10, quantity: 0 }])).toBe(0);
  });

  it('should handle decimal prices', () => {
    expect(calculateTotal([{ price: 10.99, quantity: 2 }])).toBeCloseTo(21.98);
  });
});
```

#### Error Handling Tests

```typescript
describe('calculateTotal error handling', () => {
  it('should throw for null input', () => {
    expect(() => calculateTotal(null)).toThrow('Items cannot be null');
  });

  it('should throw for negative price', () => {
    expect(() => calculateTotal([{ price: -10, quantity: 1 }])).toThrow('Price cannot be negative');
  });

  it('should throw for negative quantity', () => {
    expect(() => calculateTotal([{ price: 10, quantity: -1 }])).toThrow(
      'Quantity cannot be negative'
    );
  });
});
```

#### Integration Tests

```typescript
describe('OrderService integration', () => {
  let orderService: OrderService;
  let mockDatabase: MockDatabase;
  let mockPaymentGateway: MockPaymentGateway;

  beforeEach(() => {
    mockDatabase = new MockDatabase();
    mockPaymentGateway = new MockPaymentGateway();
    orderService = new OrderService(mockDatabase, mockPaymentGateway);
  });

  it('should create order and process payment', async () => {
    const order = { items: [...], customerId: '123' };
    mockPaymentGateway.mockSuccess();

    const result = await orderService.createOrder(order);

    expect(result.status).toBe('completed');
    expect(mockDatabase.orders).toHaveLength(1);
    expect(mockPaymentGateway.charges).toHaveLength(1);
  });
});
```

### 3. COVERAGE GOALS

```yaml
coverage_targets:
  line_coverage:
    target: 80%
    critical_paths: 100%

  branch_coverage:
    target: 70%
    error_handling: 90%

  function_coverage:
    target: 90%
    public_api: 100%

  mutation_coverage:
    target: 60%
    core_logic: 80%
```

### 4. TEST QUALITY

#### Naming Convention

```typescript
// Pattern: should_[expected]_when_[condition]
it('should return empty array when input is empty', () => {});
it('should throw ValidationError when email is invalid', () => {});
it('should retry 3 times when connection fails', () => {});
```

#### Arrange-Act-Assert Pattern

```typescript
it('should update user profile', async () => {
  // Arrange
  const userId = '123';
  const updates = { name: 'New Name' };
  const existingUser = await createTestUser(userId);

  // Act
  const result = await userService.updateProfile(userId, updates);

  // Assert
  expect(result.name).toBe('New Name');
  expect(result.updatedAt).toBeAfter(existingUser.updatedAt);
});
```

#### Mock External Dependencies

```typescript
// Good: Mock at boundary
jest.mock('../services/paymentGateway');
const mockPayment = paymentGateway as jest.Mocked<typeof paymentGateway>;

beforeEach(() => {
  mockPayment.charge.mockResolvedValue({ success: true, transactionId: 'tx123' });
});

// Bad: Mock internal implementation
// Don't mock: private methods, utility functions, data transformations
```

#### Fast Execution

```yaml
performance_targets:
  unit_test: < 100ms
  integration_test: < 1s
  e2e_test: < 10s
  full_suite: < 5min

optimization_strategies:
  - parallel_execution
  - shared_fixtures
  - in_memory_databases
  - mock_external_services
  - skip_slow_tests_in_watch_mode
```

---

## Test Generation Prompt

````markdown
Generate tests for the following code:

```[language]
[paste code here]
```
````

Requirements:

1. Generate unit tests for all public methods
2. Include happy path, edge cases, and error handling
3. Use [testing framework: jest/pytest/etc.]
4. Follow Arrange-Act-Assert pattern
5. Mock external dependencies
6. Target 80% line coverage
7. Include descriptive test names

Output format:

- Test file with all test cases
- Coverage analysis
- Suggested additional tests for edge cases

````

---

## Coverage Analysis Report

```markdown
## Coverage Report

### Summary
| Metric | Current | Target | Status |
|--------|---------|--------|--------|
| Lines | 82% | 80% | ✅ |
| Branches | 68% | 70% | ⚠️ |
| Functions | 91% | 90% | ✅ |

### Uncovered Lines

#### `src/services/orderService.ts`
- Lines 45-48: Error handling for payment timeout
- Lines 72-75: Retry logic branch

#### `src/utils/validation.ts`
- Lines 23-25: Edge case for empty string

### Recommended Additional Tests

1. **Payment timeout handling**
   ```typescript
   it('should retry payment when timeout occurs', async () => {
     mockPayment.charge.mockRejectedValueOnce(new TimeoutError());
     mockPayment.charge.mockResolvedValueOnce({ success: true });

     const result = await orderService.processPayment(order);

     expect(mockPayment.charge).toHaveBeenCalledTimes(2);
     expect(result.status).toBe('completed');
   });
````

2. **Empty string validation**
   ```typescript
   it('should reject empty string as invalid', () => {
     expect(validate('')).toEqual({ valid: false, error: 'Required' });
   });
   ```

````

---

## Integration Commands

```bash
# Generate tests for file
automation test generate src/services/userService.ts

# Generate with specific framework
automation test generate src/api.py --framework pytest

# Analyze coverage gaps
automation test coverage --report gaps

# Run mutation testing
automation test mutate src/core/

# Generate test report
automation test report --format html --output coverage/
````
