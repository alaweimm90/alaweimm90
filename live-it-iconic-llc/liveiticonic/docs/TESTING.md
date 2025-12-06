# Testing Guide - Live It Iconic

## Overview

This project uses **Vitest** for unit testing and **Testing Library** for component testing. The test suite provides comprehensive coverage of components, hooks, utilities, and business logic.

## Setup

### Testing Infrastructure

- **Test Framework**: Vitest v2.1.8
- **Component Testing**: @testing-library/react v16
- **Utilities**: @testing-library/user-event, @testing-library/jest-dom
- **Environment**: jsdom

### Configuration Files

- **vitest.config.ts** - Vitest configuration with coverage settings
- **src/test/setup.ts** - Global test setup and custom matchers
- **src/test/test-utils.tsx** - Custom render function with providers
- **src/test/mocks.ts** - Reusable mock data for tests

## Running Tests

### Run all tests
```bash
npm test
```

### Run tests in watch mode
```bash
npm test -- --watch
```

### Run tests with UI
```bash
npm test:ui
```

### Run tests with coverage
```bash
npm run test:coverage
```

### Run specific test file
```bash
npm test -- src/components/Hero.test.tsx
```

### Run tests matching pattern
```bash
npm test -- --grep "Button"
```

### Run tests with specific reporter
```bash
npm test -- --reporter=verbose
```

## Coverage Goals

| Category | Target |
|----------|--------|
| Overall | 80%+ |
| Critical Components | 90%+ |
| UI Components | 85%+ |
| Business Logic (Hooks, Contexts) | 95%+ |
| Utilities | 100% |

## Test File Organization

Tests are organized following the source code structure:

```
src/
├── components/
│   ├── Hero.tsx
│   ├── Hero.test.tsx
│   ├── Navigation.test.tsx
│   ├── ProductCard.test.tsx
│   ├── CartDrawer.test.tsx
│   ├── CartIcon.test.tsx
│   ├── ProductGrid.test.tsx
│   ├── InventoryBadge.test.tsx
│   ├── checkout/
│   │   ├── ShippingForm.tsx
│   │   └── ShippingForm.test.tsx
│   └── ui/
│       ├── button.test.tsx
│       ├── input.test.tsx
│       └── dialog.test.tsx
├── contexts/
│   ├── CartContext.tsx
│   └── CartContext.test.tsx
├── hooks/
│   ├── useLocalStorage.ts
│   └── useLocalStorage.test.ts
├── lib/
│   ├── utils.ts
│   └── utils.test.ts
└── test/
    ├── setup.ts
    ├── test-utils.tsx
    └── mocks.ts
```

## Test Patterns

### Component Testing

Components are tested using the render function from `test-utils.tsx`, which includes all necessary providers:

```typescript
import { render, screen } from '@/test/test-utils';
import userEvent from '@testing-library/user-event';

describe('MyComponent', () => {
  it('renders correctly', () => {
    render(<MyComponent />);
    expect(screen.getByText('Expected text')).toBeInTheDocument();
  });

  it('handles user interactions', async () => {
    const user = userEvent.setup();
    render(<MyComponent />);

    const button = screen.getByRole('button');
    await user.click(button);

    expect(screen.getByText('Updated text')).toBeInTheDocument();
  });
});
```

### Hook Testing

Hooks are tested using `renderHook` with the `act` utility:

```typescript
import { renderHook, act } from '@testing-library/react';
import { useCart } from '@/contexts/CartContext';

describe('useCart', () => {
  it('adds item to cart', () => {
    const { result } = renderHook(() => useCart(), { wrapper });

    act(() => {
      result.current.addItem(product);
    });

    expect(result.current.items).toHaveLength(1);
  });
});
```

### Utility Function Testing

Pure utility functions are tested directly without any special setup:

```typescript
import { cn } from '@/lib/utils';

describe('cn()', () => {
  it('combines class names', () => {
    const result = cn('px-2', 'py-1');
    expect(result).toBe('px-2 py-1');
  });
});
```

## Testing Best Practices

### 1. User-Centric Testing
Test from the user's perspective, not implementation details:

```typescript
// Good
const button = screen.getByRole('button', { name: /Submit/i });
await user.click(button);

// Avoid
const button = container.querySelector('.submit-btn');
fireEvent.click(button);
```

### 2. Accessibility Testing
Ensure components are accessible:

```typescript
it('has proper accessibility attributes', () => {
  render(<Component />);

  const button = screen.getByRole('button');
  expect(button).toHaveAttribute('aria-label', 'Expected label');
});
```

### 3. Arrange-Act-Assert Pattern
Structure tests clearly with three distinct phases:

```typescript
it('updates cart total when quantity changes', () => {
  // Arrange
  const { result } = renderHook(() => useCart(), { wrapper });
  const product = mockProduct;

  // Act
  act(() => {
    result.current.addItem(product);
    result.current.updateQuantity(product.id, 5);
  });

  // Assert
  expect(result.current.total).toBe(product.price * 5);
});
```

### 4. Mock Data
Use mock data from `src/test/mocks.ts`:

```typescript
import { mockProduct, mockCartItems } from '@/test/mocks';

it('displays cart items', () => {
  render(<CartDrawer items={mockCartItems} />);
  // Test with consistent, reusable mock data
});
```

### 5. Async Operations
Always use async/await with userEvent:

```typescript
it('submits form', async () => {
  const user = userEvent.setup();
  render(<Form />);

  const input = screen.getByRole('textbox');
  await user.type(input, 'text');

  const button = screen.getByRole('button', { name: /Submit/i });
  await user.click(button);
});
```

## Coverage Reports

After running `npm run test:coverage`, a detailed HTML report is generated in the `coverage/` directory.

To view the report:
```bash
open coverage/index.html
```

Coverage thresholds are configured in `vitest.config.ts`:
- Lines: 80%
- Functions: 80%
- Branches: 75%
- Statements: 80%

## Tier-Based Testing Approach

### Tier 1: Critical Components (90%+ coverage)
- Hero.tsx
- Navigation.tsx
- ProductCard.tsx
- CartDrawer.tsx
- ShippingForm.tsx

### Tier 2: UI Components (85%+ coverage)
- Button
- Input
- Dialog

### Tier 3: Business Logic (95%+ coverage)
- CartContext
- useLocalStorage hook

### Tier 4: Utilities (100% coverage)
- cn() classname utility
- Other helper functions

## Common Issues and Solutions

### Issue: "Cannot find module '@/...'"
**Solution**: Ensure the path alias is configured correctly in both `vite.config.ts` and `vitest.config.ts`

### Issue: "localStorage is not defined"
**Solution**: Tests already include jsdom environment. Clear localStorage in `beforeEach`:
```typescript
beforeEach(() => {
  localStorage.clear();
});
```

### Issue: Component not rendering in test
**Solution**: Ensure component is wrapped with necessary providers via `test-utils.tsx`

### Issue: userEvent.setup() not working
**Solution**: Make sure the test function is `async`:
```typescript
it('does something', async () => {
  const user = userEvent.setup();
  // Use await with user interactions
});
```

## Adding New Tests

When adding a new component or feature:

1. Create a `ComponentName.test.tsx` file in the same directory
2. Import from `test-utils`:
   ```typescript
   import { render, screen } from '@/test/test-utils';
   import userEvent from '@testing-library/user-event';
   ```
3. Add mock data to `src/test/mocks.ts` if needed
4. Write tests following the patterns above
5. Aim for the appropriate coverage tier
6. Run `npm run test:coverage` to verify coverage

## CI/CD Integration

Tests are configured to run on:
- Pre-commit hooks (if configured)
- Pull request checks
- Deployment pipelines

Ensure all tests pass and coverage thresholds are met before merging:
```bash
npm test && npm run test:coverage
```

## Resources

- [Vitest Documentation](https://vitest.dev/)
- [Testing Library Documentation](https://testing-library.com/)
- [React Testing Library](https://testing-library.com/react)
- [Vitest Coverage Guide](https://vitest.dev/guide/coverage.html)

## Support

For test-related questions or issues:
1. Check existing test examples in `src/components/`
2. Review the Testing Library documentation
3. Check vitest configuration in `vitest.config.ts`
