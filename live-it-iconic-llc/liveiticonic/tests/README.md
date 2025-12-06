# Tests Directory

This directory contains all test files for the **Live It Iconic** e-commerce platform.

## Directory Structure

```
tests/
├── unit/                      # Unit tests for individual functions/components
│   ├── services/             # Service layer tests
│   ├── components/           # Component tests
│   └── utils/                # Utility function tests
├── integration/              # Integration tests for feature workflows
│   ├── checkout/            # Complete checkout flow tests
│   ├── cart/                # Shopping cart tests
│   └── api/                 # API integration tests
├── e2e/                      # End-to-end tests with Playwright
│   ├── purchase-flow/       # Complete purchase journey
│   ├── admin/               # Admin dashboard workflows
│   └── content/             # Content pages (YouTube, Brand)
├── fixtures/                 # Test data and mock responses
├── helpers/                  # Test utility functions
└── setup/                    # Test environment configuration
```

## Testing Stack

- **Vitest** - Unit and integration testing framework
- **React Testing Library** - Component testing
- **Playwright** - End-to-end testing
- **MSW (Mock Service Worker)** - API mocking
- **Testing Library User Event** - User interaction simulation
- **Stripe Mock** - Payment testing utilities

## Running Tests

### All Tests
```bash
npm test
```

### Unit Tests Only
```bash
npm run test:unit
```

### Integration Tests
```bash
npm run test:integration
```

### End-to-End Tests
```bash
npm run test:e2e
```

### Watch Mode (for development)
```bash
npm run test:watch
```

### Coverage Report
```bash
npm run test:coverage
```

## Writing Tests

### Unit Tests

Test individual functions or components in isolation:

```typescript
// tests/unit/services/productService.test.ts
import { describe, it, expect, beforeEach } from 'vitest';
import { ProductService } from '@/services/productService';

describe('ProductService', () => {
  it('should fetch products from database', async () => {
    const products = await ProductService.getAll();

    expect(products).toBeDefined();
    expect(products.length).toBeGreaterThan(0);
    expect(products[0]).toHaveProperty('name');
    expect(products[0]).toHaveProperty('price');
  });

  it('should filter products by category', async () => {
    const hoodies = await ProductService.getByCategory('hoodies');

    expect(hoodies.every(p => p.category === 'hoodies')).toBe(true);
  });
});
```

### Component Tests

Test React components with user interactions:

```typescript
// tests/unit/components/CartDrawer.test.tsx
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, it, expect } from 'vitest';
import { CartDrawer } from '@/components/CartDrawer';

describe('CartDrawer', () => {
  it('displays cart items with correct totals', () => {
    const items = [
      { id: '1', name: 'Black Hoodie', price: 79, quantity: 2 }
    ];

    render(<CartDrawer items={items} />);

    expect(screen.getByText('Black Hoodie')).toBeInTheDocument();
    expect(screen.getByText('$158.00')).toBeInTheDocument(); // 79 * 2
  });

  it('allows updating item quantity', async () => {
    const user = userEvent.setup();
    const onUpdate = vi.fn();

    render(<CartDrawer items={[...]} onUpdate={onUpdate} />);

    const increaseButton = screen.getByRole('button', { name: /increase/i });
    await user.click(increaseButton);

    expect(onUpdate).toHaveBeenCalledWith(expect.objectContaining({
      quantity: 3
    }));
  });
});
```

### Integration Tests

Test complete workflows across multiple components/services:

```typescript
// tests/integration/checkout/complete-purchase.test.ts
import { describe, it, expect } from 'vitest';
import { CartService } from '@/services/cartService';
import { OrderService } from '@/services/orderService';
import { StripeService } from '@/services/stripeService';

describe('Complete Purchase Flow', () => {
  it('creates order and processes payment successfully', async () => {
    // Add items to cart
    const cart = CartService.create();
    CartService.addItem(cart.id, { productId: 'hoodie-1', quantity: 1 });

    // Create order
    const order = await OrderService.createFromCart(cart.id, {
      shippingAddress: {
        name: 'Test User',
        street: '123 Test St',
        city: 'TestCity',
        state: 'CA',
        zip: '12345'
      }
    });

    expect(order).toBeDefined();
    expect(order.totalAmount).toBe(79.00);
    expect(order.status).toBe('pending');

    // Process payment (using test card)
    const paymentIntent = await StripeService.createPaymentIntent({
      amount: order.totalAmount,
      orderId: order.id
    });

    expect(paymentIntent.status).toBe('requires_payment_method');

    // Simulate payment success
    const result = await StripeService.confirmPayment(paymentIntent.id, {
      payment_method: 'pm_card_visa' // Stripe test card
    });

    expect(result.status).toBe('succeeded');

    // Verify order updated
    const updatedOrder = await OrderService.getById(order.id);
    expect(updatedOrder.status).toBe('paid');
  });
});
```

### End-to-End Tests

Test complete user journeys in a real browser:

```typescript
// tests/e2e/purchase-flow/complete-checkout.spec.ts
import { test, expect } from '@playwright/test';

test('customer can complete full purchase', async ({ page }) => {
  // Visit homepage
  await page.goto('/');

  // Navigate to shop
  await page.click('text=Shop');
  await expect(page).toHaveURL('/shop');

  // Select product
  await page.click('text=Black Hoodie');
  await expect(page).toHaveURL(/\/products\/.+/);

  // Add to cart
  await page.click('button:has-text("Add to Cart")');
  await expect(page.locator('[data-testid="cart-count"]')).toHaveText('1');

  // Open cart
  await page.click('[data-testid="cart-icon"]');
  await expect(page.locator('text=Black Hoodie')).toBeVisible();

  // Proceed to checkout
  await page.click('button:has-text("Checkout")');
  await expect(page).toHaveURL('/checkout');

  // Fill shipping address
  await page.fill('[name="name"]', 'Test Customer');
  await page.fill('[name="email"]', 'test@example.com');
  await page.fill('[name="address"]', '123 Main St');
  await page.fill('[name="city"]', 'Los Angeles');
  await page.selectOption('[name="state"]', 'CA');
  await page.fill('[name="zip"]', '90001');

  // Fill payment details (Stripe test card)
  const stripeFrame = page.frameLocator('iframe[name*="__privateStripeFrame"]');
  await stripeFrame.locator('[name="cardnumber"]').fill('4242424242424242');
  await stripeFrame.locator('[name="exp-date"]').fill('1225');
  await stripeFrame.locator('[name="cvc"]').fill('123');
  await stripeFrame.locator('[name="postal"]').fill('90001');

  // Submit payment
  await page.click('button:has-text("Place Order")');

  // Verify order confirmation
  await expect(page).toHaveURL(/\/order-confirmation\/.+/);
  await expect(page.locator('text=Thank you for your order!')).toBeVisible();
  await expect(page.locator('[data-testid="order-number"]')).toBeVisible();
});
```

## Test Coverage Goals

- **Overall Coverage**: 80%+
- **Services**: 90%+
- **Components**: 75%+
- **Critical Paths**: 100%

### Critical Paths Requiring 100% Coverage

1. **Payment Processing** - Stripe integration, payment intents
2. **Order Creation** - Cart to order conversion
3. **Checkout Flow** - Complete purchase journey
4. **Authentication** - User login/logout, JWT handling
5. **Inventory Management** - Stock tracking, overselling prevention
6. **Webhook Handlers** - Stripe payment events

## Mocking

### API Mocking with MSW

```typescript
// tests/helpers/mocks/handlers.ts
import { rest } from 'msw';

export const handlers = [
  rest.get('/api/products', (req, res, ctx) => {
    return res(
      ctx.status(200),
      ctx.json([
        {
          id: '1',
          name: 'Black Hoodie',
          price: 79,
          sku: 'HOODIE-BLK',
          stockQuantity: 50
        },
        {
          id: '2',
          name: 'Black Cap',
          price: 29,
          sku: 'CAP-BLK',
          stockQuantity: 100
        }
      ])
    );
  }),

  rest.post('/api/orders', (req, res, ctx) => {
    return res(
      ctx.status(201),
      ctx.json({
        id: 'order-123',
        orderNumber: 'LII-001',
        status: 'pending',
        totalAmount: 79
      })
    );
  }),
];
```

### Stripe Mocking

```typescript
// tests/helpers/mocks/stripe.ts
import { vi } from 'vitest';

export const mockStripe = {
  paymentIntents: {
    create: vi.fn(() => Promise.resolve({
      id: 'pi_test_123',
      client_secret: 'pi_test_123_secret',
      status: 'requires_payment_method'
    })),
    confirm: vi.fn(() => Promise.resolve({
      id: 'pi_test_123',
      status: 'succeeded'
    }))
  },
  webhooks: {
    constructEvent: vi.fn((payload, sig, secret) => ({
      type: 'payment_intent.succeeded',
      data: {
        object: {
          id: 'pi_test_123',
          amount: 7900,
          status: 'succeeded'
        }
      }
    }))
  }
};
```

### Supabase Mocking

```typescript
// tests/helpers/mocks/supabase.ts
import { vi } from 'vitest';

export const mockSupabase = {
  from: vi.fn((table) => ({
    select: vi.fn().mockReturnThis(),
    insert: vi.fn().mockReturnThis(),
    update: vi.fn().mockReturnThis(),
    delete: vi.fn().mockReturnThis(),
    eq: vi.fn().mockReturnThis(),
    single: vi.fn().mockResolvedValue({ data: {}, error: null })
  })),
  auth: {
    getUser: vi.fn(() => ({ data: { user: { id: 'user-123' } }, error: null })),
    signIn: vi.fn(() => ({ data: { session: {} }, error: null }))
  }
};
```

## Test Data

Use fixtures for consistent test data:

```typescript
// tests/fixtures/products.ts
export const mockProducts = [
  {
    id: 'prod-1',
    name: 'Black Hoodie',
    description: 'Premium quality hoodie with Live It Iconic branding',
    price: 79,
    cost: 20,
    sku: 'HOODIE-BLK',
    category: 'hoodies',
    imageUrls: ['/images/products/hoodie-black-front.webp'],
    stockQuantity: 50,
    isActive: true
  },
  {
    id: 'prod-2',
    name: 'Black Cap',
    description: 'Adjustable cap with embroidered logo',
    price: 29,
    cost: 8,
    sku: 'CAP-BLK',
    category: 'caps',
    imageUrls: ['/images/products/cap-black-front.webp'],
    stockQuantity: 100,
    isActive: true
  }
];
```

## Best Practices

### 1. Test Behavior, Not Implementation
```typescript
// ❌ Bad - testing implementation details
expect(component.state.loading).toBe(false);

// ✅ Good - testing user-visible behavior
expect(screen.queryByRole('progressbar')).not.toBeInTheDocument();
```

### 2. Use Descriptive Test Names
```typescript
// ❌ Bad
it('works', () => { ... });

// ✅ Good
it('displays error message when credit card is invalid', () => { ... });
```

### 3. Arrange-Act-Assert Pattern
```typescript
it('applies 10% discount code to order total', async () => {
  // Arrange
  const cart = { items: [{ price: 79, quantity: 1 }] };
  const discountCode = 'SAVE10';

  // Act
  const order = await OrderService.createWithDiscount(cart, discountCode);

  // Assert
  expect(order.totalAmount).toBe(71.10); // 79 - 10%
  expect(order.discountApplied).toBe(7.90);
});
```

### 4. Avoid Test Interdependence
```typescript
// ❌ Bad - tests depend on each other
describe('Order Service', () => {
  let orderId: string;

  it('creates order', async () => {
    const order = await OrderService.create(...);
    orderId = order.id; // storing state
  });

  it('completes order', async () => {
    await OrderService.complete(orderId); // depends on previous test
  });
});

// ✅ Good - independent tests
describe('Order Service', () => {
  it('creates order', async () => {
    const order = await OrderService.create(...);
    expect(order.id).toBeDefined();
  });

  it('completes order', async () => {
    const order = await OrderService.create(...);
    const result = await OrderService.complete(order.id);
    expect(result.status).toBe('completed');
  });
});
```

### 5. Clean Up After Tests
```typescript
afterEach(() => {
  vi.clearAllMocks();
  cleanup(); // from @testing-library/react
});

afterAll(() => {
  vi.restoreAllMocks();
});
```

## Stripe Test Cards

For testing payment flows, use Stripe's test cards:

```typescript
// Success
const TEST_CARDS = {
  visa: '4242424242424242',
  mastercard: '5555555555554444',
  amex: '378282246310005'
};

// Decline scenarios
const DECLINE_CARDS = {
  generic_decline: '4000000000000002',
  insufficient_funds: '4000000000009995',
  lost_card: '4000000000009987',
  stolen_card: '4000000000009979'
};

// 3D Secure authentication
const SCA_REQUIRED = '4000002500003155';
```

## Continuous Integration

Tests run automatically on:
- Every commit (pre-commit hook)
- Pull requests (GitHub Actions)
- Before deployment (Vercel)

### GitHub Actions Workflow
```yaml
name: Test
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-node@v3
      - run: npm ci
      - run: npm run test:coverage
      - run: npm run test:e2e
      - uses: codecov/codecov-action@v3
```

## Debugging Tests

### Debug a Single Test
```bash
npm run test:debug -- tests/unit/services/orderService.test.ts
```

### Use `screen.debug()`
```typescript
it('renders component', () => {
  render(<MyComponent />);
  screen.debug(); // prints current DOM
});
```

### Use Playwright Inspector
```bash
npm run test:e2e -- --debug
```

## Performance Testing

For performance-critical code:

```typescript
import { performance } from 'perf_hooks';

it('calculates order total in under 10ms', () => {
  const cart = { items: Array(100).fill({ price: 50, quantity: 1 }) };

  const start = performance.now();
  const total = OrderService.calculateTotal(cart);
  const end = performance.now();

  expect(end - start).toBeLessThan(10);
  expect(total).toBe(5000);
});
```

## Key Test Scenarios

### E-Commerce Critical Paths

1. **Complete Purchase Flow**
   - Browse products → Add to cart → Checkout → Payment → Confirmation

2. **Cart Management**
   - Add items → Update quantity → Remove items → Apply discount

3. **Payment Failures**
   - Insufficient funds → Error handling → Retry logic

4. **Inventory Management**
   - Low stock warning → Out of stock prevention → Admin restocking

5. **Order Fulfillment**
   - Admin marks shipped → Tracking email sent → Customer receives update

### Edge Cases

- Empty cart checkout attempt
- Invalid discount codes
- Expired payment cards
- Network failures during checkout
- Concurrent stock purchases (race conditions)
- Webhook delivery failures

## Resources

- [Vitest Documentation](https://vitest.dev/)
- [React Testing Library](https://testing-library.com/react)
- [Playwright Documentation](https://playwright.dev/)
- [Stripe Testing](https://stripe.com/docs/testing)
- [MSW Documentation](https://mswjs.io/)

---

For more details, see:
- [E2E Testing Guide](../docs/guides/testing-guide.md)
- [Contributing Guide](../CONTRIBUTING.md)
- [CI/CD Documentation](../docs/deployment/deployment-guide.md)
