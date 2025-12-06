# E2E Testing Guide - Live It Iconic

## Overview

This guide provides comprehensive documentation for the Playwright-based End-to-End (E2E) testing infrastructure for the Live It Iconic e-commerce platform.

## Table of Contents

1. [Installation](#installation)
2. [Project Structure](#project-structure)
3. [Running Tests](#running-tests)
4. [Writing Tests](#writing-tests)
5. [Page Objects](#page-objects)
6. [Test Utilities](#test-utilities)
7. [CI/CD Integration](#cicd-integration)
8. [Debugging](#debugging)
9. [Best Practices](#best-practices)
10. [Troubleshooting](#troubleshooting)

## Installation

### Prerequisites

- Node.js 18+ installed
- npm or yarn package manager
- Git

### Setup

Playwright is already included in the project dependencies. To ensure browsers are installed:

```bash
# Install Playwright browsers
npx playwright install

# Install browsers for a specific browser
npx playwright install chromium
npx playwright install firefox
npx playwright install webkit
```

## Project Structure

```
tests/e2e/
├── pages/              # Page Object Models
│   ├── HomePage.ts
│   ├── ShopPage.ts
│   ├── ProductDetailPage.ts
│   ├── CartPage.ts
│   ├── CheckoutPage.ts
│   └── AdminPage.ts
├── fixtures/           # Test Fixtures
│   └── auth.fixture.ts
├── utils/              # Utility Functions
│   └── test-helpers.ts
└── specs/              # Test Specifications
    ├── user-journey.spec.ts
    ├── purchase-flow.spec.ts
    ├── cart-management.spec.ts
    └── admin-dashboard.spec.ts

playwright.config.ts    # Playwright Configuration
```

### Key Files

| File | Purpose |
|------|---------|
| `playwright.config.ts` | Main Playwright configuration with browser settings |
| `tests/e2e/pages/*` | Page Object Models for UI interaction |
| `tests/e2e/fixtures/auth.fixture.ts` | Test fixtures with page objects |
| `tests/e2e/utils/test-helpers.ts` | Reusable helper functions |
| `tests/e2e/specs/*` | Test specifications organized by feature |

## Running Tests

### Basic Commands

```bash
# Run all E2E tests
npm run test:e2e

# Run tests with UI (interactive mode)
npm run test:e2e:ui

# Run tests in debug mode (step through code)
npm run test:e2e:debug

# Run tests in headed mode (see browser)
npm run test:e2e:headed

# Show test report
npm run test:e2e:report
```

### Browser-Specific Tests

```bash
# Run tests on Chromium only
npm run test:e2e:chromium

# Run tests on Firefox only
npm run test:e2e:firefox

# Run tests on WebKit (Safari) only
npm run test:e2e:webkit

# Run tests on mobile devices
npm run test:e2e:mobile
```

### Running Specific Test Files

```bash
# Run single test file
npx playwright test tests/e2e/specs/user-journey.spec.ts

# Run tests matching pattern
npx playwright test -g "cart"

# Run specific test by name
npx playwright test -g "user can add item to cart"
```

### Running Tests in Parallel vs Sequential

```bash
# Run all tests in parallel (default)
npm run test:e2e

# Run tests serially (one at a time)
npx playwright test --workers=1

# Run with custom number of workers
npx playwright test --workers=4
```

## Writing Tests

### Basic Test Structure

```typescript
import { test, expect } from '../fixtures/auth.fixture';

test.describe('Feature Name', () => {
  test('should perform some action', async ({ page, homePage }) => {
    // Arrange
    await homePage.goto();

    // Act
    await homePage.clickShop();

    // Assert
    await expect(page).toHaveURL(/.*shop/);
  });

  test('should handle user interaction', async ({ shopPage }) => {
    await shopPage.goto();
    const count = await shopPage.getProductCount();
    expect(count).toBeGreaterThan(0);
  });
});
```

### Using Fixtures

Tests can use pre-defined page objects through fixtures:

```typescript
test('complete purchase flow', async ({ homePage, shopPage, cartPage, checkoutPage }) => {
  // All page objects are automatically initialized
  await homePage.goto();
  await homePage.clickShop();
  // ...
});
```

### Using Test Utilities

```typescript
import { addProductToCart, generateTestData, clearCart } from '../utils/test-helpers';

test('test with utilities', async ({ page }) => {
  // Add product to cart
  await addProductToCart(page, 0);

  // Generate test data
  const data = generateTestData();
  console.log(data.email); // test.1234567890@example.com

  // Clear cart
  await clearCart(page);
});
```

### Common Assertions

```typescript
import { expect } from '../fixtures/auth.fixture';

// Visibility
await expect(page.locator('.button')).toBeVisible();
await expect(page.locator('.hidden')).not.toBeVisible();

// Text content
await expect(page.locator('h1')).toContainText('Welcome');
await expect(page.locator('h1')).toHaveText('Exact Text');

// URL
await expect(page).toHaveURL(/.*shop/);
await expect(page).toHaveURL('http://localhost:5173/shop');

// Values
await expect(page.locator('input')).toHaveValue('typed text');
await expect(page.locator('select')).toHaveValue('option1');

// Attributes
await expect(page.locator('button')).toHaveAttribute('aria-label', 'Add to cart');

// Counts
await expect(page.locator('li')).toHaveCount(3);
await expect(page.locator('.item')).toHaveCount(n => n > 5);
```

## Page Objects

### Page Object Benefits

- Encapsulates element selectors
- Provides reusable methods
- Makes tests more maintainable
- Improves readability

### Example: HomePage

```typescript
import { Page, Locator } from '@playwright/test';

export class HomePage {
  readonly page: Page;
  readonly heroTitle: Locator;
  readonly shopButton: Locator;

  constructor(page: Page) {
    this.page = page;
    this.heroTitle = page.locator('h1');
    this.shopButton = page.getByRole('link', { name: /shop/i });
  }

  async goto() {
    await this.page.goto('/');
  }

  async clickShop() {
    await this.shopButton.click();
  }
}
```

### Available Page Objects

#### HomePage
- `goto()` - Navigate to home page
- `clickShop()` - Click shop button
- `navigateToAbout()` - Navigate to about page
- `openCart()` - Open shopping cart
- `getCartCount()` - Get number of items in cart

#### ShopPage
- `goto()` - Navigate to shop
- `getProductCount()` - Get total products displayed
- `searchProducts(query)` - Search for products
- `clickProductAt(index)` - Click specific product
- `addFirstProductToCart()` - Add first product
- `filterByCategory(category)` - Filter by category
- `sortBy(option)` - Sort products
- `getFirstProductName()` - Get first product name
- `getFirstProductPrice()` - Get first product price

#### CartPage
- `goto()` - Navigate to cart
- `getItemCount()` - Get number of items
- `isCartEmpty()` - Check if cart is empty
- `removeFirstItem()` - Remove first item
- `increaseQuantityAt(index)` - Increase quantity
- `decreaseQuantityAt(index)` - Decrease quantity
- `setQuantityAt(index, qty)` - Set specific quantity
- `getCartTotal()` - Get total price
- `applyPromoCode(code)` - Apply promotion code
- `proceedToCheckout()` - Go to checkout

#### CheckoutPage
- `goto()` - Navigate to checkout
- `fillShippingInfo(info)` - Fill shipping form
- `fillPaymentInfo(info)` - Fill payment form
- `selectShippingMethod(method)` - Select shipping
- `continueToPaymentStep()` - Continue to payment
- `placeOrder()` - Place order
- `setSameBillingAddress(same)` - Set billing address same as shipping

#### ProductDetailPage
- `getProductTitle()` - Get product name
- `getProductPrice()` - Get product price
- `selectSize(size)` - Select size variant
- `selectColor(color)` - Select color variant
- `setQuantity(qty)` - Set quantity
- `addToCart()` - Add product to cart
- `getRating()` - Get product rating

#### AdminPage
- `goto()` - Navigate to admin dashboard
- `gotoOrders()` - Go to orders section
- `gotoProducts()` - Go to products section
- `gotoUsers()` - Go to users section
- `gotoAnalytics()` - Go to analytics
- `searchFor(query)` - Search items
- `filterBy(value)` - Filter results
- `editItemAt(index)` - Edit item
- `deleteItemAt(index)` - Delete item
- `isDashboardVisible()` - Check if dashboard visible

## Test Utilities

### Available Helpers

#### Product Management
```typescript
// Add single product to cart
await addProductToCart(page, 0); // First product
await addProductToCart(page, 1); // Second product

// Add multiple products
await addMultipleProductsToCart(page, 3); // Add 3 products
```

#### Cart Operations
```typescript
// Clear entire cart
await clearCart(page);
```

#### Authentication
```typescript
// Login user
await login(page, 'user@example.com', 'password123');

// Logout user
await logout(page);
```

#### Waiting and Timing
```typescript
// Wait for API response
const response = await waitForAPIResponse(page, '/api/products');
const response2 = await waitForAPIResponse(page, /.*products.*/);

// Wait for element to be stable
await waitForElementStable(page, '.loading-spinner');

// Wait for cart badge update
await waitForCartBadgeUpdate(page, 5); // Wait for count to be 5
```

#### DOM Operations
```typescript
// Check if element is in viewport
const visible = await isElementInViewport(page, '.item');

// Scroll to element
await scrollToElement(page, '.checkout-button');
```

#### Testing Utilities
```typescript
// Generate test data
const data = generateTestData();
// Returns: { email, password, firstName, lastName, address, etc. }

// Take screenshot
await takeScreenshot(page, 'checkout-form');

// Check accessibility issues
const a11yIssues = await checkAccessibility(page);

// Check for console errors
const errors = await checkConsoleErrors(page);

// Verify page title
await verifyPageTitle(page, 'Shop');
await verifyPageTitle(page, /Products/);
```

## CI/CD Integration

### GitHub Actions Configuration

The E2E tests are configured to run in CI environments. Key settings:

```typescript
// From playwright.config.ts
forbidOnly: !!process.env.CI,        // Forbid .only in CI
retries: process.env.CI ? 2 : 0,     // Retry failed tests in CI
workers: process.env.CI ? 1 : undefined // Run serially in CI
```

### Running in CI

```bash
# Tests will automatically:
# - Run with retries (2x)
# - Run serially (1 worker)
# - Generate HTML reports
# - Capture traces on failures
CI=true npm run test:e2e
```

## Debugging

### Debug Mode

```bash
# Step through tests line by line
npm run test:e2e:debug

# Commands in debug mode:
# - Step over: f10
# - Step into: f11
# - Step out: shift+f11
# - Resume: f8 or click play
# - Exit: ctrl+c
```

### Headed Mode

```bash
# Watch tests run in real browser
npm run test:e2e:headed

# Useful for:
# - Seeing what tests do
# - Understanding failures
# - Visual debugging
```

### UI Mode

```bash
# Interactive test explorer
npm run test:e2e:ui

# Features:
# - Watch tests in real-time
# - Inspect elements
# - Step through tests
# - See debug logs
```

### Inspecting Elements

```typescript
// Pause and inspect element in browser DevTools
await page.pause();

// Or use inspector
npx playwright codegen http://localhost:5173
```

### Viewing Reports

```bash
# Generate and view HTML report
npm run test:e2e:report

# Report includes:
# - Test timeline
# - Screenshots
# - Video recordings
# - Traces
```

### Taking Screenshots and Videos

Screenshots and videos are automatically captured on failure:

```typescript
// Force screenshot
await page.screenshot({ path: 'screenshot.png' });

// Record video (configured in playwright.config.ts)
// - Automatically saves on failure
```

## Best Practices

### 1. Use Page Objects

```typescript
// Good
const shopPage = new ShopPage(page);
await shopPage.goto();
await shopPage.clickProductAt(0);

// Avoid
await page.goto('/shop');
await page.locator('[data-testid="product"]').first().click();
```

### 2. Use Data Attributes for Selectors

```typescript
// Good (stable)
await page.locator('[data-testid="add-to-cart"]').click();

// Avoid (brittle)
await page.locator('button:nth-child(3)').click();
await page.locator('.btn-primary').click();
```

### 3. Wait for Network to Stabilize

```typescript
// Good
await page.waitForLoadState('networkidle');

// Good (specific)
await expect(productList).toBeVisible();

// Avoid
await page.waitForTimeout(2000); // Hard waits
```

### 4. Use Role-Based Selectors

```typescript
// Good (semantic)
await page.getByRole('button', { name: /add/i }).click();
await page.getByRole('link', { name: /shop/i }).click();

// Better for accessibility testing
```

### 5. Keep Tests Independent

```typescript
// Good (self-contained)
test('should add product to cart', async ({ page }) => {
  await page.goto('/shop');
  // ... add product
});

// Avoid (depends on previous test state)
```

### 6. Use Descriptive Test Names

```typescript
// Good
test('user can add item to cart from product detail page', async () => {});

// Avoid
test('test 1', async () => {});
test('add', async () => {});
```

### 7. Organize Tests Logically

```typescript
test.describe('Cart Management', () => {
  test.describe('Adding Items', () => {
    test('single item', async () => {});
    test('multiple items', async () => {});
  });

  test.describe('Removing Items', () => {
    test('first item', async () => {});
    test('all items', async () => {});
  });
});
```

### 8. Use Test Hooks

```typescript
test.describe('Feature', () => {
  test.beforeEach(async ({ page }) => {
    // Setup before each test
    await page.goto('/');
  });

  test.afterEach(async ({ page }) => {
    // Cleanup after each test
    // (automatic in most cases)
  });
});
```

## Troubleshooting

### Common Issues

#### Tests Timeout
```
Error: Timeout 30000ms exceeded
```

**Solution:**
```typescript
// Increase timeout for specific test
test('slow operation', async ({ page }) => {
  test.setTimeout(60000);
  // ...
});

// Or in config:
// timeout: 60000
```

#### Element Not Found
```
Error: locator.click: No element matches selector
```

**Solution:**
1. Verify selector is correct
2. Check element is visible
3. Wait for element to appear

```typescript
// Good
await page.locator('[data-testid="button"]').waitFor({ state: 'visible' });
await page.locator('[data-testid="button"]').click();
```

#### Flaky Tests
Tests pass sometimes, fail other times

**Solution:**
1. Remove hard waits (`waitForTimeout`)
2. Use explicit waits (`waitFor()`, `toBeVisible()`)
3. Wait for network idle
4. Avoid race conditions

```typescript
// Good
await expect(page.locator('[data-testid="item"]')).toBeVisible();

// Avoid
await page.waitForTimeout(1000);
```

#### Port Already in Use

```
Error: connect EADDRINUSE :::5173
```

**Solution:**
```bash
# Kill process on port 5173
lsof -i :5173
kill -9 <PID>

# Or use different port
PORT=3000 npm run dev
```

#### Browser Installation Issues

```bash
# Reinstall browsers
npx playwright install --with-deps

# Or specific browser
npx playwright install chromium --with-deps
```

### Getting Help

- **Playwright Docs**: https://playwright.dev
- **Playwright Inspector**: `npx playwright codegen`
- **GitHub Issues**: Check the Playwright repository
- **Community**: Playwright Discord/GitHub discussions

## Test Statistics

### Current Coverage

- **Total Test Files**: 4
- **Total Test Cases**: 40+
- **Page Objects**: 6
- **Helper Functions**: 15+
- **Supported Browsers**: 5 (Chromium, Firefox, WebKit, Mobile Chrome, Mobile Safari)

### Breakdown by Test Suite

| Suite | Tests | Coverage |
|-------|-------|----------|
| User Journey | 7 | Navigation, search, filters, responsive design |
| Purchase Flow | 8 | Product addition, checkout, validation |
| Cart Management | 10 | Add/remove/update, persistence, mobile |
| Admin Dashboard | 15+ | Navigation, CRUD operations, accessibility |

## Maintenance

### Keeping Tests Updated

1. **Update Page Objects** when UI changes
2. **Update Selectors** if they become invalid
3. **Add Tests** for new features
4. **Remove Tests** for deprecated features
5. **Update Fixtures** if authentication changes

### Monitoring Test Health

```bash
# Check test results history
npm run test:e2e:report

# Look for:
# - Consistently failing tests
# - Slow tests
# - Flaky tests (intermittent failures)
```

## Next Steps

1. **Run a test**: `npm run test:e2e`
2. **View results**: `npm run test:e2e:report`
3. **Add new tests**: Create files in `tests/e2e/specs/`
4. **Debug failures**: `npm run test:e2e:debug`

## Support

For questions or issues with E2E tests:
1. Check this guide
2. Review Playwright documentation
3. Check GitHub issues
4. Ask in team chat

---

**Last Updated**: November 2024
**Playwright Version**: ^1.44.0
**Node Version**: 18+
