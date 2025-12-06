import { Page, expect } from '@playwright/test';
import { ShopPage } from '../pages/ShopPage';
import { CartPage } from '../pages/CartPage';
import { ProductDetailPage } from '../pages/ProductDetailPage';

/**
 * Add a product to cart from shop page
 */
export async function addProductToCart(page: Page, productIndex: number = 0) {
  const shopPage = new ShopPage(page);
  await shopPage.goto();

  // Get product count to ensure products are available
  const productCount = await shopPage.getProductCount();
  if (productCount === 0) {
    throw new Error('No products available in shop');
  }

  if (productIndex >= productCount) {
    throw new Error(`Product index ${productIndex} out of range (${productCount} products available)`);
  }

  await shopPage.clickProductAt(productIndex);

  const productPage = new ProductDetailPage(page);
  await productPage.addToCart();
}

/**
 * Add multiple products to cart
 */
export async function addMultipleProductsToCart(page: Page, count: number = 2) {
  const shopPage = new ShopPage(page);
  await shopPage.goto();

  const productCount = await shopPage.getProductCount();
  if (productCount < count) {
    throw new Error(`Insufficient products (${productCount}) to add ${count} items`);
  }

  for (let i = 0; i < count; i++) {
    await shopPage.clickProductAt(i);
    const productPage = new ProductDetailPage(page);
    await productPage.addToCart();
    await shopPage.goto();
  }
}

/**
 * Clear cart completely
 */
export async function clearCart(page: Page) {
  const cartPage = new CartPage(page);
  await cartPage.goto();

  let itemCount = await cartPage.getItemCount();
  while (itemCount > 0) {
    await cartPage.removeFirstItem();
    itemCount = await cartPage.getItemCount();
  }

  // Verify cart is empty
  const isEmpty = await cartPage.isCartEmpty();
  if (!isEmpty && itemCount === 0) {
    // Sometimes the empty message doesn't show until after reload
    await page.reload();
  }
}

/**
 * Login user (mock implementation)
 */
export async function login(page: Page, email: string, password: string) {
  await page.goto('/auth/signin');

  const emailInput = page.getByLabel(/email/i);
  const passwordInput = page.getByLabel(/password/i);
  const signInButton = page.getByRole('button', { name: /sign in/i });

  await emailInput.fill(email);
  await passwordInput.fill(password);
  await signInButton.click();

  // Wait for navigation away from login page
  await page.waitForURL((url) => !url.toString().includes('/auth/signin'));
}

/**
 * Logout user
 */
export async function logout(page: Page) {
  const logoutButton = page.getByRole('button', { name: /logout|sign out/i });

  if (await logoutButton.isVisible()) {
    await logoutButton.click();
  } else {
    // Try using menu if direct logout button not visible
    const menuButton = page.locator('[data-testid="user-menu"], [aria-label*="menu" i]').first();
    if (await menuButton.isVisible()) {
      await menuButton.click();
      const logoutOption = page.getByRole('menuitem', { name: /logout/i });
      if (await logoutOption.isVisible()) {
        await logoutOption.click();
      }
    }
  }
}

/**
 * Wait for API response
 */
export async function waitForAPIResponse(page: Page, pattern: string | RegExp) {
  return await page.waitForResponse(
    (response) => {
      const url = response.url();
      if (typeof pattern === 'string') {
        return url.includes(pattern);
      }
      return pattern.test(url);
    }
  );
}

/**
 * Check if element is in viewport
 */
export async function isElementInViewport(page: Page, selector: string): Promise<boolean> {
  return await page.evaluate((sel) => {
    const element = document.querySelector(sel);
    if (!element) return false;
    const rect = element.getBoundingClientRect();
    return (
      rect.top >= 0 &&
      rect.left >= 0 &&
      rect.bottom <= (window.innerHeight || document.documentElement.clientHeight) &&
      rect.right <= (window.innerWidth || document.documentElement.clientWidth)
    );
  }, selector);
}

/**
 * Scroll to element
 */
export async function scrollToElement(page: Page, selector: string) {
  await page.locator(selector).scrollIntoViewIfNeeded();
}

/**
 * Wait for element to be stable (not moving)
 */
export async function waitForElementStable(page: Page, selector: string, timeout: number = 5000) {
  const locator = page.locator(selector);
  await locator.waitFor({ state: 'attached' });

  // Give animations time to complete
  await page.waitForTimeout(Math.min(500, timeout / 10));
}

/**
 * Take a screenshot with descriptive name
 */
export async function takeScreenshot(page: Page, name: string) {
  const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
  await page.screenshot({ path: `test-results/screenshots/${timestamp}-${name}.png` });
}

/**
 * Check accessibility
 */
export async function checkAccessibility(page: Page) {
  // Check for basic a11y issues
  const results = await page.evaluate(() => {
    const issues = [];

    // Check for images without alt text
    const imagesWithoutAlt = document.querySelectorAll('img:not([alt])');
    if (imagesWithoutAlt.length > 0) {
      issues.push(`Found ${imagesWithoutAlt.length} images without alt text`);
    }

    // Check for buttons without text or aria-label
    const badButtons = Array.from(document.querySelectorAll('button')).filter(btn =>
      !btn.textContent?.trim() && !btn.getAttribute('aria-label')
    );
    if (badButtons.length > 0) {
      issues.push(`Found ${badButtons.length} buttons without text or aria-label`);
    }

    // Check for links without text or aria-label
    const badLinks = Array.from(document.querySelectorAll('a')).filter(link =>
      !link.textContent?.trim() && !link.getAttribute('aria-label')
    );
    if (badLinks.length > 0) {
      issues.push(`Found ${badLinks.length} links without text or aria-label`);
    }

    return issues;
  });

  return results;
}

/**
 * Generate test data
 */
export function generateTestData() {
  const timestamp = Date.now();
  return {
    email: `test.${timestamp}@example.com`,
    password: 'TestPassword123!',
    firstName: 'Test',
    lastName: 'User',
    address: '123 Test Street',
    city: 'Test City',
    state: 'TS',
    zip: '12345',
    country: 'United States',
    phone: '+1 (555) 123-4567',
  };
}

/**
 * Wait for cart badge to update
 */
export async function waitForCartBadgeUpdate(page: Page, expectedCount: number, timeout: number = 5000) {
  const cartBadge = page.locator('[data-testid="cart-badge"]');
  await cartBadge.waitFor({ state: 'visible', timeout });
  await expect(cartBadge).toHaveText(expectedCount.toString(), { timeout });
}

/**
 * Verify page title
 */
export async function verifyPageTitle(page: Page, expectedTitle: string | RegExp) {
  if (typeof expectedTitle === 'string') {
    await expect(page).toHaveTitle(new RegExp(expectedTitle));
  } else {
    await expect(page).toHaveTitle(expectedTitle);
  }
}

/**
 * Check if page has no console errors
 */
export async function checkConsoleErrors(page: Page): Promise<string[]> {
  const errors: string[] = [];

  page.on('console', (msg) => {
    if (msg.type() === 'error') {
      errors.push(msg.text());
    }
  });

  return errors;
}
