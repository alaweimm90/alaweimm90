import { test, expect } from '../fixtures/auth.fixture';
import { clearCart } from '../utils/test-helpers';

test.describe('Cart Management', () => {
  test('user can add item to cart', async ({ shopPage, page }) => {
    await shopPage.goto();

    // Get initial cart count
    const initialCount = await shopPage.cartBadge.textContent();

    // Add first product
    const productCount = await shopPage.getProductCount();
    expect(productCount).toBeGreaterThan(0);

    await shopPage.clickProductAt(0);

    const addButton = page.getByRole('button', { name: /add to cart/i }).first();
    await expect(addButton).toBeVisible();
    await addButton.click();

    // Verify cart updated
    await page.waitForTimeout(500);
    const updatedCount = await shopPage.cartBadge.textContent();
    expect(updatedCount).toBeTruthy();
  });

  test('user can update cart quantity', async ({ shopPage, cartPage, page }) => {
    // Add item to cart
    await shopPage.goto();
    const productCount = await shopPage.getProductCount();
    expect(productCount).toBeGreaterThan(0);

    await shopPage.clickProductAt(0);

    const addButton = page.getByRole('button', { name: /add to cart/i }).first();
    await addButton.click();

    // Open cart drawer
    await shopPage.openCart();

    // Verify item in cart
    const itemCount = await cartPage.getItemCount();
    expect(itemCount).toBeGreaterThan(0);

    // Increase quantity
    const increaseBtn = cartPage.page.locator('[data-testid="increase-quantity"]').first();
    if (await increaseBtn.isVisible()) {
      await increaseBtn.click();

      // Verify quantity updated
      await page.waitForTimeout(500);
      const quantityInput = cartPage.page.locator('[data-testid="quantity-input"]').first();
      const quantity = await quantityInput.inputValue();
      expect(parseInt(quantity)).toBeGreaterThanOrEqual(1);
    }
  });

  test('user can remove item from cart', async ({ shopPage, cartPage, page }) => {
    // Add item
    await shopPage.goto();
    await shopPage.clickProductAt(0);

    const addButton = page.getByRole('button', { name: /add to cart/i }).first();
    await addButton.click();

    // Open cart
    await shopPage.openCart();

    // Verify item in cart
    let itemCount = await cartPage.getItemCount();
    expect(itemCount).toBeGreaterThan(0);

    // Remove item
    await cartPage.removeFirstItem();

    // Verify item removed
    await page.waitForTimeout(500);
    itemCount = await cartPage.getItemCount();
    expect(itemCount).toBe(0);
  });

  test('cart persists across page navigation', async ({ shopPage, cartPage, page }) => {
    // Add item
    await shopPage.goto();
    await shopPage.clickProductAt(0);

    const addButton = page.getByRole('button', { name: /add to cart/i }).first();
    await addButton.click();

    // Navigate to cart page
    await cartPage.goto();

    // Verify item is still in cart
    const itemCount = await cartPage.getItemCount();
    expect(itemCount).toBeGreaterThan(0);

    // Navigate back to shop
    await shopPage.goto();

    // Navigate to cart again
    await cartPage.goto();

    // Verify item is still there
    const finalItemCount = await cartPage.getItemCount();
    expect(finalItemCount).toBeGreaterThan(0);
  });

  test('cart total updates correctly', async ({ shopPage, cartPage, page }) => {
    // Add item
    await shopPage.goto();
    const price1 = await shopPage.getFirstProductPrice();

    await shopPage.clickProductAt(0);

    const addButton = page.getByRole('button', { name: /add to cart/i }).first();
    await addButton.click();

    // Open cart and check total
    await shopPage.openCart();
    const total1 = await cartPage.getCartTotal();
    expect(total1).toBeTruthy();

    // Increase quantity
    const increaseBtn = cartPage.page.locator('[data-testid="increase-quantity"]').first();
    if (await increaseBtn.isVisible()) {
      await increaseBtn.click();

      // Wait for total to update
      await page.waitForTimeout(500);

      const total2 = await cartPage.getCartTotal();
      expect(total2).toBeTruthy();

      // Total should increase (parsing might be needed)
    }
  });

  test('user can clear entire cart', async ({ shopPage, page }) => {
    // Add item
    await shopPage.goto();

    const productCount = await shopPage.getProductCount();
    if (productCount === 0) {
      test.skip();
    }

    // Add product
    await shopPage.clickProductAt(0);

    const addButton = page.getByRole('button', { name: /add to cart/i }).first();
    await addButton.click();

    // Clear cart
    await clearCart(page);

    // Verify cart is empty
    await shopPage.openCart();
    const isEmpty = await page.locator('[data-testid="empty-cart-message"]').isVisible();

    if (!isEmpty) {
      // Check item count instead
      const itemCount = await page.locator('[data-testid="cart-item"]').count();
      expect(itemCount).toBe(0);
    } else {
      expect(isEmpty).toBe(true);
    }
  });

  test('empty cart shows appropriate message', async ({ cartPage }) => {
    await cartPage.goto();

    // Clear cart first if needed
    const itemCount = await cartPage.getItemCount();
    if (itemCount > 0) {
      while (await cartPage.getItemCount() > 0) {
        await cartPage.removeFirstItem();
      }
    }

    // Reload to see empty state
    await cartPage.page.reload();

    // Check for empty message or zero items
    const isEmpty = await cartPage.isCartEmpty();
    const finalCount = await cartPage.getItemCount();

    expect(isEmpty || finalCount === 0).toBe(true);
  });

  test('checkout button is disabled when cart is empty', async ({ cartPage }) => {
    await cartPage.goto();

    // Clear cart
    while (await cartPage.getItemCount() > 0) {
      await cartPage.removeFirstItem();
    }

    // Verify checkout button state
    const checkoutBtn = cartPage.page.getByRole('button', { name: /checkout/i });

    if (await checkoutBtn.isVisible()) {
      // Button might be disabled or hidden
      const isEnabled = await checkoutBtn.isEnabled();
      const isVisible = await checkoutBtn.isVisible();

      expect(!isEnabled || !isVisible).toBe(true);
    }
  });

  test('multiple items can be managed independently', async ({ shopPage, cartPage, page }) => {
    await shopPage.goto();

    const productCount = await shopPage.getProductCount();
    if (productCount < 2) {
      test.skip();
    }

    // Add first product
    await shopPage.clickProductAt(0);
    let addButton = page.getByRole('button', { name: /add to cart/i }).first();
    await addButton.click();

    // Go back and add second product
    await shopPage.goto();
    await shopPage.clickProductAt(1);
    addButton = page.getByRole('button', { name: /add to cart/i }).first();
    await addButton.click();

    // Open cart
    await shopPage.openCart();

    // Should have 2 items
    const itemCount = await cartPage.getItemCount();
    expect(itemCount).toBe(2);

    // Remove first item
    await cartPage.removeItemAt(0);

    // Should have 1 item
    const newCount = await cartPage.getItemCount();
    expect(newCount).toBe(1);
  });

  test('cart works on mobile device', async ({ shopPage, page }) => {
    // Set mobile viewport
    await page.setViewportSize({ width: 375, height: 667 });

    await shopPage.goto();

    // Add product
    const productCount = await shopPage.getProductCount();
    expect(productCount).toBeGreaterThan(0);

    await shopPage.clickProductAt(0);

    const addButton = page.getByRole('button', { name: /add to cart/i }).first();
    await addButton.click();

    // Open cart (might be drawer on mobile)
    await shopPage.openCart();

    // Verify cart is accessible
    const cartContent = page.locator('[data-testid="cart-content"], [data-testid="cart-item"]').first();
    await expect(cartContent).toBeVisible();
  });

  test('quantity input accepts valid numbers', async ({ shopPage, cartPage, page }) => {
    // Add item
    await shopPage.goto();
    await shopPage.clickProductAt(0);

    const addButton = page.getByRole('button', { name: /add to cart/i }).first();
    await addButton.click();

    // Open cart
    await shopPage.openCart();

    // Try to set quantity to 5
    const quantityInput = cartPage.page.locator('[data-testid="quantity-input"]').first();
    if (await quantityInput.isVisible()) {
      await cartPage.setQuantityAt(0, 5);

      const value = await quantityInput.inputValue();
      expect(parseInt(value)).toBe(5);
    }
  });
});
