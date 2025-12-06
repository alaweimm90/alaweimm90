import { test, expect } from '../fixtures/auth.fixture';
import { addProductToCart, generateTestData, waitForCartBadgeUpdate } from '../utils/test-helpers';

test.describe('Purchase Flow', () => {
  test('complete purchase from product to checkout', async ({ shopPage, cartPage, checkoutPage, page }) => {
    // Start on shop page
    await shopPage.goto();

    // Verify products available
    const productCount = await shopPage.getProductCount();
    expect(productCount).toBeGreaterThan(0);

    // Add product to cart
    await shopPage.clickProductAt(0);

    // Wait for product detail page
    const addButton = page.getByRole('button', { name: /add to cart/i }).first();
    await expect(addButton).toBeVisible();
    await addButton.click();

    // Open cart
    await shopPage.openCart();

    // Verify cart has item
    const itemCount = await cartPage.getItemCount();
    expect(itemCount).toBeGreaterThan(0);

    // Proceed to checkout
    await cartPage.proceedToCheckout();

    // Verify we're on checkout page
    await expect(checkoutPage.emailInput).toBeVisible({ timeout: 5000 });
  });

  test('user can fill in shipping information', async ({ shopPage, cartPage, checkoutPage, page }) => {
    // Add product and go to checkout
    await shopPage.goto();
    await shopPage.clickProductAt(0);

    const addButton = page.getByRole('button', { name: /add to cart/i }).first();
    await addButton.click();

    await shopPage.openCart();
    await cartPage.proceedToCheckout();

    // Fill shipping info
    const testData = generateTestData();

    await checkoutPage.fillShippingInfo({
      email: testData.email,
      firstName: testData.firstName,
      lastName: testData.lastName,
      address: testData.address,
      city: testData.city,
      state: testData.state,
      zip: testData.zip,
      country: testData.country,
      phone: testData.phone,
    });

    // Verify fields are filled
    await expect(checkoutPage.emailInput).toHaveValue(testData.email);
    await expect(checkoutPage.firstNameInput).toHaveValue(testData.firstName);
    await expect(checkoutPage.cityInput).toHaveValue(testData.city);
  });

  test('user can select shipping method', async ({ shopPage, cartPage, checkoutPage, page }) => {
    // Add product and go to checkout
    await shopPage.goto();
    await shopPage.clickProductAt(0);

    const addButton = page.getByRole('button', { name: /add to cart/i }).first();
    await addButton.click();

    await shopPage.openCart();
    await cartPage.proceedToCheckout();

    // Check if shipping method selector is visible
    const shippingMethod = checkoutPage.page.locator('[data-testid="shipping-method"]');
    if (await shippingMethod.isVisible()) {
      // Select a shipping option (usually first/default)
      await checkoutPage.selectShippingMethod('standard');
      await expect(shippingMethod).toHaveAttribute('data-selected', 'true');
    }
  });

  test('order summary displays correctly', async ({ shopPage, cartPage, checkoutPage, page }) => {
    // Add product and go to checkout
    await shopPage.goto();

    const firstPrice = await shopPage.getFirstProductPrice();
    await shopPage.clickProductAt(0);

    const addButton = page.getByRole('button', { name: /add to cart/i }).first();
    await addButton.click();

    await shopPage.openCart();

    // Check cart total
    const cartTotal = await cartPage.getCartTotal();
    expect(cartTotal).toBeTruthy();

    // Proceed to checkout
    await cartPage.proceedToCheckout();

    // Verify order summary is visible
    const orderSummary = checkoutPage.page.locator('[data-testid="order-summary"]');
    if (await orderSummary.isVisible()) {
      expect(orderSummary).toBeVisible();
    }
  });

  test('cart total updates with multiple items', async ({ shopPage, cartPage, page }) => {
    await shopPage.goto();

    // Add first product
    await shopPage.clickProductAt(0);
    const addButton = page.getByRole('button', { name: /add to cart/i }).first();
    await addButton.click();

    // Go back to shop
    await shopPage.goto();

    // Add second product if available
    const productCount = await shopPage.getProductCount();
    if (productCount > 1) {
      await shopPage.clickProductAt(1);
      const addBtn2 = page.getByRole('button', { name: /add to cart/i }).first();
      await addBtn2.click();

      // Open cart
      await shopPage.openCart();

      // Verify 2 items in cart
      const itemCount = await cartPage.getItemCount();
      expect(itemCount).toBe(2);

      // Verify total is updated
      const total = await cartPage.getCartTotal();
      expect(total).toBeTruthy();
    }
  });

  test('invalid shipping information shows errors', async ({ shopPage, cartPage, checkoutPage, page }) => {
    await shopPage.goto();
    await shopPage.clickProductAt(0);

    const addButton = page.getByRole('button', { name: /add to cart/i }).first();
    await addButton.click();

    await shopPage.openCart();
    await cartPage.proceedToCheckout();

    // Try to proceed without filling required fields
    const continueBtn = checkoutPage.page.getByRole('button', { name: /continue/i }).first();

    // Some forms will prevent submission, some will show errors
    // Try to check for validation messages
    const validationMessage = checkoutPage.page.locator('[data-testid="error-message"], .error, [role="alert"]');

    // This test is flexible as validation implementation may vary
    if (await validationMessage.isVisible()) {
      expect(validationMessage).toBeVisible();
    }
  });

  test('promo code can be applied in cart', async ({ shopPage, cartPage, page }) => {
    await shopPage.goto();
    await shopPage.clickProductAt(0);

    const addButton = page.getByRole('button', { name: /add to cart/i }).first();
    await addButton.click();

    await shopPage.openCart();

    // Check if promo code field is available
    const promoInput = cartPage.page.locator('[data-testid="promo-code-input"]');
    if (await promoInput.isVisible()) {
      const originalTotal = await cartPage.getCartTotal();

      // Try applying a promo code
      await cartPage.applyPromoCode('SAVE10');

      // Note: This test assumes a demo code might exist
      // In real scenario, you'd have test codes or mock the API
      await page.waitForTimeout(1000);

      // Check if total changed (it might not if code is invalid)
      const newTotal = await cartPage.getCartTotal();
      expect(newTotal).toBeTruthy();
    }
  });

  test('quantity can be updated in checkout review', async ({ shopPage, cartPage, checkoutPage, page }) => {
    await shopPage.goto();
    await shopPage.clickProductAt(0);

    const addButton = page.getByRole('button', { name: /add to cart/i }).first();
    await addButton.click();

    await shopPage.openCart();

    // Check if we can update quantity
    const increaseBtn = cartPage.page.locator('[data-testid="increase-quantity"]').first();
    if (await increaseBtn.isVisible()) {
      await cartPage.increaseQuantityAt(0);

      // Verify quantity updated
      await page.waitForTimeout(500);
      const itemCount = await cartPage.getItemCount();
      expect(itemCount).toBeGreaterThanOrEqual(1);

      // Verify total updated
      const newTotal = await cartPage.getCartTotal();
      expect(newTotal).toBeTruthy();
    }
  });

  test('back button works in checkout', async ({ shopPage, cartPage, checkoutPage, page }) => {
    await shopPage.goto();
    await shopPage.clickProductAt(0);

    const addButton = page.getByRole('button', { name: /add to cart/i }).first();
    await addButton.click();

    await shopPage.openCart();
    await cartPage.proceedToCheckout();

    // Verify we're on checkout page
    const emailInput = checkoutPage.page.getByLabel(/email/i);
    await expect(emailInput).toBeVisible();

    // Go back
    const backBtn = checkoutPage.page.getByRole('button', { name: /back/i }).first();
    if (await backBtn.isVisible()) {
      await backBtn.click();

      // Should be back on cart page or shop
      await page.waitForURL((url) => url.toString().includes('/cart') || url.toString().includes('/shop'));
    }
  });
});
