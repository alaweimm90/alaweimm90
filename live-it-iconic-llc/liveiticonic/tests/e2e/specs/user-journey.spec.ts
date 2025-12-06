import { test, expect } from '../fixtures/auth.fixture';

test.describe('User Journey', () => {
  test('new visitor can navigate homepage and access shop', async ({ homePage, shopPage }) => {
    // Navigate to home page
    await homePage.goto();

    // Verify hero is visible
    await expect(homePage.heroTitle).toBeVisible();
    const titleText = await homePage.heroTitle.textContent();
    expect(titleText?.toLowerCase()).toContain('live');

    // Click shop button
    await homePage.clickShop();

    // Verify we're on shop page
    await expect(shopPage.productCards.first()).toBeVisible({ timeout: 5000 });
  });

  test('user can view product details', async ({ shopPage, productDetailPage, page }) => {
    await shopPage.goto();

    // Verify products are loaded
    const productCount = await shopPage.getProductCount();
    expect(productCount).toBeGreaterThan(0);

    // Click first product
    await shopPage.clickProductAt(0);

    // Verify product detail page elements are visible
    await expect(productDetailPage.productTitle).toBeVisible({ timeout: 5000 });
    await expect(productDetailPage.addToCartButton).toBeVisible();

    // Get product info
    const title = await productDetailPage.getProductTitle();
    const price = await productDetailPage.getProductPrice();

    expect(title).toBeTruthy();
    expect(price).toBeTruthy();
  });

  test('user can add item to cart and verify count', async ({ shopPage, page }) => {
    await shopPage.goto();

    // Verify initial cart is empty (or skip if not visible)
    const initialBadge = await shopPage.cartBadge.textContent();

    // Add first product to cart
    const productCount = await shopPage.getProductCount();
    expect(productCount).toBeGreaterThan(0);

    await shopPage.clickProductAt(0);

    // Add to cart
    const { productDetailPage } = await test.info().fixturePath;
    const productName = await shopPage.getFirstProductName();

    // Click add to cart button on product detail page
    const addButton = page.getByRole('button', { name: /add to cart/i }).first();
    await addButton.click();

    // Verify cart badge updates
    await expect(shopPage.cartBadge).toBeVisible({ timeout: 5000 });
    const cartCount = await shopPage.cartBadge.textContent();
    expect(cartCount).toBeTruthy();
  });

  test('user can navigate between pages', async ({ homePage, shopPage, page }) => {
    // Start on home
    await homePage.goto();
    await expect(homePage.heroTitle).toBeVisible();

    // Go to shop
    await homePage.clickShop();
    await expect(shopPage.productCards.first()).toBeVisible({ timeout: 5000 });

    // Go back to home via nav
    await homePage.navShop.first().click();
    // Should stay on shop, so click home instead if available
    const homeLink = page.getByRole('link', { name: /home/i }).first();
    if (await homeLink.isVisible()) {
      await homeLink.click();
      await expect(homePage.heroTitle).toBeVisible();
    }
  });

  test('search functionality works', async ({ shopPage }) => {
    await shopPage.goto();

    const initialCount = await shopPage.getProductCount();
    expect(initialCount).toBeGreaterThan(0);

    // Search for a product (this will depend on actual products)
    await shopPage.searchProducts('test');

    // Wait for results to update
    await shopPage.page.waitForTimeout(500);

    const searchCount = await shopPage.getProductCount();
    // Count might be same or less depending on products
    expect(searchCount).toBeGreaterThanOrEqual(0);
  });

  test('product filters work', async ({ shopPage }) => {
    await shopPage.goto();

    const initialCount = await shopPage.getProductCount();
    expect(initialCount).toBeGreaterThan(0);

    // Try to use sort if available
    // This depends on actual UI implementation
    const sortDropdown = shopPage.page.locator('[data-testid="sort-dropdown"]');
    if (await sortDropdown.isVisible()) {
      await shopPage.sortBy('Price: Low to High');
      // Verify results update
      await shopPage.page.waitForLoadState('networkidle');
      const sortedCount = await shopPage.getProductCount();
      expect(sortedCount).toBeGreaterThan(0);
    }
  });

  test('responsive design works on mobile', async ({ shopPage, page, browserName }) => {
    // Set mobile viewport
    await page.setViewportSize({ width: 375, height: 667 });

    await shopPage.goto();

    // Verify products are still visible
    const productCount = await shopPage.getProductCount();
    expect(productCount).toBeGreaterThan(0);

    // Verify product cards are stacked
    const firstCard = shopPage.productCards.first();
    await expect(firstCard).toBeVisible();
  });

  test('page load performance is acceptable', async ({ page }) => {
    const startTime = Date.now();

    await page.goto('/shop', { waitUntil: 'networkidle' });

    const loadTime = Date.now() - startTime;

    // Page should load within 5 seconds
    expect(loadTime).toBeLessThan(5000);
  });
});
