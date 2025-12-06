import { test, expect } from '@playwright/test';

test.describe('Navigation', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
  });

  test('should navigate to all main pages', async ({ page }) => {
    // Test home page
    await expect(page).toHaveTitle(/LLM Works/);
    
    // Navigate to Arena
    await page.click('text=Arena');
    await expect(page).toHaveURL(/\/arena/);
    await expect(page.locator('h1')).toContainText('Arena');

    // Navigate to Bench
    await page.click('text=Bench');
    await expect(page).toHaveURL(/\/bench/);
    await expect(page.locator('h1')).toContainText('Bench');

    // Navigate to Dashboard
    await page.click('text=Dashboard');
    await expect(page).toHaveURL(/\/dashboard/);
    await expect(page.locator('h1')).toContainText('Dashboard');

    // Navigate to Settings
    await page.click('text=Settings');
    await expect(page).toHaveURL(/\/settings/);
    await expect(page.locator('h1')).toContainText('Settings');
  });

  test('should handle 404 pages gracefully', async ({ page }) => {
    await page.goto('/non-existent-page');
    await expect(page.locator('text=404')).toBeVisible();
    await expect(page.locator('text=Page not found')).toBeVisible();
    
    // Should have a way to get back home
    await page.click('text=Back to Home');
    await expect(page).toHaveURL('/');
  });

  test('should support keyboard navigation', async ({ page }) => {
    // Tab through navigation elements
    await page.keyboard.press('Tab');
    const focused = await page.evaluate(() => document.activeElement?.tagName);
    expect(['A', 'BUTTON']).toContain(focused);

    // Should be able to activate with Enter
    await page.keyboard.press('Enter');
    // Check that navigation occurred (page changed or action happened)
  });

  test('should work on mobile', async ({ page, isMobile }) => {
    if (isMobile) {
      // Mobile navigation might be different (hamburger menu)
      const menuButton = page.locator('[aria-label*="menu"], [aria-label*="Menu"]');
      if (await menuButton.isVisible()) {
        await menuButton.click();
      }
    }

    // Should still be able to navigate
    await page.click('text=Arena');
    await expect(page).toHaveURL(/\/arena/);
  });
});