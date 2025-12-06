import { test, expect } from '@playwright/test';

test.describe('Visual Regression Tests', () => {
  test('Homepage visual regression', async ({ page }) => {
    await page.goto('/');
    
    // Wait for dynamic content to load
    await page.waitForSelector('h1', { timeout: 10000 });
    await page.waitForLoadState('networkidle');
    
    // Take screenshot
    await expect(page).toHaveScreenshot('homepage.png');
  });

  test('Scan page visual regression', async ({ page }) => {
    await page.goto('/scan');
    
    // Wait for page to fully load
    await page.waitForLoadState('networkidle');
    
    await expect(page).toHaveScreenshot('scan-page.png');
  });

  test('Workspace page visual regression', async ({ page }) => {
    await page.goto('/workspace');
    
    // Wait for page to fully load
    await page.waitForLoadState('networkidle');
    
    await expect(page).toHaveScreenshot('workspace-page.png');
  });

  test('Settings drawer visual regression', async ({ page }) => {
    await page.goto('/');
    
    // Open settings drawer
    await page.click('[aria-label*="Settings"]');
    await page.waitForSelector('[role="dialog"]');
    
    await expect(page).toHaveScreenshot('settings-drawer.png');
  });

  test('Mobile homepage visual regression', async ({ page }) => {
    await page.setViewportSize({ width: 375, height: 667 });
    await page.goto('/');
    
    // Wait for dynamic content to load
    await page.waitForSelector('h1', { timeout: 10000 });
    await page.waitForLoadState('networkidle');
    
    await expect(page).toHaveScreenshot('mobile-homepage.png');
  });

  test('Dark theme visual regression', async ({ page }) => {
    await page.goto('/');
    
    // Enable dark theme (assuming there's a theme toggle)
    await page.click('[aria-label*="Settings"]');
    await page.waitForSelector('[role="dialog"]');
    
    // Look for theme toggle and switch to dark
    const darkModeToggle = page.locator('text=Dark').first();
    if (await darkModeToggle.isVisible()) {
      await darkModeToggle.click();
    }
    
    // Close settings
    await page.keyboard.press('Escape');
    await page.waitForLoadState('networkidle');
    
    await expect(page).toHaveScreenshot('dark-theme.png');
  });

  test('Algorithm flow diagram visual regression', async ({ page }) => {
    await page.goto('/');
    
    // Scroll to algorithm section
    await page.locator('#methods').scrollIntoViewIfNeeded();
    await page.waitForTimeout(1000); // Allow animations to settle
    
    // Take screenshot of just the algorithm section
    await expect(page.locator('#methods')).toHaveScreenshot('algorithm-flow.png');
  });

  test('Error state visual regression', async ({ page }) => {
    await page.goto('/scan');
    
    // Trigger an error by uploading an invalid file type
    const fileInput = page.locator('input[type="file"]');
    
    // Create a test file with wrong extension
    const buffer = Buffer.from('invalid content');
    await fileInput.setInputFiles({
      name: 'test.xyz',
      mimeType: 'application/octet-stream',
      buffer: buffer,
    });
    
    // Wait for error state
    await page.waitForSelector('[role="alert"]', { timeout: 5000 });
    
    await expect(page).toHaveScreenshot('error-state.png');
  });
});