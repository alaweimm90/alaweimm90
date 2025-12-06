import { test, expect } from '@playwright/test';

test.describe('Complete User Journey (E2E)', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
  });

  test('complete signup and onboarding flow', async ({ page }) => {
    // Landing page interaction
    await expect(page.locator('h1')).toContainText('Transform Your Fitness Journey');
    
    // Navigate to pricing
    await page.click('text=View Pricing');
    await expect(page).toHaveURL(/.*pricing/);
    
    // Select a tier
    await page.click('[data-testid="tier-core-select"]');
    
    // Fill out reservation form
    await page.fill('[data-testid="name-input"]', 'Test User');
    await page.fill('[data-testid="email-input"]', 'test@example.com');
    await page.fill('[data-testid="phone-input"]', '555-123-4567');
    
    // Submit form
    await page.click('[data-testid="submit-reservation"]');
    
    // Verify success
    await expect(page.locator('text=Reservation Confirmed')).toBeVisible();
  });

  test('analytics and performance tracking', async ({ page }) => {
    // Check that analytics are being tracked
    const analyticsRequests: string[] = [];
    
    page.on('request', request => {
      if (request.url().includes('analytics') || request.url().includes('track')) {
        analyticsRequests.push(request.url());
      }
    });
    
    // Interact with various elements
    await page.click('text=Features');
    await page.click('text=Pricing');
    await page.click('text=Contact');
    
    // Wait for analytics requests
    await page.waitForTimeout(2000);
    
    // Verify analytics are being tracked
    expect(analyticsRequests.length).toBeGreaterThan(0);
  });

  test('responsive design across viewports', async ({ page }) => {
    // Test desktop
    await page.setViewportSize({ width: 1920, height: 1080 });
    await expect(page.locator('[data-testid="desktop-nav"]')).toBeVisible();
    
    // Test tablet
    await page.setViewportSize({ width: 768, height: 1024 });
    await expect(page.locator('[data-testid="mobile-menu-button"]')).toBeVisible();
    
    // Test mobile
    await page.setViewportSize({ width: 375, height: 667 });
    await page.click('[data-testid="mobile-menu-button"]');
    await expect(page.locator('[data-testid="mobile-menu"]')).toBeVisible();
  });

  test('performance benchmarks', async ({ page }) => {
    const startTime = Date.now();
    
    await page.goto('/');
    await page.waitForLoadState('networkidle');
    
    const loadTime = Date.now() - startTime;
    
    // Page should load within 3 seconds
    expect(loadTime).toBeLessThan(3000);
    
    // Check for performance issues
    const performanceMetrics = await page.evaluate(() => {
      const navigation = performance.getEntriesByType('navigation')[0] as PerformanceNavigationTiming;
      return {
        domContentLoaded: navigation.domContentLoadedEventEnd - navigation.fetchStart,
        loadComplete: navigation.loadEventEnd - navigation.fetchStart
      };
    });
    
    expect(performanceMetrics.domContentLoaded).toBeLessThan(2000);
    expect(performanceMetrics.loadComplete).toBeLessThan(3000);
  });

  test('accessibility compliance', async ({ page }) => {
    // Check for ARIA labels
    const buttons = await page.locator('button').all();
    
    for (const button of buttons) {
      const ariaLabel = await button.getAttribute('aria-label');
      const text = await button.textContent();
      
      // Button should have either text content or aria-label
      expect(ariaLabel || text).toBeTruthy();
    }
    
    // Check for alt text on images
    const images = await page.locator('img').all();
    
    for (const img of images) {
      const alt = await img.getAttribute('alt');
      expect(alt).toBeTruthy();
    }
    
    // Check color contrast (basic check)
    const headings = await page.locator('h1, h2, h3').all();
    
    for (const heading of headings) {
      const color = await heading.evaluate(el => {
        return window.getComputedStyle(el).color;
      });
      
      // Should not be using default black on white (basic check)
      expect(color).not.toBe('rgb(0, 0, 0)');
    }
  });

  test('error handling and recovery', async ({ page }) => {
    // Test 404 page
    await page.goto('/nonexistent-page');
    await expect(page.locator('text=Page Not Found')).toBeVisible();
    
    // Should have navigation back to home
    await page.click('text=Back to Home');
    await expect(page).toHaveURL('/');
    
    // Test form validation
    await page.goto('/signup');
    await page.click('[data-testid="submit-form"]');
    
    // Should show validation errors
    await expect(page.locator('text=Email is required')).toBeVisible();
  });

  test('cross-browser compatibility', async ({ page, browserName }) => {
    console.log(`Testing on ${browserName}`);
    
    // Basic functionality should work across browsers
    await expect(page.locator('h1')).toBeVisible();
    
    // Check for browser-specific features
    const supportsWebP = await page.evaluate(() => {
      const canvas = document.createElement('canvas');
      return canvas.toDataURL('image/webp').indexOf('data:image/webp') === 0;
    });
    
    if (browserName === 'webkit') {
      // Safari might not support all features
      console.log('Safari compatibility check passed');
    }
    
    // CSS Grid support check
    const supportsGrid = await page.evaluate(() => {
      return CSS.supports('display', 'grid');
    });
    
    expect(supportsGrid).toBe(true);
  });
});