import { test, expect } from '@playwright/test';
import AxeBuilder from '@axe-core/playwright';

test.describe('Accessibility', () => {
  test('should pass axe accessibility tests on all pages', async ({ page }) => {
    const pages = ['/', '/arena', '/bench', '/dashboard', '/settings'];
    
    for (const path of pages) {
      await page.goto(path);
      
      // Wait for page to load
      await page.waitForLoadState('networkidle');
      
      const accessibilityScanResults = await new AxeBuilder({ page })
        .withTags(['wcag2a', 'wcag2aa', 'wcag21aa'])
        .analyze();
      
      expect(accessibilityScanResults.violations).toEqual([]);
    }
  });

  test('should have proper focus management', async ({ page }) => {
    await page.goto('/');
    
    // Tab through elements
    const focusableElements = [];
    let previousElement = null;
    
    for (let i = 0; i < 10; i++) {
      await page.keyboard.press('Tab');
      const currentElement = await page.evaluate(() => {
        const el = document.activeElement;
        return el ? {
          tagName: el.tagName,
          role: el.getAttribute('role'),
          ariaLabel: el.getAttribute('aria-label'),
          text: el.textContent?.trim().substring(0, 50)
        } : null;
      });
      
      if (currentElement && currentElement !== previousElement) {
        focusableElements.push(currentElement);
        previousElement = currentElement;
      }
    }
    
    // Should have focusable elements
    expect(focusableElements.length).toBeGreaterThan(0);
  });

  test('should support screen reader navigation', async ({ page }) => {
    await page.goto('/');
    
    // Check for proper heading hierarchy
    const headings = await page.locator('h1, h2, h3, h4, h5, h6').all();
    expect(headings.length).toBeGreaterThan(0);
    
    // Check for landmarks
    const main = page.locator('main');
    await expect(main).toBeVisible();
    
    const nav = page.locator('nav');
    await expect(nav).toBeVisible();
  });

  test('should have accessible forms', async ({ page }) => {
    await page.goto('/settings');
    
    // Check that all inputs have labels
    const inputs = await page.locator('input:not([type="hidden"])').all();
    
    for (const input of inputs) {
      const id = await input.getAttribute('id');
      const ariaLabel = await input.getAttribute('aria-label');
      const ariaLabelledby = await input.getAttribute('aria-labelledby');
      
      if (id) {
        const label = page.locator(`label[for="${id}"]`);
        const hasLabel = await label.count() > 0;
        
        expect(hasLabel || ariaLabel || ariaLabelledby).toBeTruthy();
      } else {
        expect(ariaLabel || ariaLabelledby).toBeTruthy();
      }
    }
  });

  test('should support keyboard-only navigation', async ({ page }) => {
    await page.goto('/arena');
    
    // Should be able to navigate without mouse
    await page.keyboard.press('Tab');
    await page.keyboard.press('Enter');
    
    // Check that actions can be performed
    const activeElement = await page.evaluate(() => document.activeElement?.tagName);
    expect(['BUTTON', 'A', 'INPUT', 'SELECT']).toContain(activeElement);
  });

  test('should handle high contrast mode', async ({ page, contextOptions }) => {
    await page.goto('/');
    
    // In forced colors mode, check that content is still visible
    if (contextOptions.forcedColors === 'active') {
      // Key content should still be visible
      await expect(page.locator('h1')).toBeVisible();
      await expect(page.locator('main')).toBeVisible();
    }
  });

  test('should respect reduced motion preferences', async ({ page }) => {
    await page.goto('/');
    
    // Check that animations are disabled or reduced
    const hasReducedMotion = await page.evaluate(() => 
      window.matchMedia('(prefers-reduced-motion: reduce)').matches
    );
    
    if (hasReducedMotion) {
      // Verify that animations are disabled
      const animatedElements = await page.locator('[class*="animate"]').all();
      
      for (const element of animatedElements) {
        const animationDuration = await element.evaluate(el => 
          window.getComputedStyle(el).animationDuration
        );
        // Should be 0 or very short for reduced motion
        expect(parseFloat(animationDuration)).toBeLessThanOrEqual(0.1);
      }
    }
  });

  test('should have accessible buttons and links', async ({ page }) => {
    await page.goto('/');
    
    // Check buttons have accessible names
    const buttons = await page.locator('button').all();
    for (const button of buttons) {
      const text = await button.textContent();
      const ariaLabel = await button.getAttribute('aria-label');
      const ariaLabelledby = await button.getAttribute('aria-labelledby');
      
      const hasAccessibleName = (text && text.trim().length > 0) || ariaLabel || ariaLabelledby;
      expect(hasAccessibleName).toBeTruthy();
    }
    
    // Check links have accessible names
    const links = await page.locator('a[href]').all();
    for (const link of links) {
      const text = await link.textContent();
      const ariaLabel = await link.getAttribute('aria-label');
      const ariaLabelledby = await link.getAttribute('aria-labelledby');
      
      const hasAccessibleName = (text && text.trim().length > 0) || ariaLabel || ariaLabelledby;
      expect(hasAccessibleName).toBeTruthy();
    }
  });

  test('should support accessibility toolbar features', async ({ page }) => {
    await page.goto('/');
    
    // Open accessibility toolbar
    const accessibilityButton = page.locator('[aria-label*="accessibility"], [title*="accessibility"]');
    if (await accessibilityButton.isVisible()) {
      await accessibilityButton.click();
      
      // Test high contrast toggle
      const highContrastToggle = page.locator('text=High Contrast').or(page.locator('[aria-label*="high contrast"]'));
      if (await highContrastToggle.isVisible()) {
        await highContrastToggle.click();
        
        // Check that high contrast class is applied
        const bodyClasses = await page.evaluate(() => document.body.className);
        expect(bodyClasses).toContain('high-contrast');
      }
      
      // Test large text toggle
      const largeTextToggle = page.locator('text=Large Text').or(page.locator('[aria-label*="large text"]'));
      if (await largeTextToggle.isVisible()) {
        await largeTextToggle.click();
        
        // Check that large text class is applied
        const bodyClasses = await page.evaluate(() => document.body.className);
        expect(bodyClasses).toContain('large-text');
      }
    }
  });
});