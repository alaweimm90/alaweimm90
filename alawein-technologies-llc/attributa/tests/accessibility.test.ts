import { test, expect } from '@playwright/test';
import { injectAxe, checkA11y } from 'axe-playwright';

test.describe('Accessibility Tests', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await injectAxe(page);
  });

  test('Homepage should be accessible', async ({ page }) => {
    await checkA11y(page, null, {
      detailedReport: true,
      detailedReportOptions: { html: true },
    });
  });

  test('Scan page should be accessible', async ({ page }) => {
    await page.goto('/scan');
    await checkA11y(page, null, {
      detailedReport: true,
      detailedReportOptions: { html: true },
    });
  });

  test('Settings drawer should be accessible', async ({ page }) => {
    // Open settings drawer
    await page.click('[aria-label*="Settings"]');
    
    // Check accessibility with drawer open
    await checkA11y(page, null, {
      detailedReport: true,
      detailedReportOptions: { html: true },
    });
  });

  test('Focus management in modals', async ({ page }) => {
    // Open settings drawer
    await page.click('[aria-label*="Settings"]');
    
    // Check that focus is trapped in the drawer
    await page.keyboard.press('Tab');
    const focusedElement = await page.locator(':focus');
    
    // Focus should be within the drawer
    expect(await focusedElement.isVisible()).toBe(true);
  });

  test('Keyboard navigation works', async ({ page }) => {
    // Test tab navigation through main elements
    await page.keyboard.press('Tab'); // Skip link
    await page.keyboard.press('Tab'); // Logo
    await page.keyboard.press('Tab'); // Workspace
    await page.keyboard.press('Tab'); // GitHub link
    await page.keyboard.press('Tab'); // Settings
    await page.keyboard.press('Tab'); // Analyze button
    
    const focusedElement = await page.locator(':focus');
    expect(await focusedElement.textContent()).toContain('Analyze');
  });

  test('Skip links work correctly', async ({ page }) => {
    // Focus the skip link (usually hidden)
    await page.keyboard.press('Tab');
    await page.keyboard.press('Enter');
    
    // Check that main content is focused
    const focusedElement = await page.locator(':focus');
    const mainContent = await page.locator('#main-content');
    
    expect(await focusedElement.getAttribute('id')).toBe('main-content');
  });

  test('Color contrast meets WCAG standards', async ({ page }) => {
    await checkA11y(page, null, {
      tags: ['wcag2a', 'wcag2aa'],
      rules: {
        'color-contrast': { enabled: true },
        'color-contrast-enhanced': { enabled: true }
      }
    });
  });

  test('Images have proper alt text', async ({ page }) => {
    await checkA11y(page, null, {
      rules: {
        'image-alt': { enabled: true }
      }
    });
  });

  test('Form labels are properly associated', async ({ page }) => {
    await page.goto('/scan');
    
    await checkA11y(page, null, {
      rules: {
        'label': { enabled: true },
        'form-field-multiple-labels': { enabled: true }
      }
    });
  });
});