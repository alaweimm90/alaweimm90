import { test, expect } from '@playwright/test';
import AxeBuilder from '@axe-core/playwright';

test.describe('WCAG 2.2 AA Compliance Testing', () => {
  test.beforeEach(async ({ page }) => {
    // Use default client auth for accessibility testing
    await page.context().storageState({ 
      path: 'tests/fixtures/auth/client-auth.json' 
    });
  });

  test('homepage accessibility compliance', async ({ page }) => {
    await page.goto('/');
    
    const accessibilityScanResults = await new AxeBuilder({ page })
      .withTags(['wcag2a', 'wcag2aa', 'wcag21aa', 'wcag22aa'])
      .analyze();

    expect(accessibilityScanResults.violations).toEqual([]);
    
    // Additional manual checks
    await expect(page.locator('h1')).toBeVisible();
    await expect(page.locator('[role="main"]')).toBeVisible();
    await expect(page.locator('[role="navigation"]')).toBeVisible();
  });

  test('authentication pages accessibility', async ({ page }) => {
    // Test login page
    await page.goto('/auth/login');
    
    let scanResults = await new AxeBuilder({ page })
      .withTags(['wcag2a', 'wcag2aa'])
      .analyze();
    
    expect(scanResults.violations).toEqual([]);
    
    // Verify form accessibility
    await expect(page.locator('label[for="email"]')).toBeVisible();
    await expect(page.locator('label[for="password"]')).toBeVisible();
    await expect(page.locator('#email')).toHaveAttribute('required');
    await expect(page.locator('#password')).toHaveAttribute('required');
    
    // Test signup page
    await page.goto('/auth/signup');
    
    scanResults = await new AxeBuilder({ page })
      .withTags(['wcag2a', 'wcag2aa'])
      .analyze();
    
    expect(scanResults.violations).toEqual([]);
    
    // Verify password requirements are announced
    await expect(page.locator('[aria-describedby="password-requirements"]')).toBeVisible();
    await expect(page.locator('#password-requirements')).toBeVisible();
  });

  test('dashboard accessibility across tiers', async ({ page }) => {
    const tiers = ['core', 'adaptive', 'performance', 'longevity'];
    
    for (const tier of tiers) {
      await page.context().storageState({ 
        path: `tests/fixtures/auth/${tier}-auth.json` 
      });
      
      await page.goto('/dashboard');
      
      const scanResults = await new AxeBuilder({ page })
        .withTags(['wcag2a', 'wcag2aa'])
        .analyze();
      
      expect(scanResults.violations).toEqual([]);
      
      // Verify semantic structure
      await expect(page.locator('main')).toBeVisible();
      await expect(page.locator('nav')).toBeVisible();
      await expect(page.locator('h1, h2, h3').first()).toBeVisible();
      
      // Test keyboard navigation
      await page.keyboard.press('Tab');
      const focusedElement = page.locator(':focus');
      await expect(focusedElement).toBeVisible();
    }
  });

  test('nutrition management accessibility', async ({ page }) => {
    await page.context().storageState({ 
      path: 'tests/fixtures/auth/adaptive-auth.json' 
    });
    
    await page.goto('/nutrition/food-database');
    
    const scanResults = await new AxeBuilder({ page })
      .withTags(['wcag2a', 'wcag2aa'])
      .analyze();
    
    expect(scanResults.violations).toEqual([]);
    
    // Test search functionality accessibility
    const searchInput = page.locator('[role="searchbox"]');
    await expect(searchInput).toBeVisible();
    await expect(searchInput).toHaveAttribute('aria-label');
    
    // Test results announcements
    await searchInput.fill('chicken');
    await page.keyboard.press('Enter');
    
    await expect(page.locator('[aria-live="polite"]')).toBeVisible();
    await expect(page.locator('[role="region"][aria-label*="search results"]')).toBeVisible();
    
    // Test food item selection
    const firstResult = page.locator('[role="button"][aria-describedby*="nutrition"]').first();
    await expect(firstResult).toBeVisible();
    
    // Verify nutrition panel accessibility
    await firstResult.click();
    await expect(page.locator('[role="dialog"], [role="region"]')).toBeVisible();
    await expect(page.locator('table[aria-label*="nutrition"]')).toBeVisible();
  });

  test('protocols management accessibility (Performance+ tier)', async ({ page }) => {
    await page.context().storageState({ 
      path: 'tests/fixtures/auth/performance-auth.json' 
    });
    
    await page.goto('/protocols/peds');
    
    const scanResults = await new AxeBuilder({ page })
      .withTags(['wcag2a', 'wcag2aa'])
      .analyze();
    
    expect(scanResults.violations).toEqual([]);
    
    // Test protocol cards accessibility
    const protocolCards = page.locator('[role="article"], [role="button"][aria-describedby]');
    await expect(protocolCards.first()).toBeVisible();
    
    // Test warning and safety information
    await expect(page.locator('[role="alert"], [aria-live="assertive"]')).toBeVisible();
    
    // Test medical consultation form
    await page.click('[data-testid="request-consultation-button"]');
    
    const formScanResults = await new AxeBuilder({ page })
      .include('form')
      .withTags(['wcag2a', 'wcag2aa'])
      .analyze();
    
    expect(formScanResults.violations).toEqual([]);
    
    // Verify form field associations
    const textareas = page.locator('textarea');
    const textareaCount = await textareas.count();
    
    for (let i = 0; i < textareaCount; i++) {
      const textarea = textareas.nth(i);
      const labelId = await textarea.getAttribute('aria-labelledby');
      const describedBy = await textarea.getAttribute('aria-describedby');
      
      expect(labelId || describedBy).toBeTruthy();
    }
  });

  test('AI assistant accessibility', async ({ page }) => {
    await page.context().storageState({ 
      path: 'tests/fixtures/auth/performance-auth.json' 
    });
    
    await page.goto('/ai/chat');
    
    const scanResults = await new AxeBuilder({ page })
      .withTags(['wcag2a', 'wcag2aa'])
      .analyze();
    
    expect(scanResults.violations).toEqual([]);
    
    // Test chat interface accessibility
    const chatInput = page.locator('[role="textbox"][aria-label*="chat"]');
    await expect(chatInput).toBeVisible();
    
    const sendButton = page.locator('[role="button"][aria-label*="send"]');
    await expect(sendButton).toBeVisible();
    
    // Test conversation thread
    await chatInput.fill('What is the optimal protein intake?');
    await sendButton.click();
    
    // Verify message announcements
    await expect(page.locator('[role="log"], [aria-live="polite"]')).toBeVisible();
    
    // Test message history navigation
    const messageThread = page.locator('[role="log"] > *');
    await expect(messageThread.first()).toBeVisible();
    
    // Test keyboard navigation through messages
    await page.keyboard.press('Tab');
    const focusedMessage = page.locator(':focus');
    await expect(focusedMessage).toBeVisible();
  });

  test('mobile accessibility and touch targets', async ({ page }) => {
    await page.setViewportSize({ width: 375, height: 667 });
    
    await page.goto('/dashboard');
    
    const scanResults = await new AxeBuilder({ page })
      .withTags(['wcag2a', 'wcag2aa'])
      .analyze();
    
    expect(scanResults.violations).toEqual([]);
    
    // Test touch target sizes (minimum 44x44px)
    const interactiveElements = page.locator('button, a, input, [role="button"], [role="link"]');
    const elementCount = await interactiveElements.count();
    
    for (let i = 0; i < Math.min(elementCount, 10); i++) {
      const element = interactiveElements.nth(i);
      const boundingBox = await element.boundingBox();
      
      if (boundingBox) {
        expect(boundingBox.width).toBeGreaterThanOrEqual(44);
        expect(boundingBox.height).toBeGreaterThanOrEqual(44);
      }
    }
    
    // Test mobile navigation
    const mobileMenuButton = page.locator('[aria-label*="menu"], [aria-expanded]');
    if (await mobileMenuButton.count() > 0) {
      await expect(mobileMenuButton.first()).toBeVisible();
      await mobileMenuButton.first().click();
      
      const expandedMenu = page.locator('[aria-expanded="true"]');
      await expect(expandedMenu).toBeVisible();
    }
  });

  test('color contrast and visual accessibility', async ({ page }) => {
    await page.goto('/');
    
    // Test with high contrast preferences
    await page.emulateMedia({ colorScheme: 'dark' });
    
    const darkModeScanResults = await new AxeBuilder({ page })
      .withTags(['wcag2aa'])
      .analyze();
    
    expect(darkModeScanResults.violations).toEqual([]);
    
    // Test with reduced motion preferences
    await page.emulateMedia({ 
      reducedMotion: 'reduce',
      colorScheme: 'light'
    });
    
    // Verify animations are disabled
    const animatedElements = page.locator('[class*="animate"], [style*="animation"]');
    const animatedCount = await animatedElements.count();
    
    for (let i = 0; i < animatedCount; i++) {
      const element = animatedElements.nth(i);
      const computedStyle = await element.evaluate(el => 
        window.getComputedStyle(el).animationPlayState
      );
      
      expect(computedStyle).toBe('paused');
    }
  });

  test('form accessibility and error handling', async ({ page }) => {
    await page.goto('/intake/start');
    
    // Test form submission without required fields
    await page.click('[type="submit"]');
    
    const scanResults = await new AxeBuilder({ page })
      .withTags(['wcag2a', 'wcag2aa'])
      .analyze();
    
    expect(scanResults.violations).toEqual([]);
    
    // Verify error announcements
    const errorMessages = page.locator('[role="alert"], [aria-live="assertive"]');
    await expect(errorMessages.first()).toBeVisible();
    
    // Test field-specific error associations
    const invalidFields = page.locator('[aria-invalid="true"]');
    const invalidCount = await invalidFields.count();
    
    for (let i = 0; i < invalidCount; i++) {
      const field = invalidFields.nth(i);
      const describedBy = await field.getAttribute('aria-describedby');
      
      expect(describedBy).toBeTruthy();
      
      const errorElement = page.locator(`#${describedBy}`);
      await expect(errorElement).toBeVisible();
    }
    
    // Test successful form completion
    await page.fill('[name="firstName"]', 'John');
    await page.fill('[name="lastName"]', 'Doe');
    await page.fill('[name="email"]', 'test@example.com');
    
    // Verify error clearing
    await expect(page.locator('[aria-invalid="true"]')).toHaveCount(0);
  });

  test('data tables and complex content accessibility', async ({ page }) => {
    await page.context().storageState({ 
      path: 'tests/fixtures/auth/adaptive-auth.json' 
    });
    
    await page.goto('/nutrition/meal-planning');
    
    const scanResults = await new AxeBuilder({ page })
      .withTags(['wcag2a', 'wcag2aa'])
      .analyze();
    
    expect(scanResults.violations).toEqual([]);
    
    // Test meal planning calendar accessibility
    const calendar = page.locator('[role="grid"], table');
    if (await calendar.count() > 0) {
      await expect(calendar.first()).toHaveAttribute('aria-label');
      
      // Test table headers
      const headers = page.locator('th, [role="columnheader"]');
      const headerCount = await headers.count();
      
      for (let i = 0; i < headerCount; i++) {
        const header = headers.nth(i);
        const scope = await header.getAttribute('scope');
        const role = await header.getAttribute('role');
        
        expect(scope || role).toBeTruthy();
      }
    }
    
    // Test nutrition charts accessibility
    const charts = page.locator('[role="img"][aria-label*="chart"], canvas[aria-label]');
    const chartCount = await charts.count();
    
    for (let i = 0; i < chartCount; i++) {
      const chart = charts.nth(i);
      const ariaLabel = await chart.getAttribute('aria-label');
      const ariaDescribedBy = await chart.getAttribute('aria-describedby');
      
      expect(ariaLabel || ariaDescribedBy).toBeTruthy();
    }
  });

  test('screen reader compatibility', async ({ page }) => {
    await page.goto('/dashboard');
    
    // Test heading structure
    const headings = page.locator('h1, h2, h3, h4, h5, h6');
    const headingCount = await headings.count();
    
    expect(headingCount).toBeGreaterThan(0);
    
    // Verify heading hierarchy
    const h1Count = await page.locator('h1').count();
    expect(h1Count).toBe(1);
    
    // Test landmark regions
    const landmarks = page.locator('[role="banner"], [role="main"], [role="navigation"], [role="contentinfo"], header, main, nav, footer');
    await expect(landmarks.first()).toBeVisible();
    
    // Test skip links
    await page.keyboard.press('Tab');
    const firstFocusable = page.locator(':focus');
    const skipLink = await firstFocusable.textContent();
    
    if (skipLink && skipLink.toLowerCase().includes('skip')) {
      await firstFocusable.press('Enter');
      const mainContent = page.locator(':focus');
      await expect(mainContent).toBeVisible();
    }
    
    // Test aria-live regions for dynamic content
    const liveRegions = page.locator('[aria-live], [role="status"], [role="alert"]');
    if (await liveRegions.count() > 0) {
      await expect(liveRegions.first()).toBeVisible();
    }
  });

  test('keyboard navigation comprehensive test', async ({ page }) => {
    await page.goto('/');
    
    // Test tab order
    const interactiveElements = page.locator('a, button, input, textarea, select, [tabindex="0"], [role="button"], [role="link"]');
    const elementCount = await interactiveElements.count();
    
    let focusedElements = 0;
    
    for (let i = 0; i < Math.min(elementCount, 20); i++) {
      await page.keyboard.press('Tab');
      
      const focusedElement = page.locator(':focus');
      if (await focusedElement.count() > 0) {
        focusedElements++;
        
        // Verify focus is visible
        const focusedBox = await focusedElement.boundingBox();
        expect(focusedBox).toBeTruthy();
        
        // Test activation with Enter/Space
        const tagName = await focusedElement.evaluate(el => el.tagName.toLowerCase());
        const role = await focusedElement.getAttribute('role');
        
        if (tagName === 'button' || role === 'button') {
          // Test space activation for buttons
          await page.keyboard.press('Space');
          await page.waitForTimeout(100);
        } else if (tagName === 'a' || role === 'link') {
          // Test enter activation for links (but prevent navigation)
          // await page.keyboard.press('Enter');
        }
      }
    }
    
    expect(focusedElements).toBeGreaterThan(0);
    
    // Test reverse tab order
    await page.keyboard.press('Shift+Tab');
    const reverseFocused = page.locator(':focus');
    await expect(reverseFocused).toBeVisible();
  });
});