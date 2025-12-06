import { test, expect } from '@playwright/test';

test.describe('Mobile Touch Target Accessibility', () => {
  test.use({ viewport: { width: 375, height: 667 } }); // iPhone SE viewport

  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');
  });

  test('All interactive elements meet 44px minimum touch target', async ({ page }) => {
    // Get all interactive elements
    const interactiveSelectors = [
      'button',
      'a[href]',
      'input',
      'select', 
      'textarea',
      '[role="button"]',
      '[role="tab"]',
      '[role="menuitem"]',
      '[tabindex]:not([tabindex="-1"])'
    ];
    
    for (const selector of interactiveSelectors) {
      const elements = page.locator(selector);
      const count = await elements.count();
      
      for (let i = 0; i < count; i++) {
        const element = elements.nth(i);
        const isVisible = await element.isVisible();
        
        if (isVisible) {
          const box = await element.boundingBox();
          if (box) {
            expect(box.width).toBeGreaterThanOrEqual(44);
            expect(box.height).toBeGreaterThanOrEqual(44);
          }
        }
      }
    }
  });

  test('Quantum gate buttons are touch-friendly', async ({ page }) => {
    const gateButtons = page.locator('.gate-palette button, [data-gate] button, button[aria-label*="gate"]');
    const count = await gateButtons.count();
    
    if (count > 0) {
      for (let i = 0; i < Math.min(count, 8); i++) {
        const button = gateButtons.nth(i);
        const isVisible = await button.isVisible();
        
        if (isVisible) {
          const box = await button.boundingBox();
          if (box) {
            // Quantum gates should be at least 56px for optimal touch interaction
            expect(box.width).toBeGreaterThanOrEqual(56);
            expect(box.height).toBeGreaterThanOrEqual(56);
          }
        }
      }
    }
  });

  test('Circuit wire interaction points are accessible', async ({ page }) => {
    const circuitPoints = page.locator('.circuit-wire button, .gate-position button, [data-circuit-position]');
    const count = await circuitPoints.count();
    
    if (count > 0) {
      for (let i = 0; i < Math.min(count, 5); i++) {
        const point = circuitPoints.nth(i);
        const isVisible = await point.isVisible();
        
        if (isVisible) {
          const box = await point.boundingBox();
          if (box) {
            expect(box.width).toBeGreaterThanOrEqual(48);
            expect(box.height).toBeGreaterThanOrEqual(48);
          }
        }
      }
    }
  });

  test('Mobile navigation touch targets', async ({ page }) => {
    // Check mobile menu button
    const menuButton = page.locator('button[aria-label*="menu"], .mobile-menu-button, button:has(svg)').first();
    
    if (await menuButton.count() > 0) {
      const box = await menuButton.boundingBox();
      if (box) {
        expect(box.width).toBeGreaterThanOrEqual(44);
        expect(box.height).toBeGreaterThanOrEqual(44);
      }
    }
    
    // Check navigation links
    const navLinks = page.locator('nav a, .nav-link');
    const linkCount = await navLinks.count();
    
    for (let i = 0; i < Math.min(linkCount, 5); i++) {
      const link = navLinks.nth(i);
      const isVisible = await link.isVisible();
      
      if (isVisible) {
        const box = await link.boundingBox();
        if (box) {
          expect(box.height).toBeGreaterThanOrEqual(44);
        }
      }
    }
  });

  test('Form controls are touch-friendly', async ({ page }) => {
    const formControls = page.locator('input, select, textarea, button[type="submit"]');
    const count = await formControls.count();
    
    for (let i = 0; i < count; i++) {
      const control = formControls.nth(i);
      const isVisible = await control.isVisible();
      
      if (isVisible) {
        const box = await control.boundingBox();
        if (box) {
          expect(box.height).toBeGreaterThanOrEqual(44);
          
          // Text inputs should be taller for better usability
          const tagName = await control.evaluate(el => el.tagName.toLowerCase());
          if (['input', 'textarea'].includes(tagName)) {
            expect(box.height).toBeGreaterThanOrEqual(48);
          }
        }
      }
    }
  });

  test('Touch target spacing prevents accidental activation', async ({ page }) => {
    const buttons = page.locator('button:visible');
    const count = await buttons.count();
    
    if (count > 1) {
      for (let i = 0; i < count - 1; i++) {
        const button1 = buttons.nth(i);
        const button2 = buttons.nth(i + 1);
        
        const box1 = await button1.boundingBox();
        const box2 = await button2.boundingBox();
        
        if (box1 && box2) {
          // Check if buttons are in the same row (similar y position)
          const sameRow = Math.abs(box1.y - box2.y) < 20;
          
          if (sameRow) {
            // Minimum 8px spacing between touch targets
            const spacing = Math.abs(box2.x - (box1.x + box1.width));
            expect(spacing).toBeGreaterThanOrEqual(8);
          }
        }
      }
    }
  });

  test('Swipe gestures and scroll areas work correctly', async ({ page }) => {
    // Test horizontal scrolling in gate palette
    const gatePalette = page.locator('.gate-palette, .horizontal-scroll').first();
    
    if (await gatePalette.count() > 0) {
      // Should be scrollable horizontally
      const isScrollable = await gatePalette.evaluate(el => {
        return el.scrollWidth > el.clientWidth;
      });
      
      if (isScrollable) {
        // Test scroll behavior
        await gatePalette.evaluate(el => el.scrollLeft = 50);
        const scrollLeft = await gatePalette.evaluate(el => el.scrollLeft);
        expect(scrollLeft).toBeGreaterThan(0);
      }
    }
    
    // Test vertical scrolling areas
    const scrollAreas = page.locator('.scroll-area, [data-scroll]');
    const scrollCount = await scrollAreas.count();
    
    for (let i = 0; i < Math.min(scrollCount, 2); i++) {
      const scrollArea = scrollAreas.nth(i);
      const isVisible = await scrollArea.isVisible();
      
      if (isVisible) {
        const isScrollable = await scrollArea.evaluate(el => {
          return el.scrollHeight > el.clientHeight;
        });
        
        // If scrollable, should respond to touch
        if (isScrollable) {
          const initialScrollTop = await scrollArea.evaluate(el => el.scrollTop);
          await scrollArea.evaluate(el => el.scrollTop = initialScrollTop + 50);
          const newScrollTop = await scrollArea.evaluate(el => el.scrollTop);
          expect(newScrollTop).toBeGreaterThanOrEqual(initialScrollTop);
        }
      }
    }
  });

  test('Tablet viewport touch targets (iPad)', async ({ page }) => {
    // Change to tablet viewport
    await page.setViewportSize({ width: 768, height: 1024 });
    await page.reload();
    await page.waitForLoadState('networkidle');
    
    // On tablets, touch targets can be slightly smaller but still accessible
    const interactiveElements = page.locator('button, a[href], input, select');
    const count = await interactiveElements.count();
    
    for (let i = 0; i < Math.min(count, 10); i++) {
      const element = interactiveElements.nth(i);
      const isVisible = await element.isVisible();
      
      if (isVisible) {
        const box = await element.boundingBox();
        if (box) {
          // Tablet minimum is 40px
          expect(box.width).toBeGreaterThanOrEqual(40);
          expect(box.height).toBeGreaterThanOrEqual(40);
        }
      }
    }
  });

  test('Landscape orientation touch targets', async ({ page }) => {
    // Switch to landscape
    await page.setViewportSize({ width: 667, height: 375 });
    await page.reload();
    await page.waitForLoadState('networkidle');
    
    // Touch targets should maintain size in landscape
    const criticalButtons = page.locator('button[aria-label*="run"], button[aria-label*="play"], .primary-action');
    const count = await criticalButtons.count();
    
    for (let i = 0; i < count; i++) {
      const button = criticalButtons.nth(i);
      const isVisible = await button.isVisible();
      
      if (isVisible) {
        const box = await button.boundingBox();
        if (box) {
          expect(box.width).toBeGreaterThanOrEqual(44);
          expect(box.height).toBeGreaterThanOrEqual(44);
        }
      }
    }
  });
});