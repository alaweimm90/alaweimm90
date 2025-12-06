import { test, expect } from '@playwright/test';

test.describe('Quantum Component Accessibility', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    // Wait for quantum components to load
    await page.waitForLoadState('networkidle');
  });

  test('Circuit Builder accessibility', async ({ page }) => {
    // Navigate to circuit builder if needed
    const circuitBuilder = page.locator('[data-component="circuit-builder"], .circuit-builder').first();
    
    if (await circuitBuilder.count() > 0) {
      // Check for proper labeling
      await expect(circuitBuilder).toHaveAttribute('role', 'region');
      
      // Check for instructions
      const instructions = page.locator('[role="region"] p, .instruction, .help-text').first();
      await expect(instructions).toBeVisible();
      
      // Test gate buttons accessibility
      const gateButtons = page.locator('button[aria-label*="gate"], button[aria-label*="Gate"]');
      const gateCount = await gateButtons.count();
      
      if (gateCount > 0) {
        for (let i = 0; i < Math.min(gateCount, 5); i++) {
          const button = gateButtons.nth(i);
          await expect(button).toHaveAttribute('aria-label');
          
          // Check minimum touch target size (44px)
          const box = await button.boundingBox();
          expect(box?.width).toBeGreaterThanOrEqual(44);
          expect(box?.height).toBeGreaterThanOrEqual(44);
        }
      }
      
      // Test keyboard navigation
      await page.keyboard.press('Tab');
      const focusedElement = page.locator(':focus');
      await expect(focusedElement).toBeVisible();
    }
  });

  test('Bloch Sphere accessibility', async ({ page }) => {
    const blochSphere = page.locator('[data-component="bloch-sphere"], .bloch-sphere, canvas').first();
    
    if (await blochSphere.count() > 0) {
      // Check for proper labeling and description
      const container = blochSphere.locator('..').first();
      
      // Should have either aria-label or aria-describedby
      const hasAriaLabel = await blochSphere.getAttribute('aria-label');
      const hasAriaDescribedBy = await blochSphere.getAttribute('aria-describedby');
      const hasDescription = page.locator('.sr-only, .visually-hidden').first();
      
      expect(hasAriaLabel || hasAriaDescribedBy || (await hasDescription.count() > 0)).toBeTruthy();
      
      // Check for alternative text representation
      const textAlternative = page.locator('[data-quantum-state], .quantum-state-text').first();
      if (await textAlternative.count() > 0) {
        await expect(textAlternative).toBeVisible();
      }
    }
  });

  test('Training Dashboard accessibility', async ({ page }) => {
    const trainingDashboard = page.locator('[data-component="training-dashboard"], .training-dashboard').first();
    
    if (await trainingDashboard.count() > 0) {
      // Check for form labels
      const inputs = trainingDashboard.locator('input, select, textarea');
      const inputCount = await inputs.count();
      
      for (let i = 0; i < inputCount; i++) {
        const input = inputs.nth(i);
        const hasLabel = await input.getAttribute('aria-label');
        const hasLabelledBy = await input.getAttribute('aria-labelledby');
        const associatedLabel = page.locator(`label[for="${await input.getAttribute('id')}"]`);
        
        expect(hasLabel || hasLabelledBy || (await associatedLabel.count() > 0)).toBeTruthy();
      }
      
      // Check training controls
      const playButton = page.locator('button[aria-label*="start"], button[aria-label*="play"], button:has-text("Start")').first();
      if (await playButton.count() > 0) {
        await expect(playButton).toHaveAttribute('aria-label');
      }
    }
  });

  test('Quantum state visualization screen reader support', async ({ page }) => {
    // Look for quantum state representations
    const stateElements = page.locator('[data-quantum-state], .quantum-state, .state-vector').first();
    
    if (await stateElements.count() > 0) {
      // Should have text alternative or description
      const hasTextContent = await stateElements.textContent();
      const hasAriaLabel = await stateElements.getAttribute('aria-label');
      const hasAriaDescribedBy = await stateElements.getAttribute('aria-describedby');
      
      expect(hasTextContent || hasAriaLabel || hasAriaDescribedBy).toBeTruthy();
    }
  });

  test('Quantum gates keyboard interaction', async ({ page }) => {
    const gateButtons = page.locator('button[data-gate], button[aria-label*="gate"]');
    const buttonCount = await gateButtons.count();
    
    if (buttonCount > 0) {
      // Test keyboard focus and activation
      await page.keyboard.press('Tab');
      
      // Find the first focusable gate button
      for (let i = 0; i < Math.min(buttonCount, 3); i++) {
        const button = gateButtons.nth(i);
        await button.focus();
        
        // Test Enter and Space activation
        const isVisible = await button.isVisible();
        if (isVisible) {
          await page.keyboard.press('Enter');
          
          // Check if action occurred (button press registered)
          const ariaPressed = await button.getAttribute('aria-pressed');
          const isSelected = await button.evaluate(el => el.classList.contains('selected') || el.classList.contains('active'));
          
          // At least one indication of interaction should be present
          expect(ariaPressed !== null || isSelected).toBeTruthy();
        }
      }
    }
  });

  test('Complex quantum operations accessibility', async ({ page }) => {
    // Test multi-qubit gates and complex operations
    const complexControls = page.locator('[data-complex-operation], .multi-qubit, [aria-label*="entangl"]').first();
    
    if (await complexControls.count() > 0) {
      // Should have detailed descriptions for complex operations
      const hasDescription = await complexControls.getAttribute('aria-describedby');
      const hasDetailedLabel = await complexControls.getAttribute('aria-label');
      
      expect(hasDescription || (hasDetailedLabel && hasDetailedLabel.length > 10)).toBeTruthy();
      
      // Check for help or explanation text
      const helpText = page.locator('.help-text, .explanation, [role="tooltip"]').first();
      if (await helpText.count() > 0) {
        await expect(helpText).toBeVisible();
      }
    }
  });

  test('Manual accessibility checks on quantum components', async ({ page }) => {
    // Check for proper ARIA attributes
    const components = page.locator('[data-component], [role="region"]');
    const count = await components.count();
    
    for (let i = 0; i < Math.min(count, 3); i++) {
      const component = components.nth(i);
      const hasAriaLabel = await component.getAttribute('aria-label');
      const hasAriaDescribedBy = await component.getAttribute('aria-describedby');
      const hasRole = await component.getAttribute('role');
      
      // At least one accessibility attribute should be present
      expect(hasAriaLabel || hasAriaDescribedBy || hasRole).toBeTruthy();
    }
  });

  test('WebGL fallback accessibility', async ({ page, browserName }) => {
    // This test specifically runs with WebGL disabled (see no-webgl project)
    if (page.context().browser()?.browserType().name() === 'chromium') {
      // Check that quantum visualizations still work without WebGL
      const canvasElements = page.locator('canvas');
      const canvasCount = await canvasElements.count();
      
      if (canvasCount > 0) {
        // Should have fallback content or alternative representation
        const fallbackContent = page.locator('.webgl-fallback, .canvas-fallback, [data-fallback]').first();
        const alternativeText = page.locator('.quantum-state-text, [data-quantum-text]').first();
        
        expect(await fallbackContent.count() > 0 || await alternativeText.count() > 0).toBeTruthy();
      }
    }
  });

  test('Quantum animation accessibility with reduced motion', async ({ page }) => {
    // Test with reduced motion preference (handled by reduced-motion project)
    const animatedElements = page.locator('.animate-, [style*="animation"], .quantum-animation');
    const animationCount = await animatedElements.count();
    
    if (animationCount > 0) {
      for (let i = 0; i < Math.min(animationCount, 3); i++) {
        const element = animatedElements.nth(i);
        const computedStyle = await element.evaluate(el => {
          return window.getComputedStyle(el).animationDuration;
        });
        
        // With reduced motion, animations should be very short or disabled
        expect(computedStyle === '0s' || computedStyle === '0.01s').toBeTruthy();
      }
    }
  });
});