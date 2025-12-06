import { test, expect } from '@playwright/test';

test.describe('Client Onboarding & Tier Selection', () => {
  test.beforeEach(async ({ page }) => {
    // Start fresh for each test
    await page.goto('/');
  });

  test('complete client onboarding flow with core tier selection', async ({ page }) => {
    // Step 1: Navigate to signup
    await page.click('[data-testid="signup-button"]');
    await expect(page).toHaveURL('/auth/signup');

    // Step 2: Create account
    const testEmail = `test-${Date.now()}@repz.com`;
    await page.fill('[data-testid="email-input"]', testEmail);
    await page.fill('[data-testid="password-input"]', 'TestPassword123!');
    await page.fill('[data-testid="confirm-password-input"]', 'TestPassword123!');
    await page.check('[data-testid="terms-checkbox"]');
    await page.click('[data-testid="create-account-button"]');

    // Step 3: Email verification (mock in test environment)
    await expect(page).toHaveURL('/auth/verify-email');
    // In test environment, auto-verify or mock the verification
    await page.goto('/intake/start');

    // Step 4: Complete 7-step intake form
    // Account Information
    await page.fill('[data-testid="first-name-input"]', 'John');
    await page.fill('[data-testid="last-name-input"]', 'Doe');
    await page.click('[data-testid="next-step-button"]');

    // Personal Information
    await page.fill('[data-testid="age-input"]', '30');
    await page.selectOption('[data-testid="gender-select"]', 'male');
    await page.fill('[data-testid="height-input"]', '180');
    await page.fill('[data-testid="weight-input"]', '80');
    await page.click('[data-testid="next-step-button"]');

    // Health Assessment
    await page.check('[data-testid="health-clearance-checkbox"]');
    await page.fill('[data-testid="medical-conditions-textarea"]', 'None');
    await page.click('[data-testid="next-step-button"]');

    // Training Experience
    await page.selectOption('[data-testid="experience-level-select"]', 'intermediate');
    await page.check('[data-testid="gym-access-checkbox"]');
    await page.click('[data-testid="next-step-button"]');

    // Nutrition Preferences
    await page.selectOption('[data-testid="diet-type-select"]', 'balanced');
    await page.fill('[data-testid="allergies-input"]', 'None');
    await page.click('[data-testid="next-step-button"]');

    // Goals & Objectives
    await page.check('[data-testid="goal-muscle-gain"]');
    await page.check('[data-testid="goal-fat-loss"]');
    await page.click('[data-testid="next-step-button"]');

    // Step 5: Tier Selection
    await expect(page).toHaveURL('/intake/tier-selection');
    
    // Verify all tiers are displayed
    await expect(page.locator('[data-testid="core-tier-card"]')).toBeVisible();
    await expect(page.locator('[data-testid="adaptive-tier-card"]')).toBeVisible();
    await expect(page.locator('[data-testid="performance-tier-card"]')).toBeVisible();
    await expect(page.locator('[data-testid="longevity-tier-card"]')).toBeVisible();

    // Select Core tier
    await page.click('[data-testid="core-tier-select-button"]');
    
    // Verify tier selection
    await expect(page.locator('[data-testid="selected-tier"]')).toHaveText('Core Program');
    await expect(page.locator('[data-testid="tier-price"]')).toHaveText('$89/month');

    // Step 6: Payment Processing
    await page.click('[data-testid="proceed-to-payment-button"]');
    await expect(page).toHaveURL('/payment/checkout');

    // Fill payment details (Stripe test card)
    await page.fill('[data-testid="card-number-input"]', '4242424242424242');
    await page.fill('[data-testid="card-expiry-input"]', '12/25');
    await page.fill('[data-testid="card-cvc-input"]', '123');
    await page.fill('[data-testid="cardholder-name-input"]', 'John Doe');

    // Process payment
    await page.click('[data-testid="complete-payment-button"]');

    // Step 7: Subscription Activation & Dashboard Access
    await expect(page).toHaveURL('/dashboard', { timeout: 30000 });
    
    // Verify subscription is active
    await expect(page.locator('[data-testid="subscription-status"]')).toHaveText('Active');
    await expect(page.locator('[data-testid="current-tier"]')).toHaveText('Core');

    // Step 8: Feature Verification by Tier
    // Core tier should have basic features
    await expect(page.locator('[data-testid="basic-workout-plans"]')).toBeVisible();
    await expect(page.locator('[data-testid="basic-nutrition-guide"]')).toBeVisible();
    
    // Should NOT have advanced features
    await expect(page.locator('[data-testid="advanced-protocols"]')).not.toBeVisible();
    await expect(page.locator('[data-testid="peds-management"]')).not.toBeVisible();
    await expect(page.locator('[data-testid="ai-assistant"]')).not.toBeVisible();
  });

  test('tier feature access control validation', async ({ page }) => {
    // Test with different tier auth states
    const tiers = ['core', 'adaptive', 'performance', 'longevity'];
    
    for (const tier of tiers) {
      // Switch to tier-specific auth state
      await page.context().storageState({ 
        path: `tests/fixtures/auth/${tier}-auth.json` 
      });
      
      await page.goto('/dashboard');
      
      // Verify tier-specific features
      switch (tier) {
        case 'core':
          await expect(page.locator('[data-testid="basic-features"]')).toBeVisible();
          await expect(page.locator('[data-testid="advanced-nutrition"]')).not.toBeVisible();
          break;
        case 'adaptive':
          await expect(page.locator('[data-testid="basic-features"]')).toBeVisible();
          await expect(page.locator('[data-testid="advanced-nutrition"]')).toBeVisible();
          await expect(page.locator('[data-testid="peds-protocols"]')).not.toBeVisible();
          break;
        case 'performance':
          await expect(page.locator('[data-testid="advanced-nutrition"]')).toBeVisible();
          await expect(page.locator('[data-testid="peds-protocols"]')).toBeVisible();
          await expect(page.locator('[data-testid="longevity-concierge"]')).not.toBeVisible();
          break;
        case 'longevity':
          await expect(page.locator('[data-testid="longevity-concierge"]')).toBeVisible();
          await expect(page.locator('[data-testid="premium-features"]')).toBeVisible();
          break;
      }
    }
  });

  test('intake form validation and error handling', async ({ page }) => {
    await page.goto('/intake/start');

    // Test required field validation
    await page.click('[data-testid="next-step-button"]');
    await expect(page.locator('[data-testid="error-first-name"]')).toBeVisible();
    await expect(page.locator('[data-testid="error-last-name"]')).toBeVisible();

    // Test invalid email format
    await page.fill('[data-testid="email-input"]', 'invalid-email');
    await page.blur('[data-testid="email-input"]');
    await expect(page.locator('[data-testid="error-email-format"]')).toBeVisible();

    // Test password strength requirements
    await page.fill('[data-testid="password-input"]', 'weak');
    await page.blur('[data-testid="password-input"]');
    await expect(page.locator('[data-testid="error-password-strength"]')).toBeVisible();

    // Test age validation
    await page.fill('[data-testid="age-input"]', '15');
    await page.blur('[data-testid="age-input"]');
    await expect(page.locator('[data-testid="error-age-minimum"]')).toBeVisible();
  });

  test('payment failure handling', async ({ page }) => {
    // Complete intake flow up to payment
    await page.goto('/intake/start');
    // ... complete intake steps (abbreviated for test focus)
    await page.goto('/payment/checkout');

    // Use declined test card
    await page.fill('[data-testid="card-number-input"]', '4000000000000002');
    await page.fill('[data-testid="card-expiry-input"]', '12/25');
    await page.fill('[data-testid="card-cvc-input"]', '123');
    await page.fill('[data-testid="cardholder-name-input"]', 'John Doe');

    await page.click('[data-testid="complete-payment-button"]');

    // Verify error handling
    await expect(page.locator('[data-testid="payment-error"]')).toBeVisible();
    await expect(page.locator('[data-testid="payment-error"]')).toContainText('declined');
    
    // Verify user can retry
    await expect(page.locator('[data-testid="retry-payment-button"]')).toBeVisible();
  });

  test('mobile responsive onboarding flow', async ({ page }) => {
    // Set mobile viewport
    await page.setViewportSize({ width: 375, height: 667 });
    
    await page.goto('/auth/signup');
    
    // Verify mobile-optimized layout
    await expect(page.locator('[data-testid="mobile-signup-form"]')).toBeVisible();
    
    // Test mobile form interactions
    await page.fill('[data-testid="email-input"]', 'mobile@test.com');
    await page.fill('[data-testid="password-input"]', 'TestPassword123!');
    
    // Verify mobile keyboard handling
    await page.locator('[data-testid="email-input"]').focus();
    await expect(page.locator('[data-testid="email-input"]')).toBeFocused();
    
    // Test mobile navigation
    await page.click('[data-testid="create-account-button"]');
    
    // Verify responsive intake form
    await page.goto('/intake/start');
    await expect(page.locator('[data-testid="mobile-intake-form"]')).toBeVisible();
  });
});