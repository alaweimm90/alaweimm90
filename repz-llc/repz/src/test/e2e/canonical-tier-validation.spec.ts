import { test, expect } from '@playwright/test'

// Test data for canonical tier system
const CANONICAL_TIERS = {
  core: {
    displayName: 'Core Program',
    price: '$89',
    features: ['basic_workouts', 'nutrition_tracking', 'progress_photos'],
    restrictedFeatures: ['weekly_checkins', 'form_review', 'ai_assistant']
  },
  adaptive: {
    displayName: 'Adaptive Engine', 
    price: '$149',
    features: ['weekly_checkins', 'form_review', 'wearable_sync', 'biomarker_integration'],
    restrictedFeatures: ['ai_fitness_assistant', 'peds', 'travel_workouts']
  },
  performance: {
    displayName: 'Performance Suite',
    price: '$229', 
    features: ['ai_fitness_assistant', 'peds', 'travel_workouts', 'nootropics'],
    restrictedFeatures: ['bioregulators', 'custom_cycling_schemes', 'in_person_training']
  },
  longevity: {
    displayName: 'Longevity Concierge',
    price: '$349',
    features: ['bioregulators', 'custom_cycling_schemes', 'in_person_training', 'concierge_service'],
    restrictedFeatures: [] // Full access
  }
}

const BILLING_CYCLES = {
  monthly: { displayName: 'Monthly', discount: '0%' },
  quarterly: { displayName: '3 Months', discount: '5% off' },
  semiannual: { displayName: '6 Months', discount: '10% off' },
  annual: { displayName: 'Annual', discount: '20% off - 2 months FREE' }
}

test.describe('Canonical Tier System Validation', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/')
  })

  test('Pricing page displays all canonical tiers correctly', async ({ page }) => {
    await page.goto('/pricing')
    
    // Verify all 4 canonical tiers are displayed
    for (const [tierId, tierConfig] of Object.entries(CANONICAL_TIERS)) {
      const tierCard = page.locator(`[data-tier="${tierId}"]`)
      await expect(tierCard).toBeVisible()
      await expect(tierCard.getByText(tierConfig.displayName)).toBeVisible()
      await expect(tierCard.getByText(tierConfig.price)).toBeVisible()
    }
    
    // Verify no deprecated tier names are present
    const deprecatedNames = ['Baseline', 'Prime', 'Precision']
    for (const deprecatedName of deprecatedNames) {
      await expect(page.getByText(deprecatedName)).not.toBeVisible()
    }
  })

  test('Billing cycle options display correctly', async ({ page }) => {
    await page.goto('/pricing')
    
    // Test each billing cycle
    for (const [cycleId, cycleConfig] of Object.entries(BILLING_CYCLES)) {
      await page.getByText(cycleConfig.displayName).click()
      await expect(page.getByText(cycleConfig.discount)).toBeVisible()
      
      // Verify pricing updates for cycle
      if (cycleId === 'annual') {
        // Annual pricing should show savings
        await expect(page.getByText('2 months FREE')).toBeVisible()
      }
    }
  })

  test('Core tier user journey and restrictions', async ({ page }) => {
    // Simulate Core tier user login
    await page.goto('/auth')
    await page.getByPlaceholder('Email').fill('core-user@repztest.com')
    await page.getByPlaceholder('Password').fill('password123')
    await page.getByRole('button', { name: 'Sign In' }).click()
    
    // Should reach dashboard
    await expect(page).toHaveURL('/dashboard')
    await expect(page.getByText('Core Program')).toBeVisible()
    
    // Test access to allowed features
    await page.goto('/workouts')
    await expect(page.getByText('Basic Workouts')).toBeVisible()
    
    // Test restrictions - weekly check-ins should show upgrade prompt
    await page.goto('/check-ins')
    await expect(page.getByText('Upgrade to Adaptive')).toBeVisible()
    await expect(page.getByText('Weekly check-ins require Adaptive tier')).toBeVisible()
  })

  test('Adaptive tier user journey and feature access', async ({ page }) => {
    // Simulate Adaptive tier user login
    await page.goto('/auth')
    await page.getByPlaceholder('Email').fill('adaptive-user@repztest.com')
    await page.getByPlaceholder('Password').fill('password123')
    await page.getByRole('button', { name: 'Sign In' }).click()
    
    // Should reach dashboard with Adaptive features
    await expect(page).toHaveURL('/dashboard')
    await expect(page.getByText('Adaptive Engine')).toBeVisible()
    
    // Test access to Adaptive features
    await page.goto('/check-ins')
    await expect(page.getByText('Weekly Check-ins')).toBeVisible()
    await expect(page.getByText('Schedule Check-in')).toBeVisible()
    
    // Test form review access
    await page.goto('/form-review')
    await expect(page.getByText('Form Analysis')).toBeVisible()
    
    // Test AI assistant restriction
    await page.goto('/ai-assistant')
    await expect(page.getByText('Upgrade to Performance')).toBeVisible()
    await expect(page.getByText('AI fitness assistant requires Performance tier')).toBeVisible()
  })

  test('Performance tier user journey and advanced features', async ({ page }) => {
    // Simulate Performance tier user login
    await page.goto('/auth')
    await page.getByPlaceholder('Email').fill('performance-user@repztest.com')
    await page.getByPlaceholder('Password').fill('password123')
    await page.getByRole('button', { name: 'Sign In' }).click()
    
    // Should reach dashboard with Performance features
    await expect(page).toHaveURL('/dashboard')
    await expect(page.getByText('Performance Suite')).toBeVisible()
    
    // Test AI assistant access
    await page.goto('/ai-assistant')
    await expect(page.getByText('AI Fitness Assistant')).toBeVisible()
    await expect(page.getByPlaceholder('Ask about your training')).toBeVisible()
    
    // Test PEDs protocols access
    await page.goto('/protocols/peds')
    await expect(page.getByText('Performance Enhancement')).toBeVisible()
    
    // Test bioregulators restriction (Longevity only)
    await page.goto('/protocols/bioregulators')
    await expect(page.getByText('Upgrade to Longevity')).toBeVisible()
  })

  test('Longevity tier user journey - full access', async ({ page }) => {
    // Simulate Longevity tier user login
    await page.goto('/auth')
    await page.getByPlaceholder('Email').fill('longevity-user@repztest.com')
    await page.getByPlaceholder('Password').fill('password123')
    await page.getByRole('button', { name: 'Sign In' }).click()
    
    // Should reach dashboard with full access
    await expect(page).toHaveURL('/dashboard')
    await expect(page.getByText('Longevity Concierge')).toBeVisible()
    
    // Test bioregulators access
    await page.goto('/protocols/bioregulators')
    await expect(page.getByText('Bioregulator Protocols')).toBeVisible()
    
    // Test in-person training access
    await page.goto('/in-person')
    await expect(page.getByText('Schedule In-Person Session')).toBeVisible()
    
    // Test concierge service
    await page.goto('/concierge')
    await expect(page.getByText('Personal Concierge')).toBeVisible()
    await expect(page.getByText('24/7 Priority Support')).toBeVisible()
    
    // Verify no upgrade prompts are shown
    const upgradeButtons = page.locator('text="Upgrade"')
    await expect(upgradeButtons).toHaveCount(0)
  })

  test('Tier upgrade flow validation', async ({ page }) => {
    // Start as Core user
    await page.goto('/auth')
    await page.getByPlaceholder('Email').fill('core-user@repztest.com')
    await page.getByPlaceholder('Password').fill('password123')
    await page.getByRole('button', { name: 'Sign In' }).click()
    
    // Navigate to restricted feature
    await page.goto('/ai-assistant')
    await expect(page.getByText('Upgrade to Performance')).toBeVisible()
    
    // Click upgrade button
    await page.getByText('Upgrade Now').click()
    
    // Should redirect to pricing with Performance tier highlighted
    await expect(page).toHaveURL(/\/pricing/)
    await expect(page.locator('[data-tier="performance"]')).toHaveClass(/highlighted|selected/)
    
    // Click subscribe on Performance tier
    await page.locator('[data-tier="performance"]').getByText('Subscribe').click()
    
    // Should redirect to Stripe checkout
    await expect(page).toHaveURL(/stripe\.com\/checkout|checkout/)
  })

  test('Feature matrix consistency across components', async ({ page }) => {
    await page.goto('/pricing')
    
    // Test feature comparison table
    await page.getByText('Compare Features').click()
    
    // Verify Core tier features
    const coreColumn = page.locator('[data-tier-column="core"]')
    await expect(coreColumn.getByText('Basic Workouts')).toBeVisible()
    await expect(coreColumn.locator('[data-feature="weekly_checkins"]')).toHaveClass(/disabled|unavailable/)
    
    // Verify Adaptive tier features
    const adaptiveColumn = page.locator('[data-tier-column="adaptive"]')
    await expect(adaptiveColumn.locator('[data-feature="weekly_checkins"]')).toHaveClass(/enabled|available/)
    await expect(adaptiveColumn.locator('[data-feature="ai_assistant"]')).toHaveClass(/disabled|unavailable/)
    
    // Verify Performance tier features
    const performanceColumn = page.locator('[data-tier-column="performance"]')
    await expect(performanceColumn.locator('[data-feature="ai_assistant"]')).toHaveClass(/enabled|available/)
    await expect(performanceColumn.locator('[data-feature="bioregulators"]')).toHaveClass(/disabled|unavailable/)
    
    // Verify Longevity tier features (all enabled)
    const longevityColumn = page.locator('[data-tier-column="longevity"]')
    const longevityFeatures = longevityColumn.locator('[data-feature]')
    const featureCount = await longevityFeatures.count()
    
    for (let i = 0; i < featureCount; i++) {
      await expect(longevityFeatures.nth(i)).toHaveClass(/enabled|available/)
    }
  })

  test('Subscription management with canonical tiers', async ({ page }) => {
    // Login as existing user
    await page.goto('/auth')
    await page.getByPlaceholder('Email').fill('adaptive-user@repztest.com')
    await page.getByPlaceholder('Password').fill('password123')
    await page.getByRole('button', { name: 'Sign In' }).click()
    
    // Navigate to subscription management
    await page.goto('/account/subscription')
    
    // Current tier should be displayed correctly
    await expect(page.getByText('Current Plan: Adaptive Engine')).toBeVisible()
    await expect(page.getByText('$149/month')).toBeVisible()
    
    // Test tier upgrade options
    await page.getByText('Upgrade Plan').click()
    await expect(page.getByText('Performance Suite')).toBeVisible()
    await expect(page.getByText('Longevity Concierge')).toBeVisible()
    
    // Should not show Core tier as downgrade option in upgrade flow
    await expect(page.getByText('Core Program')).not.toBeVisible()
    
    // Test billing cycle change
    await page.getByText('Change Billing').click()
    await page.getByText('Annual').click()
    await expect(page.getByText('20% off')).toBeVisible()
  })
})

test.describe('Deprecated Tier Name Prevention', () => {
  test('API endpoints reject deprecated tier names', async ({ page, request }) => {
    // Test API directly to ensure backend validation
    const response = await request.post('/api/subscription/update', {
      data: { tier: 'baseline' } // Deprecated name
    })
    
    await expect(response.status()).toBe(400)
    const responseBody = await response.json()
    await expect(responseBody.error).toContain('Invalid tier')
  })

  test('Form submissions with deprecated names are rejected', async ({ page }) => {
    await page.goto('/admin/user-management')
    
    // Try to create user with deprecated tier name
    await page.getByPlaceholder('Email').fill('test@example.com')
    await page.getByRole('combobox', { name: 'Tier' }).click()
    
    // Deprecated options should not be available
    await expect(page.getByText('Baseline')).not.toBeVisible()
    await expect(page.getByText('Prime')).not.toBeVisible()
    await expect(page.getByText('Precision')).not.toBeVisible()
    
    // Only canonical options should be available
    await expect(page.getByText('Core Program')).toBeVisible()
    await expect(page.getByText('Adaptive Engine')).toBeVisible()
    await expect(page.getByText('Performance Suite')).toBeVisible()
    await expect(page.getByText('Longevity Concierge')).toBeVisible()
  })
})