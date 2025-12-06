import { test, expect } from '@playwright/test'

// Test data for dual revenue streams
const SUBSCRIPTION_TIERS = {
  core: { price: 89, annual_discount: 20 },
  adaptive: { price: 149, annual_discount: 20 },
  performance: { price: 229, annual_discount: 20 },
  longevity: { price: 349, annual_discount: 20 }
}

const IN_PERSON_TRAINING = {
  gym: { price: 65, location: 'Gym Training', description: 'Professional gym environment' },
  home: { price: 95, location: 'Home Training', description: 'Convenience of your home' },
  citySports: { price: 85, location: 'City Sports', description: 'Outdoor urban training' }
}

const BILLING_CYCLES = ['monthly', 'quarterly', 'semiannual', 'annual']

test.describe('Dual Revenue Stream Validation', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/')
  })

  test('Pricing page shows both revenue streams', async ({ page }) => {
    await page.goto('/pricing')
    
    // Should have tabs for both revenue streams
    await expect(page.getByRole('tab', { name: 'Monthly Subscriptions' })).toBeVisible()
    await expect(page.getByRole('tab', { name: 'In-Person Training' })).toBeVisible()
    
    // Default view should be monthly subscriptions
    await expect(page.locator('[data-tier="core"]')).toBeVisible()
    await expect(page.getByText('Core Program')).toBeVisible()
    
    // Switch to in-person training tab
    await page.getByRole('tab', { name: 'In-Person Training' }).click()
    
    // Should show in-person training options
    await expect(page.locator('[data-location="gym"]')).toBeVisible()
    await expect(page.locator('[data-location="home"]')).toBeVisible()
    await expect(page.locator('[data-location="citySports"]')).toBeVisible()
  })

  test('Monthly subscription checkout flow', async ({ page }) => {
    await page.goto('/pricing')
    
    // Test each tier subscription flow
    for (const [tierId, tierConfig] of Object.entries(SUBSCRIPTION_TIERS)) {
      // Click subscribe button for tier
      await page.locator(`[data-tier="${tierId}"]`).getByText('Subscribe').click()
      
      // Should redirect to checkout with correct price
      await expect(page).toHaveURL(/checkout|stripe/)
      await expect(page.getByText(`$${tierConfig.price}`)).toBeVisible()
      
      // Go back to test next tier
      await page.goBack()
      await page.waitForLoadState('networkidle')
    }
  })

  test('Billing cycle price calculations', async ({ page }) => {
    await page.goto('/pricing')
    
    // Test annual discount calculations
    await page.getByText('Annual').click()
    
    for (const [tierId, tierConfig] of Object.entries(SUBSCRIPTION_TIERS)) {
      const monthlyPrice = tierConfig.price
      const annualPrice = monthlyPrice * 12 * (1 - tierConfig.annual_discount / 100)
      const annualPriceFormatted = Math.round(annualPrice)
      
      const tierCard = page.locator(`[data-tier="${tierId}"]`)
      // Should show discounted annual price
      await expect(tierCard.getByText(/\$\d+.*\/year/)).toBeVisible()
      await expect(tierCard.getByText('20% off')).toBeVisible()
    }
    
    // Test quarterly and semi-annual pricing
    await page.getByText('3 Months').click()
    await expect(page.getByText('5% off')).toBeVisible()
    
    await page.getByText('6 Months').click()
    await expect(page.getByText('10% off')).toBeVisible()
  })

  test('In-person training booking flow', async ({ page }) => {
    await page.goto('/pricing')
    await page.getByRole('tab', { name: 'In-Person Training' }).click()
    
    // Test each location booking
    for (const [locationId, locationConfig] of Object.entries(IN_PERSON_TRAINING)) {
      const locationCard = page.locator(`[data-location="${locationId}"]`)
      
      // Verify location details
      await expect(locationCard.getByText(locationConfig.location)).toBeVisible()
      await expect(locationCard.getByText(`$${locationConfig.price}`)).toBeVisible()
      await expect(locationCard.getByText(locationConfig.description)).toBeVisible()
      
      // Click book session
      await locationCard.getByText('Book Session').click()
      
      // Should open booking modal or redirect to booking page
      await expect(page.getByText('Schedule Session')).toBeVisible()
      await expect(page.getByText(`$${locationConfig.price}/session`)).toBeVisible()
      
      // Close modal and test next location
      await page.keyboard.press('Escape')
      await page.waitForTimeout(500)
    }
  })

  test('Combined subscription + in-person training flow', async ({ page }) => {
    // Login as Longevity tier user (has in-person training access)
    await page.goto('/auth')
    await page.getByPlaceholder('Email').fill('longevity-user@repztest.com')
    await page.getByPlaceholder('Password').fill('password123')
    await page.getByRole('button', { name: 'Sign In' }).click()
    
    // Navigate to dashboard
    await expect(page).toHaveURL('/dashboard')
    await expect(page.getByText('Longevity Concierge')).toBeVisible()
    
    // Should have access to in-person training booking
    await page.getByText('Book In-Person Session').click()
    
    // Should show location options
    await expect(page.getByText('Gym Training')).toBeVisible()
    await expect(page.getByText('Home Training')).toBeVisible()
    await expect(page.getByText('City Sports')).toBeVisible()
    
    // Book a gym session
    await page.locator('[data-location="gym"]').getByText('Book').click()
    
    // Fill booking form
    await page.getByText('Select Date').click()
    await page.getByText('15').click() // Select 15th of current month
    await page.getByText('Select Time').click()
    await page.getByText('10:00 AM').click()
    
    // Confirm booking
    await page.getByText('Confirm Booking').click()
    
    // Should show confirmation
    await expect(page.getByText('Session Booked Successfully')).toBeVisible()
    await expect(page.getByText('Gym Training - $65')).toBeVisible()
  })

  test('Pricing integration with Stripe', async ({ page }) => {
    await page.goto('/pricing')
    
    // Mock Stripe integration test
    await page.route('**/create-checkout', async route => {
      const request = route.request()
      const postData = request.postDataJSON()
      
      // Verify correct tier and pricing data is sent
      expect(postData.tier).toMatch(/core|adaptive|performance|longevity/)
      expect(postData.price).toBeGreaterThan(0)
      
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          url: 'https://checkout.stripe.com/pay/test_session',
          sessionId: 'cs_test_123'
        })
      })
    })
    
    // Test subscription checkout
    await page.locator('[data-tier="adaptive"]').getByText('Subscribe').click()
    
    // Should redirect to mocked Stripe checkout
    await expect(page).toHaveURL(/checkout\.stripe\.com/)
  })

  test('In-person training restrictions by tier', async ({ page }) => {
    // Test Core tier user (no in-person training access)
    await page.goto('/auth')
    await page.getByPlaceholder('Email').fill('core-user@repztest.com')
    await page.getByPlaceholder('Password').fill('password123')
    await page.getByRole('button', { name: 'Sign In' }).click()
    
    // Navigate to dashboard
    await expect(page).toHaveURL('/dashboard')
    
    // Should not have in-person training option
    await expect(page.getByText('Book In-Person Session')).not.toBeVisible()
    
    // Try to access in-person training directly
    await page.goto('/in-person')
    await expect(page.getByText('Upgrade to Longevity')).toBeVisible()
    await expect(page.getByText('In-person training requires Longevity tier')).toBeVisible()
  })

  test('Revenue stream analytics tracking', async ({ page }) => {
    // Mock analytics endpoints
    const analyticsEvents: any[] = []
    
    await page.route('**/analytics/**', async route => {
      const request = route.request()
      const postData = request.postDataJSON()
      analyticsEvents.push(postData)
      
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ success: true })
      })
    })
    
    await page.goto('/pricing')
    
    // Track subscription interest
    await page.locator('[data-tier="performance"]').hover()
    await page.locator('[data-tier="performance"]').getByText('Subscribe').click()
    
    // Go back and track in-person training interest
    await page.goBack()
    await page.getByRole('tab', { name: 'In-Person Training' }).click()
    await page.locator('[data-location="home"]').getByText('Book Session').click()
    
    // Verify analytics events were tracked
    await page.waitForTimeout(1000)
    expect(analyticsEvents.length).toBeGreaterThan(0)
    
    // Should track both subscription and in-person training events
    const subscriptionEvents = analyticsEvents.filter(e => e.eventType === 'subscription_interest')
    const trainingEvents = analyticsEvents.filter(e => e.eventType === 'training_interest')
    
    expect(subscriptionEvents.length).toBeGreaterThan(0)
    expect(trainingEvents.length).toBeGreaterThan(0)
  })

  test('Cross-selling between revenue streams', async ({ page }) => {
    // Start with subscription flow
    await page.goto('/pricing')
    await page.locator('[data-tier="adaptive"]').getByText('Subscribe').click()
    
    // Mock successful subscription
    await page.route('**/checkout/success**', async route => {
      await route.fulfill({
        status: 200,
        contentType: 'text/html',
        body: '<html><body><h1>Payment Successful</h1></body></html>'
      })
    })
    
    await page.goto('/checkout/success?tier=adaptive')
    
    // Should show cross-sell for in-person training
    await expect(page.getByText('Add In-Person Training')).toBeVisible()
    await expect(page.getByText('Enhance your Adaptive plan')).toBeVisible()
    
    // Click cross-sell CTA
    await page.getByText('Add Training Sessions').click()
    
    // Should redirect to in-person training booking
    await expect(page).toHaveURL(/in-person|training/)
    await expect(page.getByText('Select Training Location')).toBeVisible()
  })

  test('Subscription + training bundle pricing', async ({ page }) => {
    await page.goto('/pricing')
    
    // Should show bundle option for higher tiers
    await page.getByRole('tab', { name: 'Bundles' }).click()
    
    // Longevity tier should show included in-person training
    const longevityBundle = page.locator('[data-bundle="longevity-complete"]')
    await expect(longevityBundle.getByText('Longevity + Training')).toBeVisible()
    await expect(longevityBundle.getByText('$399/month')).toBeVisible()
    await expect(longevityBundle.getByText('Includes 2 in-person sessions')).toBeVisible()
    
    // Performance bundle option
    const performanceBundle = page.locator('[data-bundle="performance-plus"]')
    await expect(performanceBundle.getByText('Performance + Training')).toBeVisible()
    await expect(performanceBundle.getByText('$279/month')).toBeVisible()
    await expect(performanceBundle.getByText('Includes 1 in-person session')).toBeVisible()
  })

  test('Mobile responsive revenue stream navigation', async ({ page }) => {
    // Test mobile viewport
    await page.setViewportSize({ width: 375, height: 667 })
    await page.goto('/pricing')
    
    // Mobile should have compact tab navigation
    await expect(page.getByRole('tab', { name: 'Subscriptions' })).toBeVisible()
    await expect(page.getByRole('tab', { name: 'Training' })).toBeVisible()
    
    // Switch between tabs on mobile
    await page.getByRole('tab', { name: 'Training' }).click()
    await expect(page.locator('[data-location="gym"]')).toBeVisible()
    
    // Mobile booking flow should work
    await page.locator('[data-location="home"]').getByText('Book').click()
    await expect(page.getByText('Home Training')).toBeVisible()
    await expect(page.getByText('$95/session')).toBeVisible()
  })
})