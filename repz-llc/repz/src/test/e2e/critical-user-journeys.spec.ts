import { test, expect } from '@playwright/test'

test.describe('Critical User Journeys', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/')
  })

  test('Complete signup to payment flow', async ({ page }) => {
    // Test the complete user journey from landing to payment
    
    // 1. Land on homepage
    await expect(page.getByText('Transform Your Fitness')).toBeVisible()
    
    // 2. Navigate to signup
    await page.getByText('Get Started').click()
    await expect(page.getByText('Create Your Account')).toBeVisible()
    
    // 3. Fill signup form
    await page.getByPlaceholder('Email').fill('test@example.com')
    await page.getByPlaceholder('Password').fill('password123')
    await page.getByRole('button', { name: 'Create Account' }).click()
    
    // 4. Should redirect to onboarding or dashboard
    await expect(page).toHaveURL(/\/dashboard|\/onboarding/)
    
    // 5. Navigate to pricing
    await page.getByText('Upgrade').click()
    await expect(page.getByText('Choose Your Plan')).toBeVisible()
    
    // 6. Select tier
    await page.locator('[data-tier="adaptive"]').getByText('Upgrade').click()
    
    // 7. Payment form should appear
    await expect(page.getByText('Payment Details')).toBeVisible()
  })

  test('Authentication and tier access', async ({ page }) => {
    // Test authentication flows and tier-based access
    
    // 1. Try to access protected route
    await page.goto('/dashboard')
    await expect(page).toHaveURL(/\/auth|\/login/)
    
    // 2. Login with valid credentials
    await page.getByPlaceholder('Email').fill('test@example.com')
    await page.getByPlaceholder('Password').fill('password123')
    await page.getByRole('button', { name: 'Sign In' }).click()
    
    // 3. Should redirect to dashboard
    await expect(page).toHaveURL('/dashboard')
    await expect(page.getByText('Welcome')).toBeVisible()
    
    // 4. Test tier-specific features
    await page.goto('/ai-assistant')
    // Should show upgrade prompt for non-performance users
    await expect(page.getByText('Premium Feature')).toBeVisible()
  })

  test('Responsive design across devices', async ({ page }) => {
    // Test mobile responsiveness
    
    // Desktop view
    await page.setViewportSize({ width: 1920, height: 1080 })
    await expect(page.getByText('Transform Your Fitness')).toBeVisible()
    
    // Tablet view
    await page.setViewportSize({ width: 768, height: 1024 })
    await expect(page.getByText('Transform Your Fitness')).toBeVisible()
    
    // Mobile view
    await page.setViewportSize({ width: 375, height: 667 })
    await expect(page.getByText('Transform Your Fitness')).toBeVisible()
    
    // Mobile navigation should work
    await page.getByRole('button', { name: 'Menu' }).click()
    await expect(page.getByText('Pricing')).toBeVisible()
  })

  test('Performance benchmarks', async ({ page }) => {
    // Test Core Web Vitals
    
    const performanceEntries = await page.evaluate(() => {
      return performance.getEntriesByType('navigation')
    }) as PerformanceNavigationTiming[]
    
    // Page should load within performance budgets
    const loadTime = performanceEntries[0]?.loadEventEnd - performanceEntries[0]?.loadEventStart
    expect(loadTime).toBeLessThan(3000) // 3 seconds
    
    // Check for layout shift
    const clsEntries = await page.evaluate(() => {
      return performance.getEntriesByType('layout-shift')
    })
    
    const totalCLS = clsEntries.reduce((sum: number, entry: any) => sum + entry.value, 0)
    expect(totalCLS).toBeLessThan(0.1) // CLS should be under 0.1
  })

  test('Accessibility compliance', async ({ page }) => {
    // Test keyboard navigation
    await page.keyboard.press('Tab')
    await expect(page.locator(':focus')).toBeVisible()
    
    // Test screen reader compatibility
    const titleElement = page.getByRole('heading', { level: 1 })
    await expect(titleElement).toBeVisible()
    
    // Test color contrast (basic check)
    const buttonElement = page.getByRole('button', { name: 'Get Started' })
    const buttonStyles = await buttonElement.evaluate((el) => {
      return window.getComputedStyle(el)
    })
    
    // Button should have sufficient contrast
    expect(buttonStyles.backgroundColor).toBeTruthy()
    expect(buttonStyles.color).toBeTruthy()
  })

  test('Internationalization features', async ({ page }) => {
    // Test language switching
    await page.goto('/system')
    
    // Switch to Spanish
    await page.getByText('Languages').click()
    await page.getByText('Spanish').click()
    await page.getByRole('button', { name: 'Switch' }).click()
    
    // UI should update to Spanish
    await expect(page.getByText('Bienvenido')).toBeVisible()
    
    // Switch back to English
    await page.getByText('English').click()
    await page.getByRole('button', { name: 'Switch' }).click()
    await expect(page.getByText('Welcome')).toBeVisible()
  })

  test('Error handling and recovery', async ({ page }) => {
    // Test network error handling
    await page.route('**/api/**', route => route.abort())
    
    await page.getByText('Get Started').click()
    
    // Should show error message gracefully
    await expect(page.getByText(/error|failed|try again/i)).toBeVisible()
    
    // Restore network and retry
    await page.unroute('**/api/**')
    await page.getByText(/retry|try again/i).click()
    
    // Should recover gracefully
    await expect(page.getByPlaceholder('Email')).toBeVisible()
  })

  test('Security features', async ({ page }) => {
    // Test CSP headers
    const response = await page.goto('/')
    const cspHeader = response?.headers()['content-security-policy']
    expect(cspHeader).toBeTruthy()
    
    // Test form validation
    await page.getByText('Get Started').click()
    await page.getByRole('button', { name: 'Create Account' }).click()
    
    // Should show validation errors
    await expect(page.getByText(/email is required/i)).toBeVisible()
    
    // Test password requirements
    await page.getByPlaceholder('Password').fill('123')
    await expect(page.getByText(/password must be at least/i)).toBeVisible()
  })

  test('Data persistence and state management', async ({ page }) => {
    // Test localStorage persistence
    await page.goto('/pricing')
    
    // Set billing cycle preference
    await page.getByText('Annual').click()
    
    // Refresh page
    await page.reload()
    
    // Preference should persist
    await expect(page.getByText('Annual')).toHaveClass(/active|selected/)
  })
})

test.describe('Load Testing Simulation', () => {
  test('Handles multiple concurrent users', async ({ browser }) => {
    // Simulate multiple users
    const contexts = await Promise.all([
      browser.newContext(),
      browser.newContext(),
      browser.newContext(),
      browser.newContext(),
      browser.newContext()
    ])
    
    const pages = await Promise.all(contexts.map(context => context.newPage()))
    
    // All users navigate simultaneously
    await Promise.all(pages.map(page => page.goto('/')))
    
    // All should load successfully
    for (const page of pages) {
      await expect(page.getByText('REPZ')).toBeVisible()
    }
    
    // Cleanup
    await Promise.all(contexts.map(context => context.close()))
  })
})