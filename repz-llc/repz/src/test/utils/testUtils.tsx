import React from 'react'
import { render, RenderOptions } from '@testing-library/react'
import { BrowserRouter } from 'react-router-dom'
import { vi } from 'vitest'
import { AuthProvider } from '@/contexts/AuthContext'
import { mockUsers, mockClientProfiles } from '../fixtures/mockData'

// Custom render function that includes providers
const AllTheProviders = ({ children }: { children: React.ReactNode }) => {
  return (
    <BrowserRouter>
      <AuthProvider>
        {children}
      </AuthProvider>
    </BrowserRouter>
  )
}

const customRender = (
  ui: React.ReactElement,
  options?: Omit<RenderOptions, 'wrapper'>
) => render(ui, { wrapper: AllTheProviders, ...options })

// Re-export everything
export * from '@testing-library/react'
export { customRender as render }

// Test utilities
export const waitForLoadingToFinish = () => 
  new Promise(resolve => setTimeout(resolve, 0))

export const mockUserSession = (userType: 'core' | 'adaptive' | 'performance' | 'longevity' = 'core') => ({
  user: mockUsers[`${userType}User`],
  session: {
    access_token: 'mock-token',
    token_type: 'bearer',
    expires_in: 3600,
    user: mockUsers[`${userType}User`]
  }
})

export const mockTierAccess = (tier: string) => ({
  hasAccess: (feature: string) => {
    const tierFeatures = mockClientProfiles[tier as keyof typeof mockClientProfiles]?.tier_features
    return tierFeatures?.[feature as keyof typeof tierFeatures] || false
  },
  userTier: tier,
  upgradeRequired: false
})

export const createMockAnalytics = () => ({
  trackConversion: {
    tierReservation: vi.fn(),
    checkoutStarted: vi.fn(),
    subscriptionUpgrade: vi.fn()
  },
  trackFunnel: {
    pricingView: vi.fn(),
    tierSelected: vi.fn(),
    reservationFormStart: vi.fn(),
    subscriptionModalView: vi.fn()
  },
  trackCustom: vi.fn()
})