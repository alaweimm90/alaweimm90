import React, { ReactElement } from 'react'
import { render, RenderOptions } from '@testing-library/react'
import { BrowserRouter } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { SecurityProvider } from '@/security/SecurityProvider'
import { AccessibilityProvider } from '@/accessibility/AccessibilityProvider'
import { TooltipProvider } from '@/components/ui/tooltip'

// Mock AuthProvider for testing
const MockAuthProvider = ({ children }: { children: React.ReactNode }) => {
  return <>{children}</>
}

// Custom render function with all providers
const AllTheProviders = ({ children }: { children: React.ReactNode }) => {
  const testQueryClient = new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
        staleTime: Infinity,
      },
    },
  })

  return (
    <QueryClientProvider client={testQueryClient}>
      <SecurityProvider>
        <AccessibilityProvider>
          <MockAuthProvider>
            <BrowserRouter>
              <TooltipProvider>
                {children}
              </TooltipProvider>
            </BrowserRouter>
          </MockAuthProvider>
        </AccessibilityProvider>
      </SecurityProvider>
    </QueryClientProvider>
  )
}

const customRender = (
  ui: ReactElement,
  options?: Omit<RenderOptions, 'wrapper'>,
) => render(ui, { wrapper: AllTheProviders, ...options })

export * from '@testing-library/react'
export { customRender as render }

// Test data factories
export const createMockUser = (overrides = {}) => ({
  id: 'test-user-id',
  email: 'test@example.com',
  user_metadata: { tier: 'core' },
  created_at: new Date().toISOString(),
  ...overrides
})

export const createMockTierData = (tier: string) => ({
  core: {
    name: 'Core',
    price: 29,
    features: ['Basic workouts', 'Progress tracking']
  },
  adaptive: {
    name: 'Adaptive',
    price: 49,
    features: ['AI coaching', 'Custom plans', 'Weekly check-ins']
  },
  performance: {
    name: 'Performance',
    price: 89,
    features: ['Advanced analytics', 'Form analysis', 'Competition prep']
  },
  longevity: {
    name: 'Longevity',
    price: 149,
    features: ['Longevity protocols', 'In-person training', 'Concierge service']
  }
}[tier] || {})

// Helper functions for testing
export const waitForLoadingToFinish = () => {
  return new Promise(resolve => setTimeout(resolve, 0))
}

export const mockLocalStorage = async () => {
  const { vi } = await import('vitest')
  const localStorageMock = {
    getItem: vi.fn(),
    setItem: vi.fn(),
    removeItem: vi.fn(),
    clear: vi.fn(),
  }

  Object.defineProperty(window, 'localStorage', {
    value: localStorageMock
  })

  return localStorageMock
}