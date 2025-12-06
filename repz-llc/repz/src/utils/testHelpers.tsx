import { render, RenderOptions } from '@testing-library/react'
import { ReactElement } from 'react'
import { BrowserRouter } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { vi } from 'vitest'

// Mock data generators
export const generateMockUser = (overrides = {}) => ({
  id: 'test-user-id',
  email: 'test@example.com',
  name: 'Test User',
  created_at: new Date().toISOString(),
  ...overrides
})

export const generateMockWorkout = (overrides = {}) => ({
  id: 'test-workout-id',
  user_id: 'test-user-id',
  name: 'Test Workout',
  exercises: [
    {
      name: 'Push-ups',
      sets: 3,
      reps: 10,
      weight: null
    }
  ],
  completed_at: new Date().toISOString(),
  ...overrides
})

export const generateMockProgress = (overrides = {}) => ({
  id: 'test-progress-id',
  user_id: 'test-user-id',
  weight: 180,
  body_fat: 20,
  muscle_mass: 140,
  recorded_at: new Date().toISOString(),
  ...overrides
})

// Test wrapper component
interface TestWrapperProps {
  children: React.ReactNode
}

const TestWrapper = ({ children }: TestWrapperProps) => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
      },
    },
  })

  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        {children}
      </BrowserRouter>
    </QueryClientProvider>
  )
}

// Custom render function
const customRender = (
  ui: ReactElement,
  options?: Omit<RenderOptions, 'wrapper'>
) => render(ui, { wrapper: TestWrapper, ...options })

// Mock Supabase client
export const mockSupabaseClient = {
  auth: {
    getUser: vi.fn().mockResolvedValue({
      data: { user: generateMockUser() },
      error: null
    }),
    signInWithPassword: vi.fn(),
    signUp: vi.fn(),
    signOut: vi.fn(),
    onAuthStateChange: vi.fn().mockReturnValue({
      data: { subscription: { unsubscribe: vi.fn() } }
    })
  },
  from: vi.fn().mockReturnThis(),
  select: vi.fn().mockReturnThis(),
  insert: vi.fn().mockReturnThis(),
  update: vi.fn().mockReturnThis(),
  delete: vi.fn().mockReturnThis(),
  eq: vi.fn().mockReturnThis(),
  order: vi.fn().mockReturnThis(),
  limit: vi.fn().mockReturnThis(),
  single: vi.fn(),
  data: null,
  error: null
}

// Mock localStorage
export const mockLocalStorage = (() => {
  let store: { [key: string]: string } = {}

  return {
    getItem: (key: string) => store[key] || null,
    setItem: (key: string, value: string) => {
      store[key] = value.toString()
    },
    removeItem: (key: string) => {
      delete store[key]
    },
    clear: () => {
      store = {}
    }
  }
})()

// Mock console methods for testing
export const mockConsole = {
  log: vi.fn(),
  warn: vi.fn(),
  error: vi.fn(),
  info: vi.fn()
}

// Utility functions for testing
export const waitForElement = (element: HTMLElement) => {
  return new Promise((resolve) => {
    if (element) {
      resolve(element)
      return
    }

    const observer = new MutationObserver(() => {
      if (element) {
        resolve(element)
        observer.disconnect()
      }
    })

    observer.observe(document.body, {
      childList: true,
      subtree: true
    })
  })
}

export const createMockIntersectionObserver = () => {
  const mockIntersectionObserver = vi.fn()
  mockIntersectionObserver.mockReturnValue({
    observe: () => null,
    unobserve: () => null,
    disconnect: () => null
  })
  window.IntersectionObserver = mockIntersectionObserver
}

// Re-export everything
export * from '@testing-library/react'
export { customRender as render }
