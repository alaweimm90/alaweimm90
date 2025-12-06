import '@testing-library/jest-dom'
import { expect, afterEach, beforeAll, afterAll, vi } from 'vitest'
import { cleanup } from '@testing-library/react'
import { setupServer } from 'msw/node'
import { http, HttpResponse } from 'msw'

// Clean up after each test
afterEach(() => {
  cleanup()
})

// Mock Supabase
const mockSupabaseHandlers = [
  http.post('*/auth/v1/token*', () => {
    return HttpResponse.json({
      access_token: 'mock-token',
      user: {
        id: 'test-user-id',
        email: 'test@example.com',
        user_metadata: { tier: 'core' }
      }
    })
  }),
  http.get('*/rest/v1/*', () => {
    return HttpResponse.json([])
  }),
  http.post('*/rest/v1/*', () => {
    return HttpResponse.json({})
  })
]

// Mock Stripe
const mockStripeHandlers = [
  http.post('*/v1/payment_intents', () => {
    return HttpResponse.json({
      id: 'pi_test_123',
      client_secret: 'pi_test_123_secret',
      status: 'requires_payment_method'
    })
  })
]

// Set up MSW server
const server = setupServer(...mockSupabaseHandlers, ...mockStripeHandlers)

beforeAll(() => {
  server.listen({ onUnhandledRequest: 'warn' })
})

afterEach(() => {
  server.resetHandlers()
})

afterAll(() => {
  server.close()
})

// Mock environment variables
Object.defineProperty(window, 'location', {
  value: {
    href: 'https://test.example.com',
    origin: 'https://test.example.com'
  }
})

// Mock ResizeObserver
global.ResizeObserver = class ResizeObserver {
  observe() {}
  unobserve() {}
  disconnect() {}
}

// Mock IntersectionObserver
global.IntersectionObserver = vi.fn().mockImplementation((callback, options) => ({
  observe: vi.fn(),
  unobserve: vi.fn(),
  disconnect: vi.fn(),
  root: null,
  rootMargin: '0px',
  thresholds: [0],
  takeRecords: vi.fn().mockReturnValue([])
})) as unknown as typeof IntersectionObserver

// Mock matchMedia
Object.defineProperty(window, 'matchMedia', {
  writable: true,
  value: vi.fn().mockImplementation((query: string) => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: vi.fn(), // Deprecated
    removeListener: vi.fn(), // Deprecated
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    dispatchEvent: vi.fn(),
  })),
})

export { server }
vi.mock('@/integrations/supabase/client', () => ({
  supabase: {
    auth: {
      getSession: vi.fn().mockResolvedValue({ data: { session: { user: { id: 'test-user-id', email: 'test@example.com' } } } }),
      onAuthStateChange: vi.fn().mockReturnValue({ data: { subscription: { unsubscribe: vi.fn() } } }),
      signInWithPassword: vi.fn().mockResolvedValue({ error: null }),
      signUp: vi.fn().mockResolvedValue({ error: null }),
      signOut: vi.fn().mockResolvedValue({ error: null }),
    },
    from: vi.fn().mockReturnValue({
      select: vi.fn().mockReturnValue({
        eq: vi.fn().mockReturnValue({
          single: vi.fn().mockResolvedValue({ data: null })
        })
      })
    }),
    functions: {
      invoke: vi.fn().mockResolvedValue({ data: { url: 'https://example.com/checkout' }, error: null })
    }
  }
}))
