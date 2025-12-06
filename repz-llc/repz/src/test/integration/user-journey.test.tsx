import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent, waitFor } from '@/test/utils/test-utils'
import { BrowserRouter } from 'react-router-dom'
import App from '@/App'

// Mock external dependencies
vi.mock('@/integrations/supabase/client', () => ({
  supabase: {
    auth: {
      signInWithPassword: vi.fn().mockResolvedValue({
        data: { user: { id: 'test-user', email: 'test@example.com' } },
        error: null
      }),
      signUp: vi.fn().mockResolvedValue({
        data: { user: { id: 'test-user', email: 'test@example.com' } },
        error: null
      }),
      getSession: vi.fn().mockResolvedValue({
        data: { session: null },
        error: null
      }),
      onAuthStateChange: vi.fn().mockReturnValue({
        data: { subscription: { unsubscribe: vi.fn() } }
      })
    },
    from: vi.fn().mockReturnThis(),
    select: vi.fn().mockReturnThis(),
    insert: vi.fn().mockReturnThis(),
    update: vi.fn().mockReturnThis(),
    eq: vi.fn().mockReturnThis(),
    single: vi.fn().mockResolvedValue({ data: null, error: null })
  }
}))

vi.mock('@stripe/react-stripe-js', () => ({
  useStripe: () => ({
    createPaymentMethod: vi.fn().mockResolvedValue({
      paymentMethod: { id: 'pm_test_123' }
    })
  }),
  useElements: () => ({
    getElement: vi.fn().mockReturnValue({})
  }),
  Elements: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  loadStripe: vi.fn()
}))

describe('Complete User Journey Integration Tests', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    // Reset window location
    delete (window as any).location
    window.location = { href: 'http://localhost:3000/', pathname: '/' } as any
  })

  it('completes full signup to dashboard journey', async () => {
    render(<App />)
    
    // 1. User lands on homepage
    expect(screen.getByText(/transform your fitness/i)).toBeInTheDocument()
    
    // 2. Navigate to signup
    const getStartedButton = screen.getByText(/get started/i)
    fireEvent.click(getStartedButton)
    
    await waitFor(() => {
      expect(screen.getByText(/create your account/i)).toBeInTheDocument()
    })
    
    // 3. Fill out signup form
    const emailInput = screen.getByPlaceholderText(/email/i)
    const passwordInput = screen.getByPlaceholderText(/password/i)
    const signupButton = screen.getByRole('button', { name: /create account/i })
    
    fireEvent.change(emailInput, { target: { value: 'test@example.com' } })
    fireEvent.change(passwordInput, { target: { value: 'password123' } })
    fireEvent.click(signupButton)
    
    // 4. Should redirect to dashboard after successful signup
    await waitFor(() => {
      expect(screen.getByText(/welcome/i)).toBeInTheDocument()
    }, { timeout: 5000 })
  })

  it('completes tier selection and payment flow', async () => {
    render(<App />)
    
    // Navigate to pricing page
    const pricingLink = screen.getByText(/pricing/i)
    fireEvent.click(pricingLink)
    
    await waitFor(() => {
      expect(screen.getByText(/choose your plan/i)).toBeInTheDocument()
    })
    
    // Select adaptive tier
    const adaptiveCard = screen.getByText(/adaptive/i).closest('div')
    const upgradeButton = adaptiveCard?.querySelector('button')
    
    if (upgradeButton) {
      fireEvent.click(upgradeButton)
    }
    
    // Should open payment modal
    await waitFor(() => {
      expect(screen.getByText(/payment details/i)).toBeInTheDocument()
    })
  })

  it('handles authentication errors gracefully', async () => {
    // Mock authentication failure
    const mockSupabase = await vi.importActual('@/integrations/supabase/client') as any
    mockSupabase.supabase.auth.signInWithPassword
      .mockResolvedValueOnce({
        data: { user: null },
        error: { message: 'Invalid credentials' }
      })

    render(<App />)
    
    // Navigate to login
    const loginLink = screen.getByText(/sign in/i)
    fireEvent.click(loginLink)
    
    await waitFor(() => {
      const emailInput = screen.getByPlaceholderText(/email/i)
      const passwordInput = screen.getByPlaceholderText(/password/i)
      const loginButton = screen.getByRole('button', { name: /sign in/i })
      
      fireEvent.change(emailInput, { target: { value: 'test@example.com' } })
      fireEvent.change(passwordInput, { target: { value: 'wrongpassword' } })
      fireEvent.click(loginButton)
    })
    
    // Should show error message
    await waitFor(() => {
      expect(screen.getByText(/invalid credentials/i)).toBeInTheDocument()
    })
  })

  it('maintains user session across page refreshes', async () => {
    // Mock existing session
    const mockSupabase = await vi.importActual('@/integrations/supabase/client') as any
    mockSupabase.supabase.auth.getSession
      .mockResolvedValueOnce({
        data: {
          session: {
            user: { id: 'test-user', email: 'test@example.com' },
            access_token: 'mock-token'
          }
        },
        error: null
      })

    render(<App />)
    
    // Should automatically redirect to dashboard for authenticated users
    await waitFor(() => {
      expect(screen.getByText(/dashboard/i)).toBeInTheDocument()
    })
  })

  it('protects routes that require authentication', async () => {
    render(<App />)
    
    // Try to access protected route directly
    window.history.pushState({}, '', '/dashboard')
    
    await waitFor(() => {
      // Should redirect to login page
      expect(screen.getByText(/sign in/i)).toBeInTheDocument()
    })
  })

  it('shows tier-specific features correctly', async () => {
    // Mock user with performance tier
    const mockSupabase = await vi.importActual('@/integrations/supabase/client') as any
    mockSupabase.supabase.auth.getSession
      .mockResolvedValueOnce({
        data: {
          session: {
            user: {
              id: 'test-user',
              email: 'test@example.com',
              user_metadata: { tier: 'performance' }
            },
            access_token: 'mock-token'
          }
        },
        error: null
      })

    render(<App />)
    
    // Navigate to dashboard
    window.history.pushState({}, '', '/dashboard')
    
    await waitFor(() => {
      // Should show performance tier features
      expect(screen.getByText(/ai assistant/i)).toBeInTheDocument()
      expect(screen.getByText(/performance/i)).toBeInTheDocument()
    })
  })

  it('handles offline functionality', async () => {
    // Mock offline state
    Object.defineProperty(navigator, 'onLine', {
      writable: true,
      value: false
    })

    render(<App />)
    
    // Should still render basic functionality
    expect(screen.getByText(/repzcoach/i)).toBeInTheDocument()
    
    // Restore online state
    Object.defineProperty(navigator, 'onLine', {
      writable: true,
      value: true
    })
  })
})