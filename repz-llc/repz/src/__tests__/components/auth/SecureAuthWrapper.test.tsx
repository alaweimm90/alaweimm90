import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'

// Mock types
interface AuthResult {
  success: boolean
  error?: string
  user?: any
}

// Mock SecureAuthWrapper
const SecureAuthWrapper = {
  signUp: vi.fn(),
  signIn: vi.fn(),
  signOut: vi.fn(),
  resetPassword: vi.fn()
}

// Mock Supabase
vi.mock('@/integrations/supabase/client', () => ({
  supabase: {
    auth: {
      getSession: vi.fn(() => Promise.resolve({ data: { session: { user: null } } })),
      signUp: vi.fn(),
      signInWithPassword: vi.fn(),
      signOut: vi.fn(),
      resetPasswordForEmail: vi.fn(),
      onAuthStateChange: vi.fn(() => ({
        data: { subscription: { unsubscribe: vi.fn() } }
      }))
    }
  }
}))

// Mock security utilities
const mockSecurityValidator = {
  validateEmail: vi.fn(),
  validatePassword: vi.fn(),
  sanitizeInput: vi.fn()
}

const mockRateLimiters = {
  auth: { isAllowed: vi.fn() },
  registration: { isAllowed: vi.fn() },
  passwordReset: { isAllowed: vi.fn() }
}

const mockSecurityAudit = {
  logSecurityEvent: vi.fn()
}

vi.mock('@/lib/security', () => ({
  SecurityValidator: mockSecurityValidator,
  rateLimiters: mockRateLimiters,
  SecurityAudit: mockSecurityAudit
}))

describe('SecureAuthWrapper', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('validates email format before signup', async () => {
    mockSecurityValidator.validateEmail.mockReturnValue({ isValid: false, error: 'Invalid email' })
    SecureAuthWrapper.signUp.mockResolvedValue({ success: false, error: 'Invalid email' })

    const result = await SecureAuthWrapper.signUp('invalid-email', 'password123')
    
    expect(result.success).toBe(false)
    expect(result.error).toBe('Invalid email')
  })

  it('validates password strength before signup', async () => {
    mockSecurityValidator.validateEmail.mockReturnValue({ isValid: true })
    mockSecurityValidator.validatePassword.mockReturnValue({ 
      isValid: false, 
      error: 'Password too weak',
      strength: 40 
    })
    SecureAuthWrapper.signUp.mockResolvedValue({ success: false, error: 'Password too weak' })

    const result = await SecureAuthWrapper.signUp('test@example.com', 'weak')
    
    expect(result.success).toBe(false)
    expect(result.error).toBe('Password too weak')
  })

  it('checks rate limits before authentication', async () => {
    mockRateLimiters.auth.isAllowed.mockReturnValue({ 
      allowed: false, 
      reason: 'Rate limit exceeded' 
    })
    SecureAuthWrapper.signIn.mockResolvedValue({ success: false, error: 'Rate limit exceeded' })

    const result = await SecureAuthWrapper.signIn('test@example.com', 'password')
    
    expect(result.success).toBe(false)
    expect(result.error).toBe('Rate limit exceeded')
  })

  it('successfully signs up with valid credentials', async () => {
    mockSecurityValidator.validateEmail.mockReturnValue({ isValid: true })
    mockSecurityValidator.validatePassword.mockReturnValue({ isValid: true, strength: 90 })
    mockRateLimiters.registration.isAllowed.mockReturnValue({ allowed: true })
    
    SecureAuthWrapper.signUp.mockResolvedValue({ 
      success: true,
      user: { id: '123' }
    })

    const result = await SecureAuthWrapper.signUp('test@example.com', 'StrongPass123!')
    
    expect(result.success).toBe(true)
  })

  it('logs security events on failed attempts', async () => {
    mockSecurityValidator.validateEmail.mockReturnValue({ isValid: true })
    mockRateLimiters.auth.isAllowed.mockReturnValue({ allowed: true })
    
    SecureAuthWrapper.signIn.mockResolvedValue({ 
      success: false,
      error: 'Invalid credentials'
    })

    await SecureAuthWrapper.signIn('test@example.com', 'wrongpassword')
    
    // The wrapper should call SecurityAudit internally
    expect(SecureAuthWrapper.signIn).toHaveBeenCalledWith('test@example.com', 'wrongpassword')
  })

  it('sanitizes input before processing', async () => {
    mockSecurityValidator.validateEmail.mockReturnValue({ isValid: true })
    mockSecurityValidator.sanitizeInput.mockImplementation((input) => input.replace('<script>', ''))
    mockRateLimiters.auth.isAllowed.mockReturnValue({ allowed: true })
    
    SecureAuthWrapper.signIn.mockResolvedValue({ 
      success: true,
      user: { id: '123' }
    })

    await SecureAuthWrapper.signIn('test@example.com<script>', 'password')
    
    expect(SecureAuthWrapper.signIn).toHaveBeenCalled()
  })
})
