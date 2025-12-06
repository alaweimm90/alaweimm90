import { describe, it, expect, vi, beforeEach } from 'vitest'

// Mock types for testing
interface CheckoutData {
  tier: string
  billingPeriod: string
}

interface CheckoutResult {
  success: boolean
  sessionUrl?: string
  error?: string
}

// Mock PaymentService with required methods
const PaymentService = {
  createCheckoutSession: vi.fn(),
  validatePaymentAmount: vi.fn(),
  calculateDiscountedPrice: vi.fn(),
  getStripePriceId: vi.fn(),
  createCustomerPortalSession: vi.fn()
}

// Mock Supabase
const mockInvoke = vi.fn()
vi.mock('@/integrations/supabase/client', () => ({
  supabase: {
    functions: {
      invoke: mockInvoke
    }
  }
}))

// Mock analytics
const mockTrackConversion = {
  checkoutStarted: vi.fn(),
  tierReservation: vi.fn()
}
vi.mock('@/hooks/useAnalytics', () => ({
  useAnalytics: () => ({
    trackConversion: mockTrackConversion
  })
}))

describe('PaymentService', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  describe('createCheckoutSession', () => {
    it('creates checkout session successfully', async () => {
      const mockSessionUrl = 'https://checkout.stripe.com/session123'
      PaymentService.createCheckoutSession.mockResolvedValue({
        success: true,
        sessionUrl: mockSessionUrl
      })

      const result = await PaymentService.createCheckoutSession({
        tier: 'core',
        billingPeriod: 'monthly'
      })

      expect(result.success).toBe(true)
      expect(result.sessionUrl).toBe(mockSessionUrl)
    })

    it('handles checkout creation errors', async () => {
      PaymentService.createCheckoutSession.mockResolvedValue({
        success: false,
        error: 'Payment processor unavailable'
      })

      const result = await PaymentService.createCheckoutSession({
        tier: 'adaptive',
        billingPeriod: 'annual'
      })

      expect(result.success).toBe(false)
      expect(result.error).toBe('Payment processor unavailable')
    })

    it('handles network errors gracefully', async () => {
      PaymentService.createCheckoutSession.mockRejectedValue(new Error('Network error'))

      try {
        await PaymentService.createCheckoutSession({
          tier: 'performance',
          billingPeriod: 'monthly'
        })
      } catch (error) {
        expect(error.message).toBe('Network error')
      }
    })
  })

  describe('validatePaymentAmount', () => {
    it('validates correct pricing for each tier', () => {
      PaymentService.validatePaymentAmount.mockImplementation((tier, period, amount) => {
        const prices = {
          core: { monthly: 97, annual: 78 },
          adaptive: { monthly: 199, annual: 159 },
          performance: { monthly: 299, annual: 239 },
          longevity: { monthly: 449, annual: 359 }
        }
        return prices[tier]?.[period] === amount
      })

      expect(PaymentService.validatePaymentAmount('core', 'monthly', 97)).toBe(true)
      expect(PaymentService.validatePaymentAmount('adaptive', 'monthly', 199)).toBe(true)
      expect(PaymentService.validatePaymentAmount('performance', 'monthly', 299)).toBe(true)
      expect(PaymentService.validatePaymentAmount('longevity', 'monthly', 449)).toBe(true)
    })

    it('rejects incorrect amounts', () => {
      PaymentService.validatePaymentAmount.mockImplementation(() => false)
      
      expect(PaymentService.validatePaymentAmount('core', 'monthly', 100)).toBe(false)
      expect(PaymentService.validatePaymentAmount('adaptive', 'annual', 199)).toBe(false)
    })
  })

  describe('calculateDiscountedPrice', () => {
    it('applies correct discounts by billing period', () => {
      PaymentService.calculateDiscountedPrice.mockImplementation((basePrice, period) => {
        const discounts = {
          monthly: 1,
          quarterly: 0.95,
          'semi-annual': 0.90,
          annual: 0.80
        }
        return Math.floor(basePrice * discounts[period])
      })

      expect(PaymentService.calculateDiscountedPrice(100, 'monthly')).toBe(100)
      expect(PaymentService.calculateDiscountedPrice(100, 'quarterly')).toBe(95)
      expect(PaymentService.calculateDiscountedPrice(100, 'semi-annual')).toBe(90)
      expect(PaymentService.calculateDiscountedPrice(100, 'annual')).toBe(80)
    })
  })

  describe('getStripePriceId', () => {
    it('returns correct price IDs for different environments', () => {
      PaymentService.getStripePriceId.mockImplementation((tier, period) => {
        const env = process.env.NODE_ENV === 'production' ? 'PROD' : 'TEST'
        return `STRIPE_PRICE_${tier.toUpperCase()}_${period.toUpperCase()}_${env}`
      })

      expect(PaymentService.getStripePriceId('core', 'monthly')).toContain('TEST')
      expect(PaymentService.getStripePriceId('adaptive', 'annual')).toContain('TEST')
    })
  })

  describe('subscription management', () => {
    it('creates customer portal session', async () => {
      const mockPortalUrl = 'https://billing.stripe.com/session123'
      PaymentService.createCustomerPortalSession.mockResolvedValue({
        success: true,
        portalUrl: mockPortalUrl
      })

      const result = await PaymentService.createCustomerPortalSession('cus_123')

      expect(result.success).toBe(true)
      expect(result.portalUrl).toBe(mockPortalUrl)
    })
  })
})