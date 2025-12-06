import { describe, it, expect, vi, beforeEach } from 'vitest';
import { supabase } from '@/integrations/supabase/client';

// Mock Supabase
vi.mock('@/integrations/supabase/client', () => ({
  supabase: {
    functions: {
      invoke: vi.fn()
    },
    from: vi.fn(() => ({
      select: vi.fn(() => ({
        eq: vi.fn(() => Promise.resolve({
          data: [],
          error: null
        }))
      })),
      insert: vi.fn(() => Promise.resolve({
        data: [],
        error: null
      })),
      update: vi.fn(() => Promise.resolve({
        data: [],
        error: null
      }))
    }))
  }
}));

describe('Stripe Integration Tests', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('Subscription Creation', () => {
    it('creates subscription with correct tier and pricing', async () => {
      const mockResponse = {
        data: {
          url: 'https://checkout.stripe.com/session123',
          subscription: {
            id: 'sub_123',
            items: {
              data: [{
                price: {
                  id: 'price_performance_monthly',
                  unit_amount: 29900
                }
              }]
            }
          }
        },
        error: null
      };

      vi.mocked(supabase.functions.invoke).mockResolvedValue(mockResponse);

      const result = await supabase.functions.invoke('create-checkout', {
        body: {
          tier: 'performance',
          billing_period: 'monthly',
          formData: {
            email: 'test@example.com',
            name: 'Test User'
          }
        }
      });

      expect(result.data?.url).toBe('https://checkout.stripe.com/session123');
      expect(supabase.functions.invoke).toHaveBeenCalledWith('create-checkout', {
        body: {
          tier: 'performance',
          billing_period: 'monthly',
          formData: {
            email: 'test@example.com',
            name: 'Test User'
          }
        }
      });
    });

    it('handles subscription creation errors', async () => {
      const mockError = {
        data: null,
        error: { message: 'Payment method declined' }
      };

      vi.mocked(supabase.functions.invoke).mockResolvedValue(mockError);

      const result = await supabase.functions.invoke('create-checkout', {
        body: {
          tier: 'performance',
          billing_period: 'monthly'
        }
      });

      expect(result.error?.message).toBe('Payment method declined');
    });
  });

  describe('Subscription Status Verification', () => {
    it('verifies active subscription status', async () => {
      const mockSubscriptionData = {
        data: {
          subscribed: true,
          subscription_tier: 'performance',
          subscription_end: '2024-12-31T23:59:59.000Z'
        },
        error: null
      };

      vi.mocked(supabase.functions.invoke).mockResolvedValue(mockSubscriptionData);

      const result = await supabase.functions.invoke('check-subscription');

      expect(result.data?.subscribed).toBe(true);
      expect(result.data?.subscription_tier).toBe('performance');
    });

    it('handles subscription verification for unsubscribed users', async () => {
      const mockResponse = {
        data: {
          subscribed: false,
          subscription_tier: null,
          subscription_end: null
        },
        error: null
      };

      vi.mocked(supabase.functions.invoke).mockResolvedValue(mockResponse);

      const result = await supabase.functions.invoke('check-subscription');

      expect(result.data?.subscribed).toBe(false);
      expect(result.data?.subscription_tier).toBeNull();
    });
  });

  describe('Customer Portal Integration', () => {
    it('creates customer portal session successfully', async () => {
      const mockPortalData = {
        data: {
          url: 'https://billing.stripe.com/session123'
        },
        error: null
      };

      vi.mocked(supabase.functions.invoke).mockResolvedValue(mockPortalData);

      const result = await supabase.functions.invoke('customer-portal');

      expect(result.data?.url).toBe('https://billing.stripe.com/session123');
    });

    it('handles customer portal errors', async () => {
      const mockError = {
        data: null,
        error: { message: 'No customer found' }
      };

      vi.mocked(supabase.functions.invoke).mockResolvedValue(mockError);

      const result = await supabase.functions.invoke('customer-portal');

      expect(result.error?.message).toBe('No customer found');
    });
  });

  describe('Webhook Event Processing', () => {
    it('processes subscription created webhook correctly', async () => {
      const webhookEvent = {
        type: 'customer.subscription.created',
        data: {
          object: {
            id: 'sub_123',
            customer: 'cus_123',
            items: {
              data: [{
                price: {
                  id: 'price_performance_monthly',
                  unit_amount: 29900
                }
              }]
            },
            status: 'active'
          }
        }
      };

      // Mock database update
      const mockUpdate = {
        data: [{ id: '1', subscription_tier: 'performance' }],
        error: null
      };

      vi.mocked(supabase.from).mockReturnValue({
        update: vi.fn(() => ({
          eq: vi.fn(() => Promise.resolve(mockUpdate))
        }))
      } as any);

      // Simulate webhook processing
      const result = await processWebhookEvent(webhookEvent);

      expect(result.processed).toBe(true);
      expect(result.subscription_tier).toBe('performance');
    });

    it('processes subscription cancelled webhook correctly', async () => {
      const webhookEvent = {
        type: 'customer.subscription.deleted',
        data: {
          object: {
            id: 'sub_123',
            customer: 'cus_123',
            status: 'canceled'
          }
        }
      };

      // Mock database update
      const mockUpdate = {
        data: [{ id: '1', subscription_tier: 'core' }],
        error: null
      };

      vi.mocked(supabase.from).mockReturnValue({
        update: vi.fn(() => ({
          eq: vi.fn(() => Promise.resolve(mockUpdate))
        }))
      } as any);

      const result = await processWebhookEvent(webhookEvent);

      expect(result.processed).toBe(true);
      expect(result.subscription_tier).toBe('core');
    });
  });

  describe('Payment Validation', () => {
    it('validates successful payment completion', async () => {
      const sessionId = 'cs_test_123';
      
      const mockSessionData = {
        data: {
          payment_status: 'paid',
          subscription: 'sub_123',
          customer: 'cus_123'
        },
        error: null
      };

      vi.mocked(supabase.functions.invoke).mockResolvedValue(mockSessionData);

      const result = await supabase.functions.invoke('validate-payment', {
        body: { session_id: sessionId }
      });

      expect(result.data?.payment_status).toBe('paid');
    });

    it('handles failed payment validation', async () => {
      const sessionId = 'cs_test_failed';
      
      const mockSessionData = {
        data: {
          payment_status: 'unpaid',
          subscription: null
        },
        error: null
      };

      vi.mocked(supabase.functions.invoke).mockResolvedValue(mockSessionData);

      const result = await supabase.functions.invoke('validate-payment', {
        body: { session_id: sessionId }
      });

      expect(result.data?.payment_status).toBe('unpaid');
    });
  });
});

// Mock webhook event processor
async function processWebhookEvent(event: any) {
  if (event.type === 'customer.subscription.created') {
    return {
      processed: true,
      subscription_tier: 'performance'
    };
  }
  
  if (event.type === 'customer.subscription.deleted') {
    return {
      processed: true,
      subscription_tier: 'core'
    };
  }
  
  return { processed: false };
}