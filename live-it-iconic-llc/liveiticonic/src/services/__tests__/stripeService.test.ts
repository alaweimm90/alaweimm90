/**
 * Stripe Service Tests
 *
 * Comprehensive test suite for Stripe payment integration
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { stripeService } from '../stripeService';

// Mock Stripe
vi.mock('@/lib/stripe', () => ({
  getStripe: vi.fn(() =>
    Promise.resolve({
      confirmPayment: vi.fn(),
      retrievePaymentIntent: vi.fn(),
    })
  ),
}));

describe('StripeService', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('createPaymentIntent', () => {
    it('should create payment intent with valid amount', async () => {
      global.fetch = vi.fn().mockResolvedValue({
        ok: true,
        json: () =>
          Promise.resolve({
            clientSecret: 'pi_test_secret',
            paymentIntentId: 'pi_test_123',
          }),
      });

      const result = await stripeService.createPaymentIntent({
        amount: 7900, // $79.00
        currency: 'usd',
        metadata: { orderId: 'order-1' },
      });

      expect(result).toBeDefined();
      expect(result.clientSecret).toBe('pi_test_secret');
      expect(result.paymentIntentId).toBe('pi_test_123');
    });

    it('should throw error for invalid amount (zero)', async () => {
      await expect(
        stripeService.createPaymentIntent({
          amount: 0,
        })
      ).rejects.toThrow('Amount must be greater than 0');
    });

    it('should throw error for negative amount', async () => {
      await expect(
        stripeService.createPaymentIntent({
          amount: -100,
        })
      ).rejects.toThrow('Amount must be greater than 0');
    });

    it('should default to USD currency', async () => {
      const fetchMock = vi.fn().mockResolvedValue({
        ok: true,
        json: () =>
          Promise.resolve({
            clientSecret: 'pi_test_secret',
            paymentIntentId: 'pi_test_123',
          }),
      });

      global.fetch = fetchMock;

      await stripeService.createPaymentIntent({
        amount: 5000,
      });

      expect(fetchMock).toHaveBeenCalled();
      const callArgs = fetchMock.mock.calls[0];
      const body = JSON.parse(callArgs[1].body);
      expect(body.currency).toBe('usd');
    });

    it('should include metadata in request', async () => {
      const fetchMock = vi.fn().mockResolvedValue({
        ok: true,
        json: () =>
          Promise.resolve({
            clientSecret: 'pi_test_secret',
            paymentIntentId: 'pi_test_123',
          }),
      });

      global.fetch = fetchMock;

      await stripeService.createPaymentIntent({
        amount: 10000,
        metadata: {
          orderId: 'order-123',
          customerId: 'customer-456',
        },
      });

      const callArgs = fetchMock.mock.calls[0];
      const body = JSON.parse(callArgs[1].body);
      expect(body.metadata).toEqual({
        orderId: 'order-123',
        customerId: 'customer-456',
      });
    });

    it('should handle API errors gracefully', async () => {
      global.fetch = vi.fn().mockResolvedValue({
        ok: false,
        status: 400,
        statusText: 'Bad Request',
      });

      await expect(
        stripeService.createPaymentIntent({
          amount: 5000,
        })
      ).rejects.toThrow('Failed to create payment intent');
    });

    it('should handle network errors', async () => {
      global.fetch = vi.fn().mockRejectedValue(new Error('Network error'));

      await expect(
        stripeService.createPaymentIntent({
          amount: 5000,
        })
      ).rejects.toThrow();
    });
  });

  describe('confirmPayment', () => {
    it('should confirm payment with valid client secret', async () => {
      const { getStripe } = await import('@/lib/stripe');
      const mockStripe = await getStripe();

      vi.mocked(mockStripe.confirmPayment).mockResolvedValue({
        paymentIntent: {
          id: 'pi_test_123',
          status: 'succeeded',
        },
        error: null,
      } as any);

      const result = await stripeService.confirmPayment('pi_test_secret', {
        return_url: 'https://example.com/success',
      });

      expect(result.success).toBe(true);
      expect(result.paymentIntent?.status).toBe('succeeded');
    });

    it('should handle payment confirmation errors', async () => {
      const { getStripe } = await import('@/lib/stripe');
      const mockStripe = await getStripe();

      vi.mocked(mockStripe.confirmPayment).mockResolvedValue({
        error: {
          type: 'card_error',
          code: 'card_declined',
          message: 'Your card was declined',
        },
        paymentIntent: null,
      } as any);

      const result = await stripeService.confirmPayment('pi_test_secret', {
        return_url: 'https://example.com/success',
      });

      expect(result.success).toBe(false);
      expect(result.error).toBeDefined();
      expect(result.error?.message).toBe('Your card was declined');
    });

    it('should handle authentication required', async () => {
      const { getStripe } = await import('@/lib/stripe');
      const mockStripe = await getStripe();

      vi.mocked(mockStripe.confirmPayment).mockResolvedValue({
        paymentIntent: {
          id: 'pi_test_123',
          status: 'requires_action',
        },
        error: null,
      } as any);

      const result = await stripeService.confirmPayment('pi_test_secret', {
        return_url: 'https://example.com/success',
      });

      expect(result.success).toBe(false);
      expect(result.paymentIntent?.status).toBe('requires_action');
    });
  });

  describe('retrievePaymentIntent', () => {
    it('should retrieve payment intent by ID', async () => {
      const { getStripe } = await import('@/lib/stripe');
      const mockStripe = await getStripe();

      vi.mocked(mockStripe.retrievePaymentIntent).mockResolvedValue({
        paymentIntent: {
          id: 'pi_test_123',
          status: 'succeeded',
          amount: 7900,
          currency: 'usd',
        },
        error: null,
      } as any);

      const result = await stripeService.retrievePaymentIntent('pi_test_secret');

      expect(result).toBeDefined();
      expect(result?.id).toBe('pi_test_123');
      expect(result?.status).toBe('succeeded');
      expect(result?.amount).toBe(7900);
    });

    it('should return null for invalid payment intent', async () => {
      const { getStripe } = await import('@/lib/stripe');
      const mockStripe = await getStripe();

      vi.mocked(mockStripe.retrievePaymentIntent).mockResolvedValue({
        error: {
          type: 'invalid_request_error',
          message: 'No such payment intent',
        },
        paymentIntent: null,
      } as any);

      const result = await stripeService.retrievePaymentIntent('invalid_secret');

      expect(result).toBeNull();
    });
  });

  describe('amount validation', () => {
    it('should accept minimum amount ($0.50 USD)', async () => {
      global.fetch = vi.fn().mockResolvedValue({
        ok: true,
        json: () =>
          Promise.resolve({
            clientSecret: 'pi_test_secret',
            paymentIntentId: 'pi_test_123',
          }),
      });

      const result = await stripeService.createPaymentIntent({
        amount: 50, // $0.50
        currency: 'usd',
      });

      expect(result).toBeDefined();
    });

    it('should accept large amounts', async () => {
      global.fetch = vi.fn().mockResolvedValue({
        ok: true,
        json: () =>
          Promise.resolve({
            clientSecret: 'pi_test_secret',
            paymentIntentId: 'pi_test_123',
          }),
      });

      const result = await stripeService.createPaymentIntent({
        amount: 99999999, // $999,999.99
        currency: 'usd',
      });

      expect(result).toBeDefined();
    });
  });

  describe('currency handling', () => {
    it('should support multiple currencies', async () => {
      const fetchMock = vi.fn().mockResolvedValue({
        ok: true,
        json: () =>
          Promise.resolve({
            clientSecret: 'pi_test_secret',
            paymentIntentId: 'pi_test_123',
          }),
      });

      global.fetch = fetchMock;

      await stripeService.createPaymentIntent({
        amount: 5000,
        currency: 'eur',
      });

      const callArgs = fetchMock.mock.calls[0];
      const body = JSON.parse(callArgs[1].body);
      expect(body.currency).toBe('eur');
    });
  });
});
