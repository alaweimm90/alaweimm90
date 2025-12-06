/**
 * Payment Service Tests
 *
 * Comprehensive test suite for payment processing with Stripe
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { paymentService } from '../paymentService';
import type { ShippingAddress } from '@/types/order';

// Mock fetch globally
const mockFetch = vi.fn();
global.fetch = mockFetch;

// Test data
const mockShippingAddress: ShippingAddress = {
  firstName: 'John',
  lastName: 'Doe',
  email: 'john.doe@example.com',
  phone: '+1 555-0123',
  address: '123 Main St',
  city: 'Los Angeles',
  state: 'CA',
  zipCode: '90001',
  country: 'US',
};

describe('PaymentService', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('createPaymentIntent', () => {
    it('should create payment intent successfully', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: async () => ({
          clientSecret: 'pi_test_secret_abc123',
          paymentIntentId: 'pi_test_abc123',
        }),
      });

      const result = await paymentService.createPaymentIntent({
        amount: 79.0, // $79.00
        currency: 'usd',
      });

      expect(result.clientSecret).toBe('pi_test_secret_abc123');
      expect(result.paymentIntentId).toBe('pi_test_abc123');

      expect(mockFetch).toHaveBeenCalledWith('/api/payments/create-intent', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          amount: 7900, // Converted to cents
          currency: 'usd',
          orderData: undefined,
        }),
      });
    });

    it('should convert amount to cents correctly', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: async () => ({
          clientSecret: 'pi_secret',
          paymentIntentId: 'pi_123',
        }),
      });

      await paymentService.createPaymentIntent({
        amount: 213.83, // $213.83
      });

      const callArgs = JSON.parse(mockFetch.mock.calls[0][1].body);
      expect(callArgs.amount).toBe(21383); // 213.83 * 100
    });

    it('should default currency to usd', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: async () => ({
          clientSecret: 'pi_secret',
          paymentIntentId: 'pi_123',
        }),
      });

      await paymentService.createPaymentIntent({
        amount: 50.0,
      });

      const callArgs = JSON.parse(mockFetch.mock.calls[0][1].body);
      expect(callArgs.currency).toBe('usd');
    });

    it('should support different currencies', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: async () => ({
          clientSecret: 'pi_secret',
          paymentIntentId: 'pi_123',
        }),
      });

      await paymentService.createPaymentIntent({
        amount: 100.0,
        currency: 'eur',
      });

      const callArgs = JSON.parse(mockFetch.mock.calls[0][1].body);
      expect(callArgs.currency).toBe('eur');
    });

    it('should include order data when provided', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: async () => ({
          clientSecret: 'pi_secret',
          paymentIntentId: 'pi_123',
        }),
      });

      await paymentService.createPaymentIntent({
        amount: 79.0,
        orderData: {
          items: [
            {
              id: 'item-1',
              name: 'Black Hoodie',
              quantity: 1,
              price: 7900,
            },
          ],
          shippingAddress: mockShippingAddress,
        },
      });

      const callArgs = JSON.parse(mockFetch.mock.calls[0][1].body);
      expect(callArgs.orderData).toBeDefined();
      expect(callArgs.orderData.items).toHaveLength(1);
      expect(callArgs.orderData.shippingAddress.firstName).toBe('John');
    });

    it('should round amount to avoid floating point issues', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: async () => ({
          clientSecret: 'pi_secret',
          paymentIntentId: 'pi_123',
        }),
      });

      await paymentService.createPaymentIntent({
        amount: 29.997, // Should round to 3000 cents
      });

      const callArgs = JSON.parse(mockFetch.mock.calls[0][1].body);
      expect(callArgs.amount).toBe(3000); // Math.round(29.997 * 100) = 3000
    });

    it('should handle zero amount', async () => {
      mockFetch.mockResolvedValue({
        ok: false,
        status: 400,
        json: async () => ({
          message: 'Amount must be greater than 0',
        }),
      });

      await expect(
        paymentService.createPaymentIntent({
          amount: 0,
        })
      ).rejects.toThrow('Amount must be greater than 0');
    });

    it('should handle negative amount', async () => {
      mockFetch.mockResolvedValue({
        ok: false,
        status: 400,
        json: async () => ({
          message: 'Invalid amount',
        }),
      });

      await expect(
        paymentService.createPaymentIntent({
          amount: -50,
        })
      ).rejects.toThrow('Invalid amount');
    });

    it('should handle API errors', async () => {
      mockFetch.mockResolvedValue({
        ok: false,
        status: 500,
        json: async () => ({
          message: 'Stripe API unavailable',
        }),
      });

      await expect(
        paymentService.createPaymentIntent({
          amount: 100,
        })
      ).rejects.toThrow('Stripe API unavailable');
    });

    it('should handle network errors', async () => {
      mockFetch.mockRejectedValue(new Error('Network timeout'));

      await expect(
        paymentService.createPaymentIntent({
          amount: 100,
        })
      ).rejects.toThrow('Network timeout');
    });

    it('should handle generic failures with fallback message', async () => {
      mockFetch.mockResolvedValue({
        ok: false,
        status: 400,
        json: async () => ({}), // No message
      });

      await expect(
        paymentService.createPaymentIntent({
          amount: 100,
        })
      ).rejects.toThrow('Failed to create payment intent');
    });

    it('should console.error on failure', async () => {
      const consoleErrorSpy = vi.spyOn(console, 'error').mockImplementation(() => {});

      mockFetch.mockRejectedValue(new Error('API error'));

      await expect(
        paymentService.createPaymentIntent({
          amount: 100,
        })
      ).rejects.toThrow();

      expect(consoleErrorSpy).toHaveBeenCalledWith(
        'Payment intent creation failed:',
        expect.any(Error)
      );

      consoleErrorSpy.mockRestore();
    });

    it('should handle large amounts', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: async () => ({
          clientSecret: 'pi_secret',
          paymentIntentId: 'pi_large',
        }),
      });

      await paymentService.createPaymentIntent({
        amount: 9999.99, // $9,999.99
      });

      const callArgs = JSON.parse(mockFetch.mock.calls[0][1].body);
      expect(callArgs.amount).toBe(999999); // 9999.99 * 100
    });
  });

  describe('confirmPayment', () => {
    it('should confirm payment successfully', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: async () => ({
          success: true,
          orderId: 'order-confirmed-123',
        }),
      });

      const result = await paymentService.confirmPayment({
        paymentIntentId: 'pi_test_123',
        paymentMethodId: 'pm_card_visa',
      });

      expect(result.success).toBe(true);
      expect(result.orderId).toBe('order-confirmed-123');

      expect(mockFetch).toHaveBeenCalledWith('/api/payments/confirm-payment', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          paymentIntentId: 'pi_test_123',
          paymentMethodId: 'pm_card_visa',
        }),
      });
    });

    it('should confirm payment without orderId', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: async () => ({
          success: true,
          // orderId may not be returned in some cases
        }),
      });

      const result = await paymentService.confirmPayment({
        paymentIntentId: 'pi_test_456',
        paymentMethodId: 'pm_card_mastercard',
      });

      expect(result.success).toBe(true);
      expect(result.orderId).toBeUndefined();
    });

    it('should handle card declined error', async () => {
      mockFetch.mockResolvedValue({
        ok: false,
        status: 402,
        json: async () => ({
          message: 'Your card was declined',
        }),
      });

      await expect(
        paymentService.confirmPayment({
          paymentIntentId: 'pi_test_123',
          paymentMethodId: 'pm_card_declined',
        })
      ).rejects.toThrow('Your card was declined');
    });

    it('should handle insufficient funds error', async () => {
      mockFetch.mockResolvedValue({
        ok: false,
        status: 402,
        json: async () => ({
          message: 'Insufficient funds',
        }),
      });

      await expect(
        paymentService.confirmPayment({
          paymentIntentId: 'pi_test_123',
          paymentMethodId: 'pm_insufficient_funds',
        })
      ).rejects.toThrow('Insufficient funds');
    });

    it('should handle invalid payment method error', async () => {
      mockFetch.mockResolvedValue({
        ok: false,
        status: 400,
        json: async () => ({
          message: 'Invalid payment method',
        }),
      });

      await expect(
        paymentService.confirmPayment({
          paymentIntentId: 'pi_test_123',
          paymentMethodId: 'pm_invalid',
        })
      ).rejects.toThrow('Invalid payment method');
    });

    it('should handle payment intent not found', async () => {
      mockFetch.mockResolvedValue({
        ok: false,
        status: 404,
        json: async () => ({
          message: 'Payment intent not found',
        }),
      });

      await expect(
        paymentService.confirmPayment({
          paymentIntentId: 'pi_nonexistent',
          paymentMethodId: 'pm_card_visa',
        })
      ).rejects.toThrow('Payment intent not found');
    });

    it('should handle network errors', async () => {
      mockFetch.mockRejectedValue(new Error('Connection timeout'));

      await expect(
        paymentService.confirmPayment({
          paymentIntentId: 'pi_test_123',
          paymentMethodId: 'pm_card_visa',
        })
      ).rejects.toThrow('Connection timeout');
    });

    it('should handle generic failure with fallback message', async () => {
      mockFetch.mockResolvedValue({
        ok: false,
        status: 500,
        json: async () => ({}), // No message
      });

      await expect(
        paymentService.confirmPayment({
          paymentIntentId: 'pi_test_123',
          paymentMethodId: 'pm_card_visa',
        })
      ).rejects.toThrow('Payment confirmation failed');
    });

    it('should console.error on failure', async () => {
      const consoleErrorSpy = vi.spyOn(console, 'error').mockImplementation(() => {});

      mockFetch.mockRejectedValue(new Error('Payment error'));

      await expect(
        paymentService.confirmPayment({
          paymentIntentId: 'pi_test_123',
          paymentMethodId: 'pm_card_visa',
        })
      ).rejects.toThrow();

      expect(consoleErrorSpy).toHaveBeenCalledWith(
        'Payment confirmation failed:',
        expect.any(Error)
      );

      consoleErrorSpy.mockRestore();
    });

    it('should handle authentication required (3D Secure)', async () => {
      mockFetch.mockResolvedValue({
        ok: false,
        status: 402,
        json: async () => ({
          message: 'Authentication required',
        }),
      });

      await expect(
        paymentService.confirmPayment({
          paymentIntentId: 'pi_requires_auth',
          paymentMethodId: 'pm_card_3ds',
        })
      ).rejects.toThrow('Authentication required');
    });
  });

  describe('payment flow integration', () => {
    it('should handle complete payment flow (create then confirm)', async () => {
      // Step 1: Create payment intent
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          clientSecret: 'pi_flow_secret',
          paymentIntentId: 'pi_flow_123',
        }),
      });

      const intent = await paymentService.createPaymentIntent({
        amount: 79.0,
      });

      expect(intent.paymentIntentId).toBe('pi_flow_123');

      // Step 2: Confirm payment
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          success: true,
          orderId: 'order-completed',
        }),
      });

      const confirmation = await paymentService.confirmPayment({
        paymentIntentId: intent.paymentIntentId,
        paymentMethodId: 'pm_card_visa',
      });

      expect(confirmation.success).toBe(true);
      expect(confirmation.orderId).toBe('order-completed');
    });
  });
});
