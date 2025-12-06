/**
 * Email Service Tests
 *
 * Comprehensive test suite for email notifications
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { emailService } from '../emailService';
import type { Order, ShippingAddress, OrderItem } from '@/types/order';
import type { CartItem } from '@/types/cart';

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

const mockOrderItems: OrderItem[] = [
  {
    id: 'item-1',
    name: 'Black Hoodie',
    price: 7900,
    quantity: 2,
    image: '/images/hoodie-black.webp',
    variant: 'L',
  },
];

const mockOrder: Order = {
  id: 'order-123',
  orderNumber: 'LII-001',
  customerId: 'customer-456',
  items: mockOrderItems,
  subtotal: 15800,
  shipping: 1000,
  tax: 1512,
  total: 18312,
  shippingAddress: mockShippingAddress,
  status: 'pending',
  paymentStatus: 'paid',
  createdAt: new Date('2024-01-01T00:00:00Z'),
  stripePaymentIntentId: 'pi_test_123',
};

const mockCartItems: CartItem[] = [
  {
    id: 'cart-1',
    productId: 'product-1',
    name: 'Black Hoodie',
    price: 79,
    quantity: 2,
    image: '/images/hoodie-black.webp',
    variant: { size: 'L' },
  },
];

describe('EmailService', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('sendOrderConfirmation', () => {
    it('should send order confirmation email successfully', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: async () => ({}),
      });

      const result = await emailService.sendOrderConfirmation(
        'customer@example.com',
        mockOrder
      );

      expect(result).toBe(true);
      expect(mockFetch).toHaveBeenCalledWith('/api/email/order-confirmation', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          email: 'customer@example.com',
          order: mockOrder,
        }),
      });
    });

    it('should send to shipping address email', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: async () => ({}),
      });

      const result = await emailService.sendOrderConfirmation(
        mockOrder.shippingAddress.email,
        mockOrder
      );

      expect(result).toBe(true);
    });

    it('should return false when email send fails', async () => {
      mockFetch.mockResolvedValue({
        ok: false,
        status: 400,
        json: async () => ({
          message: 'Invalid email address',
        }),
      });

      const result = await emailService.sendOrderConfirmation(
        'invalid-email',
        mockOrder
      );

      expect(result).toBe(false);
    });

    it('should return false on network error', async () => {
      mockFetch.mockRejectedValue(new Error('Network timeout'));

      const result = await emailService.sendOrderConfirmation(
        'customer@example.com',
        mockOrder
      );

      expect(result).toBe(false);
    });

    it('should console.error on failure', async () => {
      const consoleErrorSpy = vi.spyOn(console, 'error').mockImplementation(() => {});

      mockFetch.mockRejectedValue(new Error('Email service unavailable'));

      await emailService.sendOrderConfirmation('customer@example.com', mockOrder);

      expect(consoleErrorSpy).toHaveBeenCalledWith(
        'Failed to send order confirmation email:',
        expect.any(Error)
      );

      consoleErrorSpy.mockRestore();
    });

    it('should handle API server errors', async () => {
      mockFetch.mockResolvedValue({
        ok: false,
        status: 500,
        json: async () => ({
          message: 'Email server error',
        }),
      });

      const result = await emailService.sendOrderConfirmation(
        'customer@example.com',
        mockOrder
      );

      expect(result).toBe(false);
    });

    it('should send email with complete order data', async () => {
      const fullOrder: Order = {
        ...mockOrder,
        billingAddress: mockShippingAddress,
        notes: 'Please handle with care',
        trackingNumber: 'TRACK-123',
      };

      mockFetch.mockResolvedValue({
        ok: true,
        json: async () => ({}),
      });

      const result = await emailService.sendOrderConfirmation(
        'customer@example.com',
        fullOrder
      );

      expect(result).toBe(true);

      const callArgs = JSON.parse(mockFetch.mock.calls[0][1].body);
      expect(callArgs.order.notes).toBe('Please handle with care');
    });
  });

  describe('sendShippingNotification', () => {
    it('should send shipping notification successfully', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: async () => ({}),
      });

      const result = await emailService.sendShippingNotification(
        'customer@example.com',
        mockOrder,
        'TRACK-123456'
      );

      expect(result).toBe(true);
      expect(mockFetch).toHaveBeenCalledWith('/api/email/shipping-notification', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          email: 'customer@example.com',
          order: mockOrder,
          trackingNumber: 'TRACK-123456',
        }),
      });
    });

    it('should send notification with USPS tracking number', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: async () => ({}),
      });

      const result = await emailService.sendShippingNotification(
        'customer@example.com',
        mockOrder,
        '9400111899562844058085'
      );

      expect(result).toBe(true);

      const callArgs = JSON.parse(mockFetch.mock.calls[0][1].body);
      expect(callArgs.trackingNumber).toBe('9400111899562844058085');
    });

    it('should send notification with FedEx tracking number', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: async () => ({}),
      });

      const result = await emailService.sendShippingNotification(
        'customer@example.com',
        mockOrder,
        '7771234567890'
      );

      expect(result).toBe(true);
    });

    it('should return false when email send fails', async () => {
      mockFetch.mockResolvedValue({
        ok: false,
        status: 400,
        json: async () => ({
          message: 'Invalid email',
        }),
      });

      const result = await emailService.sendShippingNotification(
        'invalid@',
        mockOrder,
        'TRACK-123'
      );

      expect(result).toBe(false);
    });

    it('should return false on network error', async () => {
      mockFetch.mockRejectedValue(new Error('Connection failed'));

      const result = await emailService.sendShippingNotification(
        'customer@example.com',
        mockOrder,
        'TRACK-123'
      );

      expect(result).toBe(false);
    });

    it('should console.error on failure', async () => {
      const consoleErrorSpy = vi.spyOn(console, 'error').mockImplementation(() => {});

      mockFetch.mockRejectedValue(new Error('Email error'));

      await emailService.sendShippingNotification(
        'customer@example.com',
        mockOrder,
        'TRACK-123'
      );

      expect(consoleErrorSpy).toHaveBeenCalledWith(
        'Failed to send shipping notification:',
        expect.any(Error)
      );

      consoleErrorSpy.mockRestore();
    });

    it('should handle API errors', async () => {
      mockFetch.mockResolvedValue({
        ok: false,
        status: 500,
        json: async () => ({
          message: 'Email service down',
        }),
      });

      const result = await emailService.sendShippingNotification(
        'customer@example.com',
        mockOrder,
        'TRACK-123'
      );

      expect(result).toBe(false);
    });
  });

  describe('sendAbandonedCartEmail', () => {
    it('should send abandoned cart email successfully', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: async () => ({}),
      });

      const result = await emailService.sendAbandonedCartEmail(
        'customer@example.com',
        mockCartItems
      );

      expect(result).toBe(true);
      expect(mockFetch).toHaveBeenCalledWith('/api/email/abandoned-cart', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          email: 'customer@example.com',
          cartItems: mockCartItems,
        }),
      });
    });

    it('should send email with single cart item', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: async () => ({}),
      });

      const singleItem = [mockCartItems[0]];

      const result = await emailService.sendAbandonedCartEmail(
        'customer@example.com',
        singleItem
      );

      expect(result).toBe(true);
    });

    it('should send email with multiple cart items', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: async () => ({}),
      });

      const multipleItems: CartItem[] = [
        ...mockCartItems,
        {
          id: 'cart-2',
          productId: 'product-2',
          name: 'Black Cap',
          price: 29,
          quantity: 1,
          image: '/images/cap-black.webp',
        },
      ];

      const result = await emailService.sendAbandonedCartEmail(
        'customer@example.com',
        multipleItems
      );

      expect(result).toBe(true);

      const callArgs = JSON.parse(mockFetch.mock.calls[0][1].body);
      expect(callArgs.cartItems.length).toBe(2);
    });

    it('should send email with empty cart', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: async () => ({}),
      });

      const result = await emailService.sendAbandonedCartEmail('customer@example.com', []);

      expect(result).toBe(true);
    });

    it('should return false when email send fails', async () => {
      mockFetch.mockResolvedValue({
        ok: false,
        status: 400,
        json: async () => ({
          message: 'Invalid request',
        }),
      });

      const result = await emailService.sendAbandonedCartEmail(
        'customer@example.com',
        mockCartItems
      );

      expect(result).toBe(false);
    });

    it('should return false on network error', async () => {
      mockFetch.mockRejectedValue(new Error('Network error'));

      const result = await emailService.sendAbandonedCartEmail(
        'customer@example.com',
        mockCartItems
      );

      expect(result).toBe(false);
    });

    it('should console.error on failure', async () => {
      const consoleErrorSpy = vi.spyOn(console, 'error').mockImplementation(() => {});

      mockFetch.mockRejectedValue(new Error('Send failed'));

      await emailService.sendAbandonedCartEmail('customer@example.com', mockCartItems);

      expect(consoleErrorSpy).toHaveBeenCalledWith(
        'Failed to send abandoned cart email:',
        expect.any(Error)
      );

      consoleErrorSpy.mockRestore();
    });
  });

  describe('subscribeNewsletter', () => {
    it('should subscribe to newsletter successfully', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: async () => ({}),
      });

      const result = await emailService.subscribeNewsletter('subscriber@example.com');

      expect(result).toBe(true);
      expect(mockFetch).toHaveBeenCalledWith('/api/email/newsletter', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          email: 'subscriber@example.com',
        }),
      });
    });

    it('should return false for invalid email', async () => {
      mockFetch.mockResolvedValue({
        ok: false,
        status: 400,
        json: async () => ({
          message: 'Invalid email format',
        }),
      });

      const result = await emailService.subscribeNewsletter('invalid-email');

      expect(result).toBe(false);
    });

    it('should return false when email already subscribed', async () => {
      mockFetch.mockResolvedValue({
        ok: false,
        status: 409,
        json: async () => ({
          message: 'Email already subscribed',
        }),
      });

      const result = await emailService.subscribeNewsletter('existing@example.com');

      expect(result).toBe(false);
    });

    it('should return false on network error', async () => {
      mockFetch.mockRejectedValue(new Error('Connection timeout'));

      const result = await emailService.subscribeNewsletter('subscriber@example.com');

      expect(result).toBe(false);
    });

    it('should console.error on failure', async () => {
      const consoleErrorSpy = vi.spyOn(console, 'error').mockImplementation(() => {});

      mockFetch.mockRejectedValue(new Error('Newsletter error'));

      await emailService.subscribeNewsletter('subscriber@example.com');

      expect(consoleErrorSpy).toHaveBeenCalledWith(
        'Failed to subscribe newsletter:',
        expect.any(Error)
      );

      consoleErrorSpy.mockRestore();
    });

    it('should handle API errors', async () => {
      mockFetch.mockResolvedValue({
        ok: false,
        status: 500,
        json: async () => ({
          message: 'Server error',
        }),
      });

      const result = await emailService.subscribeNewsletter('subscriber@example.com');

      expect(result).toBe(false);
    });

    it('should handle various email formats', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: async () => ({}),
      });

      // Standard email
      let result = await emailService.subscribeNewsletter('user@example.com');
      expect(result).toBe(true);

      // Email with dots
      result = await emailService.subscribeNewsletter('first.last@example.com');
      expect(result).toBe(true);

      // Email with subdomain
      result = await emailService.subscribeNewsletter('user@mail.example.com');
      expect(result).toBe(true);
    });
  });

  describe('email flow integration', () => {
    it('should handle complete order email flow', async () => {
      // Step 1: Send order confirmation
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({}),
      });

      const confirmationSent = await emailService.sendOrderConfirmation(
        'customer@example.com',
        mockOrder
      );

      expect(confirmationSent).toBe(true);

      // Step 2: Send shipping notification
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({}),
      });

      const shippedOrder = {
        ...mockOrder,
        status: 'shipped' as const,
        trackingNumber: 'TRACK-789',
      };

      const shippingNotificationSent = await emailService.sendShippingNotification(
        'customer@example.com',
        shippedOrder,
        'TRACK-789'
      );

      expect(shippingNotificationSent).toBe(true);
    });

    it('should handle abandoned cart recovery flow', async () => {
      // Send abandoned cart email
      mockFetch.mockResolvedValue({
        ok: true,
        json: async () => ({}),
      });

      const abandonedCartSent = await emailService.sendAbandonedCartEmail(
        'customer@example.com',
        mockCartItems
      );

      expect(abandonedCartSent).toBe(true);
    });
  });

  describe('error handling', () => {
    it('should return false for all methods on persistent network errors', async () => {
      mockFetch.mockRejectedValue(new Error('Network down'));

      const orderConfirmation = await emailService.sendOrderConfirmation(
        'test@example.com',
        mockOrder
      );
      expect(orderConfirmation).toBe(false);

      const shipping = await emailService.sendShippingNotification(
        'test@example.com',
        mockOrder,
        'TRACK-123'
      );
      expect(shipping).toBe(false);

      const abandonedCart = await emailService.sendAbandonedCartEmail(
        'test@example.com',
        mockCartItems
      );
      expect(abandonedCart).toBe(false);

      const newsletter = await emailService.subscribeNewsletter('test@example.com');
      expect(newsletter).toBe(false);
    });
  });
});
