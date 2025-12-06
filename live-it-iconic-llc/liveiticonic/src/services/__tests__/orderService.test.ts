/**
 * Order Service Tests
 *
 * Comprehensive test suite for order management
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { orderService } from '../orderService';
import type { Order, ShippingAddress, OrderItem } from '@/types/order';

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
    price: 7900, // $79.00 in cents
    quantity: 2,
    image: '/images/hoodie-black.webp',
    variant: 'L',
  },
  {
    id: 'item-2',
    name: 'Black Cap',
    price: 2900, // $29.00 in cents
    quantity: 1,
    image: '/images/cap-black.webp',
  },
};

const mockOrder: Order = {
  id: 'order-123',
  orderNumber: 'LII-001',
  customerId: 'customer-456',
  items: mockOrderItems,
  subtotal: 18700, // $187.00
  shipping: 1000, // $10.00
  tax: 1683, // $16.83
  total: 21383, // $213.83
  shippingAddress: mockShippingAddress,
  status: 'pending',
  paymentStatus: 'paid',
  createdAt: new Date('2024-01-01T00:00:00Z'),
  stripePaymentIntentId: 'pi_test_123',
};

describe('OrderService', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('createOrder', () => {
    it('should create order successfully', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: async () => ({
          id: 'order-new-123',
          orderNumber: 'LII-002',
          status: 'pending',
          paymentStatus: 'paid',
        }),
      });

      const orderId = await orderService.createOrder({
        shippingData: mockShippingAddress,
        amount: 21383,
        paymentMethod: 'card',
        cardLast4: '4242',
      });

      expect(orderId).toBe('order-new-123');
      expect(mockFetch).toHaveBeenCalledWith('/api/orders/create', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          shippingData: mockShippingAddress,
          amount: 21383,
          paymentMethod: 'card',
          status: 'pending',
          paymentStatus: 'paid',
        }),
      });
    });

    it('should create order with minimal shipping data', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: async () => ({
          id: 'order-minimal',
        }),
      });

      const minimalAddress: ShippingAddress = {
        firstName: 'Jane',
        lastName: 'Smith',
        email: 'jane@example.com',
        phone: '5551234',
        address: '456 Oak Ave',
        city: 'Boston',
        state: 'MA',
        zipCode: '02101',
        country: 'US',
      };

      const orderId = await orderService.createOrder({
        shippingData: minimalAddress,
        amount: 5000,
        paymentMethod: 'card',
      });

      expect(orderId).toBe('order-minimal');
    });

    it('should throw error when API returns 400', async () => {
      mockFetch.mockResolvedValue({
        ok: false,
        status: 400,
        json: async () => ({
          message: 'Invalid shipping address',
        }),
      });

      await expect(
        orderService.createOrder({
          shippingData: mockShippingAddress,
          amount: 10000,
          paymentMethod: 'card',
        })
      ).rejects.toThrow('Invalid shipping address');
    });

    it('should throw error when amount is zero', async () => {
      mockFetch.mockResolvedValue({
        ok: false,
        status: 400,
        json: async () => ({
          message: 'Order amount must be greater than 0',
        }),
      });

      await expect(
        orderService.createOrder({
          shippingData: mockShippingAddress,
          amount: 0,
          paymentMethod: 'card',
        })
      ).rejects.toThrow('Order amount must be greater than 0');
    });

    it('should throw error when amount is negative', async () => {
      mockFetch.mockResolvedValue({
        ok: false,
        status: 400,
        json: async () => ({
          message: 'Invalid amount',
        }),
      });

      await expect(
        orderService.createOrder({
          shippingData: mockShippingAddress,
          amount: -100,
          paymentMethod: 'card',
        })
      ).rejects.toThrow('Invalid amount');
    });

    it('should handle network errors', async () => {
      mockFetch.mockRejectedValue(new Error('Network timeout'));

      await expect(
        orderService.createOrder({
          shippingData: mockShippingAddress,
          amount: 10000,
          paymentMethod: 'card',
        })
      ).rejects.toThrow('Network timeout');
    });

    it('should throw generic error when API fails without message', async () => {
      mockFetch.mockResolvedValue({
        ok: false,
        status: 500,
        json: async () => ({}),
      });

      await expect(
        orderService.createOrder({
          shippingData: mockShippingAddress,
          amount: 10000,
          paymentMethod: 'card',
        })
      ).rejects.toThrow('Failed to create order');
    });

    it('should handle payment processor errors', async () => {
      mockFetch.mockResolvedValue({
        ok: false,
        status: 402,
        json: async () => ({
          message: 'Payment processing failed',
        }),
      });

      await expect(
        orderService.createOrder({
          shippingData: mockShippingAddress,
          amount: 15000,
          paymentMethod: 'card',
          cardLast4: '0002', // Card declined test card
        })
      ).rejects.toThrow('Payment processing failed');
    });

    it('should console.error on failure', async () => {
      const consoleErrorSpy = vi.spyOn(console, 'error').mockImplementation(() => {});

      mockFetch.mockRejectedValue(new Error('API down'));

      await expect(
        orderService.createOrder({
          shippingData: mockShippingAddress,
          amount: 10000,
          paymentMethod: 'card',
        })
      ).rejects.toThrow();

      expect(consoleErrorSpy).toHaveBeenCalledWith(
        'Failed to create order:',
        expect.any(Error)
      );

      consoleErrorSpy.mockRestore();
    });
  });

  describe('getOrder', () => {
    it('should fetch order successfully', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: async () => mockOrder,
      });

      const order = await orderService.getOrder('order-123');

      expect(order).toEqual(mockOrder);
      expect(mockFetch).toHaveBeenCalledWith('/api/orders/order-123', {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      });
    });

    it('should return null when order not found', async () => {
      mockFetch.mockResolvedValue({
        ok: false,
        status: 404,
        json: async () => ({
          message: 'Order not found',
        }),
      });

      const order = await orderService.getOrder('nonexistent-order');

      expect(order).toBeNull();
    });

    it('should return null when API returns error', async () => {
      mockFetch.mockResolvedValue({
        ok: false,
        status: 500,
        json: async () => ({
          message: 'Internal server error',
        }),
      });

      const order = await orderService.getOrder('order-123');

      expect(order).toBeNull();
    });

    it('should handle network errors gracefully', async () => {
      mockFetch.mockRejectedValue(new Error('Network error'));

      const order = await orderService.getOrder('order-123');

      expect(order).toBeNull();
    });

    it('should console.error on failure', async () => {
      const consoleErrorSpy = vi.spyOn(console, 'error').mockImplementation(() => {});

      mockFetch.mockRejectedValue(new Error('Fetch failed'));

      const order = await orderService.getOrder('order-123');

      expect(order).toBeNull();
      expect(consoleErrorSpy).toHaveBeenCalledWith(
        'Failed to get order:',
        expect.any(Error)
      );

      consoleErrorSpy.mockRestore();
    });

    it('should handle order with full data (including optional fields)', async () => {
      const fullOrder: Order = {
        ...mockOrder,
        billingAddress: mockShippingAddress,
        trackingNumber: 'TRACK-123456',
        notes: 'Handle with care',
        updatedAt: new Date('2024-01-02T00:00:00Z'),
        completedAt: new Date('2024-01-03T00:00:00Z'),
      };

      mockFetch.mockResolvedValue({
        ok: true,
        json: async () => fullOrder,
      });

      const order = await orderService.getOrder('order-full');

      expect(order).toEqual(fullOrder);
      expect(order?.trackingNumber).toBe('TRACK-123456');
      expect(order?.notes).toBe('Handle with care');
    });
  });

  describe('getCustomerOrders', () => {
    it('should fetch customer orders successfully', async () => {
      const orders = [mockOrder, { ...mockOrder, id: 'order-456' }];

      mockFetch.mockResolvedValue({
        ok: true,
        json: async () => ({
          orders,
        }),
      });

      const result = await orderService.getCustomerOrders('customer-456');

      expect(result).toEqual(orders);
      expect(result.length).toBe(2);
      expect(mockFetch).toHaveBeenCalledWith('/api/orders/customer/customer-456', {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      });
    });

    it('should return empty array when customer has no orders', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: async () => ({
          orders: [],
        }),
      });

      const orders = await orderService.getCustomerOrders('new-customer');

      expect(orders).toEqual([]);
    });

    it('should return empty array when orders property is missing', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: async () => ({}), // No orders property
      });

      const orders = await orderService.getCustomerOrders('customer-123');

      expect(orders).toEqual([]);
    });

    it('should return empty array when API returns error', async () => {
      mockFetch.mockResolvedValue({
        ok: false,
        status: 404,
        json: async () => ({
          message: 'Customer not found',
        }),
      });

      const orders = await orderService.getCustomerOrders('nonexistent-customer');

      expect(orders).toEqual([]);
    });

    it('should handle network errors gracefully', async () => {
      mockFetch.mockRejectedValue(new Error('Network timeout'));

      const orders = await orderService.getCustomerOrders('customer-123');

      expect(orders).toEqual([]);
    });

    it('should console.error on failure', async () => {
      const consoleErrorSpy = vi.spyOn(console, 'error').mockImplementation(() => {});

      mockFetch.mockRejectedValue(new Error('API error'));

      const orders = await orderService.getCustomerOrders('customer-123');

      expect(orders).toEqual([]);
      expect(consoleErrorSpy).toHaveBeenCalledWith(
        'Failed to get customer orders:',
        expect.any(Error)
      );

      consoleErrorSpy.mockRestore();
    });

    it('should handle multiple orders with different statuses', async () => {
      const orders = [
        { ...mockOrder, id: 'order-1', status: 'pending' as const },
        { ...mockOrder, id: 'order-2', status: 'confirmed' as const },
        { ...mockOrder, id: 'order-3', status: 'shipped' as const },
        { ...mockOrder, id: 'order-4', status: 'delivered' as const },
      ];

      mockFetch.mockResolvedValue({
        ok: true,
        json: async () => ({
          orders,
        }),
      });

      const result = await orderService.getCustomerOrders('customer-multi');

      expect(result.length).toBe(4);
      expect(result[0].status).toBe('pending');
      expect(result[1].status).toBe('confirmed');
      expect(result[2].status).toBe('shipped');
      expect(result[3].status).toBe('delivered');
    });
  });

  describe('updateOrderStatus', () => {
    it('should update order status successfully', async () => {
      const updatedOrder = {
        ...mockOrder,
        status: 'confirmed' as const,
        updatedAt: new Date('2024-01-02T00:00:00Z'),
      };

      mockFetch.mockResolvedValue({
        ok: true,
        json: async () => updatedOrder,
      });

      const result = await orderService.updateOrderStatus('order-123', 'confirmed');

      expect(result).toEqual(updatedOrder);
      expect(result?.status).toBe('confirmed');
      expect(mockFetch).toHaveBeenCalledWith('/api/orders/order-123', {
        method: 'PATCH',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ status: 'confirmed' }),
      });
    });

    it('should update status to shipped', async () => {
      const shippedOrder = {
        ...mockOrder,
        status: 'shipped' as const,
        trackingNumber: 'TRACK-789',
      };

      mockFetch.mockResolvedValue({
        ok: true,
        json: async () => shippedOrder,
      });

      const result = await orderService.updateOrderStatus('order-123', 'shipped');

      expect(result?.status).toBe('shipped');
    });

    it('should update status to cancelled', async () => {
      const cancelledOrder = {
        ...mockOrder,
        status: 'cancelled' as const,
      };

      mockFetch.mockResolvedValue({
        ok: true,
        json: async () => cancelledOrder,
      });

      const result = await orderService.updateOrderStatus('order-123', 'cancelled');

      expect(result?.status).toBe('cancelled');
    });

    it('should return null when order not found', async () => {
      mockFetch.mockResolvedValue({
        ok: false,
        status: 404,
        json: async () => ({
          message: 'Order not found',
        }),
      });

      const result = await orderService.updateOrderStatus('nonexistent', 'confirmed');

      expect(result).toBeNull();
    });

    it('should return null when invalid status provided', async () => {
      mockFetch.mockResolvedValue({
        ok: false,
        status: 400,
        json: async () => ({
          message: 'Invalid status',
        }),
      });

      const result = await orderService.updateOrderStatus('order-123', 'invalid-status');

      expect(result).toBeNull();
    });

    it('should return null when API returns error', async () => {
      mockFetch.mockResolvedValue({
        ok: false,
        status: 500,
        json: async () => ({
          message: 'Internal server error',
        }),
      });

      const result = await orderService.updateOrderStatus('order-123', 'confirmed');

      expect(result).toBeNull();
    });

    it('should handle network errors gracefully', async () => {
      mockFetch.mockRejectedValue(new Error('Network error'));

      const result = await orderService.updateOrderStatus('order-123', 'confirmed');

      expect(result).toBeNull();
    });

    it('should console.error on failure', async () => {
      const consoleErrorSpy = vi.spyOn(console, 'error').mockImplementation(() => {});

      mockFetch.mockRejectedValue(new Error('Update failed'));

      const result = await orderService.updateOrderStatus('order-123', 'confirmed');

      expect(result).toBeNull();
      expect(consoleErrorSpy).toHaveBeenCalledWith(
        'Failed to update order status:',
        expect.any(Error)
      );

      consoleErrorSpy.mockRestore();
    });

    it('should handle status progression', async () => {
      // pending -> confirmed
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ ...mockOrder, status: 'confirmed' }),
      });

      let result = await orderService.updateOrderStatus('order-123', 'confirmed');
      expect(result?.status).toBe('confirmed');

      // confirmed -> processing
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ ...mockOrder, status: 'processing' }),
      });

      result = await orderService.updateOrderStatus('order-123', 'processing');
      expect(result?.status).toBe('processing');

      // processing -> shipped
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ ...mockOrder, status: 'shipped' }),
      });

      result = await orderService.updateOrderStatus('order-123', 'shipped');
      expect(result?.status).toBe('shipped');

      // shipped -> delivered
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ ...mockOrder, status: 'delivered' }),
      });

      result = await orderService.updateOrderStatus('order-123', 'delivered');
      expect(result?.status).toBe('delivered');
    });
  });
});
