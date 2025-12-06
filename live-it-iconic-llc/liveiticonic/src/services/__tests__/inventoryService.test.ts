/**
 * Inventory Service Tests
 *
 * Comprehensive test suite for inventory management
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { inventoryService } from '../inventoryService';

// Mock fetch globally
const mockFetch = vi.fn();
global.fetch = mockFetch;

describe('InventoryService', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('checkAvailability', () => {
    it('should check availability successfully', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: async () => ({
          available: true,
        }),
      });

      const result = await inventoryService.checkAvailability({
        productId: 'product-123',
        quantity: 2,
      });

      expect(result).toBe(true);
      expect(mockFetch).toHaveBeenCalledWith('/api/inventory/check', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          productId: 'product-123',
          quantity: 2,
          variantId: undefined,
        }),
      });
    });

    it('should return false when product is out of stock', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: async () => ({
          available: false,
        }),
      });

      const result = await inventoryService.checkAvailability({
        productId: 'product-456',
        quantity: 100, // More than available
      });

      expect(result).toBe(false);
    });

    it('should check availability with variant', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: async () => ({
          available: true,
        }),
      });

      const result = await inventoryService.checkAvailability({
        productId: 'hoodie-123',
        quantity: 1,
        variantId: 'size-l',
      });

      expect(result).toBe(true);

      const callArgs = JSON.parse(mockFetch.mock.calls[0][1].body);
      expect(callArgs.variantId).toBe('size-l');
    });

    it('should return false when API returns error', async () => {
      mockFetch.mockResolvedValue({
        ok: false,
        status: 404,
        json: async () => ({
          message: 'Product not found',
        }),
      });

      const result = await inventoryService.checkAvailability({
        productId: 'nonexistent',
        quantity: 1,
      });

      expect(result).toBe(false);
    });

    it('should return false when network error occurs', async () => {
      mockFetch.mockRejectedValue(new Error('Network timeout'));

      const result = await inventoryService.checkAvailability({
        productId: 'product-123',
        quantity: 1,
      });

      expect(result).toBe(false);
    });

    it('should return false when available field is missing', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: async () => ({}), // No available field
      });

      const result = await inventoryService.checkAvailability({
        productId: 'product-123',
        quantity: 1,
      });

      expect(result).toBe(false);
    });

    it('should console.error on failure', async () => {
      const consoleErrorSpy = vi.spyOn(console, 'error').mockImplementation(() => {});

      mockFetch.mockRejectedValue(new Error('API error'));

      await inventoryService.checkAvailability({
        productId: 'product-123',
        quantity: 1,
      });

      expect(consoleErrorSpy).toHaveBeenCalledWith(
        'Inventory check failed:',
        expect.any(Error)
      );

      consoleErrorSpy.mockRestore();
    });

    it('should check quantity of 1 successfully', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: async () => ({
          available: true,
        }),
      });

      const result = await inventoryService.checkAvailability({
        productId: 'product-single',
        quantity: 1,
      });

      expect(result).toBe(true);
    });

    it('should check large quantity', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: async () => ({
          available: false,
        }),
      });

      const result = await inventoryService.checkAvailability({
        productId: 'product-bulk',
        quantity: 1000,
      });

      expect(result).toBe(false);
    });
  });

  describe('reserveInventory', () => {
    it('should reserve inventory successfully', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: async () => ({}),
      });

      const result = await inventoryService.reserveInventory({
        productId: 'product-123',
        quantity: 2,
        orderId: 'order-456',
      });

      expect(result).toBe(true);
      expect(mockFetch).toHaveBeenCalledWith('/api/inventory/reserve', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          productId: 'product-123',
          quantity: 2,
          orderId: 'order-456',
          variantId: undefined,
        }),
      });
    });

    it('should reserve inventory with variant', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: async () => ({}),
      });

      const result = await inventoryService.reserveInventory({
        productId: 'hoodie-789',
        quantity: 1,
        orderId: 'order-123',
        variantId: 'size-xl',
      });

      expect(result).toBe(true);

      const callArgs = JSON.parse(mockFetch.mock.calls[0][1].body);
      expect(callArgs.variantId).toBe('size-xl');
    });

    it('should return false when insufficient stock', async () => {
      mockFetch.mockResolvedValue({
        ok: false,
        status: 400,
        json: async () => ({
          message: 'Insufficient stock',
        }),
      });

      const result = await inventoryService.reserveInventory({
        productId: 'product-low-stock',
        quantity: 50,
        orderId: 'order-999',
      });

      expect(result).toBe(false);
    });

    it('should return false when product not found', async () => {
      mockFetch.mockResolvedValue({
        ok: false,
        status: 404,
        json: async () => ({
          message: 'Product not found',
        }),
      });

      const result = await inventoryService.reserveInventory({
        productId: 'nonexistent',
        quantity: 1,
        orderId: 'order-123',
      });

      expect(result).toBe(false);
    });

    it('should return false on network error', async () => {
      mockFetch.mockRejectedValue(new Error('Network error'));

      const result = await inventoryService.reserveInventory({
        productId: 'product-123',
        quantity: 1,
        orderId: 'order-456',
      });

      expect(result).toBe(false);
    });

    it('should console.error on failure', async () => {
      const consoleErrorSpy = vi.spyOn(console, 'error').mockImplementation(() => {});

      mockFetch.mockRejectedValue(new Error('Reservation failed'));

      await inventoryService.reserveInventory({
        productId: 'product-123',
        quantity: 1,
        orderId: 'order-456',
      });

      expect(consoleErrorSpy).toHaveBeenCalledWith(
        'Inventory reservation failed:',
        expect.any(Error)
      );

      consoleErrorSpy.mockRestore();
    });

    it('should handle API server errors', async () => {
      mockFetch.mockResolvedValue({
        ok: false,
        status: 500,
        json: async () => ({
          message: 'Internal server error',
        }),
      });

      const result = await inventoryService.reserveInventory({
        productId: 'product-123',
        quantity: 1,
        orderId: 'order-456',
      });

      expect(result).toBe(false);
    });
  });

  describe('confirmReservation', () => {
    it('should confirm reservation successfully', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: async () => ({}),
      });

      const result = await inventoryService.confirmReservation({
        productId: 'product-123',
        quantity: 2,
      });

      expect(result).toBe(true);
      expect(mockFetch).toHaveBeenCalledWith('/api/inventory/confirm', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          productId: 'product-123',
          quantity: 2,
          variantId: undefined,
        }),
      });
    });

    it('should confirm reservation with variant', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: async () => ({}),
      });

      const result = await inventoryService.confirmReservation({
        productId: 'tshirt-456',
        quantity: 3,
        variantId: 'size-m-color-black',
      });

      expect(result).toBe(true);

      const callArgs = JSON.parse(mockFetch.mock.calls[0][1].body);
      expect(callArgs.variantId).toBe('size-m-color-black');
    });

    it('should return false when reservation not found', async () => {
      mockFetch.mockResolvedValue({
        ok: false,
        status: 404,
        json: async () => ({
          message: 'Reservation not found',
        }),
      });

      const result = await inventoryService.confirmReservation({
        productId: 'product-nonexistent',
        quantity: 1,
      });

      expect(result).toBe(false);
    });

    it('should return false on network error', async () => {
      mockFetch.mockRejectedValue(new Error('Network timeout'));

      const result = await inventoryService.confirmReservation({
        productId: 'product-123',
        quantity: 1,
      });

      expect(result).toBe(false);
    });

    it('should console.error on failure', async () => {
      const consoleErrorSpy = vi.spyOn(console, 'error').mockImplementation(() => {});

      mockFetch.mockRejectedValue(new Error('Confirmation error'));

      await inventoryService.confirmReservation({
        productId: 'product-123',
        quantity: 1,
      });

      expect(consoleErrorSpy).toHaveBeenCalledWith(
        'Inventory confirmation failed:',
        expect.any(Error)
      );

      consoleErrorSpy.mockRestore();
    });

    it('should handle API errors gracefully', async () => {
      mockFetch.mockResolvedValue({
        ok: false,
        status: 400,
        json: async () => ({
          message: 'Invalid confirmation',
        }),
      });

      const result = await inventoryService.confirmReservation({
        productId: 'product-123',
        quantity: 1,
      });

      expect(result).toBe(false);
    });
  });

  describe('getLowStockProducts', () => {
    it('should get low stock products successfully', async () => {
      const lowStockProducts = [
        { productId: 'product-1', quantity: 3 },
        { productId: 'product-2', quantity: 1 },
        { productId: 'product-3', quantity: 4 },
      ];

      mockFetch.mockResolvedValue({
        ok: true,
        json: async () => ({
          products: lowStockProducts,
        }),
      });

      const result = await inventoryService.getLowStockProducts(5);

      expect(result).toEqual(lowStockProducts);
      expect(result.length).toBe(3);
      expect(mockFetch).toHaveBeenCalledWith('/api/inventory/low-stock?threshold=5', {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      });
    });

    it('should use default threshold of 5', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: async () => ({
          products: [],
        }),
      });

      await inventoryService.getLowStockProducts();

      expect(mockFetch).toHaveBeenCalledWith('/api/inventory/low-stock?threshold=5', {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      });
    });

    it('should support custom threshold', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: async () => ({
          products: [{ productId: 'product-1', quantity: 2 }],
        }),
      });

      await inventoryService.getLowStockProducts(10);

      expect(mockFetch).toHaveBeenCalledWith('/api/inventory/low-stock?threshold=10', {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      });
    });

    it('should return empty array when no low stock products', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: async () => ({
          products: [],
        }),
      });

      const result = await inventoryService.getLowStockProducts(5);

      expect(result).toEqual([]);
    });

    it('should return empty array when products field is missing', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: async () => ({}), // No products field
      });

      const result = await inventoryService.getLowStockProducts(5);

      expect(result).toEqual([]);
    });

    it('should return empty array on API error', async () => {
      mockFetch.mockResolvedValue({
        ok: false,
        status: 500,
        json: async () => ({
          message: 'Database error',
        }),
      });

      const result = await inventoryService.getLowStockProducts(5);

      expect(result).toEqual([]);
    });

    it('should return empty array on network error', async () => {
      mockFetch.mockRejectedValue(new Error('Network error'));

      const result = await inventoryService.getLowStockProducts(5);

      expect(result).toEqual([]);
    });

    it('should console.error on failure', async () => {
      const consoleErrorSpy = vi.spyOn(console, 'error').mockImplementation(() => {});

      mockFetch.mockRejectedValue(new Error('Fetch error'));

      await inventoryService.getLowStockProducts(5);

      expect(consoleErrorSpy).toHaveBeenCalledWith(
        'Failed to get low stock products:',
        expect.any(Error)
      );

      consoleErrorSpy.mockRestore();
    });

    it('should handle threshold of 0', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: async () => ({
          products: [{ productId: 'out-of-stock', quantity: 0 }],
        }),
      });

      const result = await inventoryService.getLowStockProducts(0);

      expect(result.length).toBe(1);
      expect(result[0].quantity).toBe(0);
    });

    it('should handle large threshold', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: async () => ({
          products: [
            { productId: 'product-1', quantity: 45 },
            { productId: 'product-2', quantity: 30 },
          ],
        }),
      });

      const result = await inventoryService.getLowStockProducts(50);

      expect(result.length).toBe(2);
    });
  });

  describe('inventory flow integration', () => {
    it('should handle complete inventory flow (check, reserve, confirm)', async () => {
      // Step 1: Check availability
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          available: true,
        }),
      });

      const available = await inventoryService.checkAvailability({
        productId: 'hoodie-black-l',
        quantity: 2,
        variantId: 'size-l',
      });

      expect(available).toBe(true);

      // Step 2: Reserve inventory
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({}),
      });

      const reserved = await inventoryService.reserveInventory({
        productId: 'hoodie-black-l',
        quantity: 2,
        orderId: 'order-flow-123',
        variantId: 'size-l',
      });

      expect(reserved).toBe(true);

      // Step 3: Confirm reservation
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({}),
      });

      const confirmed = await inventoryService.confirmReservation({
        productId: 'hoodie-black-l',
        quantity: 2,
        variantId: 'size-l',
      });

      expect(confirmed).toBe(true);
    });
  });
});
