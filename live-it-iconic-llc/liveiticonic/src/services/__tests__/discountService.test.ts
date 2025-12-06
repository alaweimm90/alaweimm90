/**
 * Discount Service Tests
 *
 * Comprehensive test suite for discount code functionality
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { discountService } from '../discountService';

// Mock Supabase
vi.mock('@/lib/supabase', () => ({
  supabase: {
    from: vi.fn(() => ({
      select: vi.fn().mockReturnThis(),
      insert: vi.fn().mockReturnThis(),
      update: vi.fn().mockReturnThis(),
      eq: vi.fn().mockReturnThis(),
      single: vi.fn(),
      order: vi.fn().mockReturnThis(),
    })),
    rpc: vi.fn(),
  },
}));

describe('DiscountService', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('validateDiscount', () => {
    it('should validate a valid percentage discount', async () => {
      const mockDiscount = {
        id: 'discount-1',
        code: 'SAVE10',
        type: 'percentage',
        value: 10,
        is_active: true,
        valid_from: new Date('2024-01-01').toISOString(),
        valid_until: new Date('2025-12-31').toISOString(),
        times_used: 5,
        max_uses: 100,
        created_at: new Date().toISOString(),
      };

      const { supabase } = await import('@/lib/supabase');
      vi.mocked(supabase.from).mockReturnValue({
        select: vi.fn().mockReturnThis(),
        eq: vi.fn().mockReturnThis(),
        single: vi.fn().mockResolvedValue({ data: mockDiscount, error: null }),
      } as any);

      const result = await discountService.validateDiscount('SAVE10', 10000); // $100 order

      expect(result.isValid).toBe(true);
      expect(result.discount).toBeDefined();
      expect(result.discount?.discountAmount).toBe(1000); // 10% of $100 = $10
      expect(result.discount?.type).toBe('percentage');
    });

    it('should validate a valid fixed amount discount', async () => {
      const mockDiscount = {
        id: 'discount-1',
        code: 'SAVE500',
        type: 'fixed_amount',
        value: 500, // $5.00
        is_active: true,
        valid_from: new Date('2024-01-01').toISOString(),
        times_used: 0,
        created_at: new Date().toISOString(),
      };

      const { supabase } = await import('@/lib/supabase');
      vi.mocked(supabase.from).mockReturnValue({
        select: vi.fn().mockReturnThis(),
        eq: vi.fn().mockReturnThis(),
        single: vi.fn().mockResolvedValue({ data: mockDiscount, error: null }),
      } as any);

      const result = await discountService.validateDiscount('save500', 10000);

      expect(result.isValid).toBe(true);
      expect(result.discount?.discountAmount).toBe(500); // Fixed $5
    });

    it('should normalize discount codes to uppercase', async () => {
      const mockDiscount = {
        code: 'SAVE10',
        type: 'percentage',
        value: 10,
        is_active: true,
        valid_from: new Date('2024-01-01').toISOString(),
        times_used: 0,
        created_at: new Date().toISOString(),
      };

      const { supabase } = await import('@/lib/supabase');
      const eqMock = vi.fn().mockReturnThis();

      vi.mocked(supabase.from).mockReturnValue({
        select: vi.fn().mockReturnThis(),
        eq: eqMock,
        single: vi.fn().mockResolvedValue({ data: mockDiscount, error: null }),
      } as any);

      await discountService.validateDiscount('  save10  ', 10000); // lowercase with spaces

      expect(eqMock).toHaveBeenCalledWith('code', 'SAVE10');
    });

    it('should reject invalid discount code', async () => {
      const { supabase } = await import('@/lib/supabase');
      vi.mocked(supabase.from).mockReturnValue({
        select: vi.fn().mockReturnThis(),
        eq: vi.fn().mockReturnThis(),
        single: vi.fn().mockResolvedValue({ data: null, error: new Error('Not found') }),
      } as any);

      const result = await discountService.validateDiscount('INVALID', 10000);

      expect(result.isValid).toBe(false);
      expect(result.error).toBe('Invalid discount code');
    });

    it('should reject inactive discount', async () => {
      const mockDiscount = {
        code: 'EXPIRED',
        type: 'percentage',
        value: 20,
        is_active: false, // Inactive
        valid_from: new Date('2024-01-01').toISOString(),
        times_used: 0,
        created_at: new Date().toISOString(),
      };

      const { supabase } = await import('@/lib/supabase');
      vi.mocked(supabase.from).mockReturnValue({
        select: vi.fn().mockReturnThis(),
        eq: vi.fn().mockReturnThis(),
        single: vi.fn().mockResolvedValue({ data: mockDiscount, error: null }),
      } as any);

      const result = await discountService.validateDiscount('EXPIRED', 10000);

      expect(result.isValid).toBe(false);
      expect(result.error).toBe('This discount code is no longer active');
    });

    it('should reject not-yet-valid discount', async () => {
      const mockDiscount = {
        code: 'FUTURE',
        type: 'percentage',
        value: 15,
        is_active: true,
        valid_from: new Date('2099-01-01').toISOString(), // Future date
        times_used: 0,
        created_at: new Date().toISOString(),
      };

      const { supabase } = await import('@/lib/supabase');
      vi.mocked(supabase.from).mockReturnValue({
        select: vi.fn().mockReturnThis(),
        eq: vi.fn().mockReturnThis(),
        single: vi.fn().mockResolvedValue({ data: mockDiscount, error: null }),
      } as any);

      const result = await discountService.validateDiscount('FUTURE', 10000);

      expect(result.isValid).toBe(false);
      expect(result.error).toBe('This discount code is not yet valid');
    });

    it('should reject expired discount', async () => {
      const mockDiscount = {
        code: 'OLDCODE',
        type: 'percentage',
        value: 25,
        is_active: true,
        valid_from: new Date('2020-01-01').toISOString(),
        valid_until: new Date('2020-12-31').toISOString(), // Expired
        times_used: 0,
        created_at: new Date().toISOString(),
      };

      const { supabase } = await import('@/lib/supabase');
      vi.mocked(supabase.from).mockReturnValue({
        select: vi.fn().mockReturnThis(),
        eq: vi.fn().mockReturnThis(),
        single: vi.fn().mockResolvedValue({ data: mockDiscount, error: null }),
      } as any);

      const result = await discountService.validateDiscount('OLDCODE', 10000);

      expect(result.isValid).toBe(false);
      expect(result.error).toBe('This discount code has expired');
    });

    it('should reject discount at usage limit', async () => {
      const mockDiscount = {
        code: 'MAXED',
        type: 'percentage',
        value: 20,
        is_active: true,
        valid_from: new Date('2024-01-01').toISOString(),
        max_uses: 10,
        times_used: 10, // At limit
        created_at: new Date().toISOString(),
      };

      const { supabase } = await import('@/lib/supabase');
      vi.mocked(supabase.from).mockReturnValue({
        select: vi.fn().mockReturnThis(),
        eq: vi.fn().mockReturnThis(),
        single: vi.fn().mockResolvedValue({ data: mockDiscount, error: null }),
      } as any);

      const result = await discountService.validateDiscount('MAXED', 10000);

      expect(result.isValid).toBe(false);
      expect(result.error).toBe('This discount code has reached its usage limit');
    });

    it('should reject order below minimum amount', async () => {
      const mockDiscount = {
        code: 'BIG50',
        type: 'percentage',
        value: 50,
        is_active: true,
        valid_from: new Date('2024-01-01').toISOString(),
        min_order_amount: 10000, // $100 minimum
        times_used: 0,
        created_at: new Date().toISOString(),
      };

      const { supabase } = await import('@/lib/supabase');
      vi.mocked(supabase.from).mockReturnValue({
        select: vi.fn().mockReturnThis(),
        eq: vi.fn().mockReturnThis(),
        single: vi.fn().mockResolvedValue({ data: mockDiscount, error: null }),
      } as any);

      const result = await discountService.validateDiscount('BIG50', 5000); // Only $50

      expect(result.isValid).toBe(false);
      expect(result.error).toContain('Minimum order amount of $100.00 required');
    });

    it('should cap discount at order amount', async () => {
      const mockDiscount = {
        code: 'HUGE100',
        type: 'fixed_amount',
        value: 10000, // $100 off
        is_active: true,
        valid_from: new Date('2024-01-01').toISOString(),
        times_used: 0,
        created_at: new Date().toISOString(),
      };

      const { supabase } = await import('@/lib/supabase');
      vi.mocked(supabase.from).mockReturnValue({
        select: vi.fn().mockReturnThis(),
        eq: vi.fn().mockReturnThis(),
        single: vi.fn().mockResolvedValue({ data: mockDiscount, error: null }),
      } as any);

      const result = await discountService.validateDiscount('HUGE100', 5000); // Only $50 order

      expect(result.isValid).toBe(true);
      expect(result.discount?.discountAmount).toBe(5000); // Capped at order amount
    });
  });

  describe('redeemDiscount', () => {
    it('should redeem discount and record redemption', async () => {
      const { supabase } = await import('@/lib/supabase');

      vi.mocked(supabase.rpc).mockResolvedValue({ error: null });
      vi.mocked(supabase.from).mockReturnValue({
        insert: vi.fn().mockResolvedValue({ error: null }),
      } as any);

      const result = await discountService.redeemDiscount('SAVE10', 'order-1');

      expect(result).toBe(true);
      expect(supabase.rpc).toHaveBeenCalledWith('increment_discount_usage', {
        discount_code: 'SAVE10',
      });
    });

    it('should return false on redemption error', async () => {
      const { supabase } = await import('@/lib/supabase');

      vi.mocked(supabase.rpc).mockResolvedValue({ error: new Error('Failed') });

      const result = await discountService.redeemDiscount('SAVE10', 'order-1');

      expect(result).toBe(false);
    });
  });

  describe('createDiscountCode', () => {
    it('should create a percentage discount', async () => {
      const mockCreated = {
        id: 'discount-1',
        code: 'NEW10',
        type: 'percentage',
        value: 10,
        is_active: true,
        valid_from: new Date().toISOString(),
        times_used: 0,
        created_at: new Date().toISOString(),
      };

      const { supabase } = await import('@/lib/supabase');
      vi.mocked(supabase.from).mockReturnValue({
        insert: vi.fn().mockReturnThis(),
        select: vi.fn().mockReturnThis(),
        single: vi.fn().mockResolvedValue({ data: mockCreated, error: null }),
      } as any);

      const result = await discountService.createDiscountCode({
        code: 'new10',
        type: 'percentage',
        value: 10,
      });

      expect(result.code).toBe('NEW10'); // Normalized
      expect(result.type).toBe('percentage');
      expect(result.value).toBe(10);
    });

    it('should throw error for invalid percentage', async () => {
      await expect(
        discountService.createDiscountCode({
          code: 'INVALID',
          type: 'percentage',
          value: 150, // > 100%
        })
      ).rejects.toThrow('Percentage discount must be between 0 and 100');
    });

    it('should throw error for negative fixed amount', async () => {
      await expect(
        discountService.createDiscountCode({
          code: 'INVALID',
          type: 'fixed_amount',
          value: -100,
        })
      ).rejects.toThrow('Fixed amount discount must be positive');
    });

    it('should create discount with custom validity period', async () => {
      const validFrom = new Date('2025-01-01');
      const validUntil = new Date('2025-12-31');

      const mockCreated = {
        id: 'discount-1',
        code: 'NEWYEAR',
        type: 'percentage',
        value: 25,
        is_active: true,
        valid_from: validFrom.toISOString(),
        valid_until: validUntil.toISOString(),
        times_used: 0,
        created_at: new Date().toISOString(),
      };

      const { supabase } = await import('@/lib/supabase');
      vi.mocked(supabase.from).mockReturnValue({
        insert: vi.fn().mockReturnThis(),
        select: vi.fn().mockReturnThis(),
        single: vi.fn().mockResolvedValue({ data: mockCreated, error: null }),
      } as any);

      const result = await discountService.createDiscountCode({
        code: 'NEWYEAR',
        type: 'percentage',
        value: 25,
        validFrom,
        validUntil,
      });

      expect(result.validFrom).toBeInstanceOf(Date);
      expect(result.validUntil).toBeInstanceOf(Date);
    });
  });

  describe('getAllDiscounts', () => {
    it('should fetch all discount codes', async () => {
      const mockDiscounts = [
        {
          id: 'discount-1',
          code: 'CODE1',
          type: 'percentage',
          value: 10,
          is_active: true,
          valid_from: new Date().toISOString(),
          times_used: 5,
          created_at: new Date().toISOString(),
        },
        {
          id: 'discount-2',
          code: 'CODE2',
          type: 'fixed_amount',
          value: 500,
          is_active: true,
          valid_from: new Date().toISOString(),
          times_used: 2,
          created_at: new Date().toISOString(),
        },
      ];

      const { supabase } = await import('@/lib/supabase');
      vi.mocked(supabase.from).mockReturnValue({
        select: vi.fn().mockReturnThis(),
        order: vi.fn().mockResolvedValue({ data: mockDiscounts, error: null }),
      } as any);

      const discounts = await discountService.getAllDiscounts();

      expect(discounts).toHaveLength(2);
      expect(discounts[0].code).toBe('CODE1');
      expect(discounts[1].code).toBe('CODE2');
    });
  });

  describe('deactivateDiscount', () => {
    it('should deactivate a discount code', async () => {
      const { supabase } = await import('@/lib/supabase');

      vi.mocked(supabase.from).mockReturnValue({
        update: vi.fn().mockReturnThis(),
        eq: vi.fn().mockResolvedValue({ error: null }),
      } as any);

      const result = await discountService.deactivateDiscount('OLDCODE');

      expect(result).toBe(true);
    });
  });

  describe('getDiscountStats', () => {
    it('should calculate discount statistics', async () => {
      const mockDiscount = {
        code: 'SAVE10',
        times_used: 15,
        max_uses: 100,
        is_active: true,
        valid_from: new Date().toISOString(),
        created_at: new Date().toISOString(),
      };

      const mockOrders = [
        { total_amount: 9000 }, // $90
        { total_amount: 8500 }, // $85
        { total_amount: 10000 }, // $100
      ];

      const { supabase } = await import('@/lib/supabase');

      vi.mocked(supabase.from).mockReturnValueOnce({
        select: vi.fn().mockReturnThis(),
        eq: vi.fn().mockReturnThis(),
        single: vi.fn().mockResolvedValue({ data: mockDiscount, error: null }),
      } as any);

      vi.mocked(supabase.from).mockReturnValueOnce({
        select: vi.fn().mockReturnThis(),
        eq: vi.fn().mockResolvedValue({ data: mockOrders, error: null }),
      } as any);

      const stats = await discountService.getDiscountStats('SAVE10');

      expect(stats).toBeDefined();
      expect(stats?.code).toBe('SAVE10');
      expect(stats?.timesUsed).toBe(15);
      expect(stats?.maxUses).toBe(100);
      expect(stats?.remainingUses).toBe(85);
      expect(stats?.totalRevenue).toBe(27500); // $275 total
    });

    it('should return null for invalid code', async () => {
      const { supabase } = await import('@/lib/supabase');

      vi.mocked(supabase.from).mockReturnValue({
        select: vi.fn().mockReturnThis(),
        eq: vi.fn().mockReturnThis(),
        single: vi.fn().mockResolvedValue({ data: null, error: null }),
      } as any);

      const stats = await discountService.getDiscountStats('INVALID');

      expect(stats).toBeNull();
    });
  });
});
