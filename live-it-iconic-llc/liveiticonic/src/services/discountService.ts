/**
 * Discount Service
 *
 * Handles discount code validation, application, and redemption tracking.
 * Supports percentage and fixed-amount discounts with usage limits.
 *
 * @module services/discountService
 */

import { supabase } from '@/lib/supabase';

export interface DiscountCode {
  id: string;
  code: string;
  type: 'percentage' | 'fixed_amount';
  value: number; // Percentage (0-100) or fixed amount in cents
  minOrderAmount?: number;
  maxUses?: number;
  timesUsed: number;
  validFrom: Date;
  validUntil?: Date;
  isActive: boolean;
  createdAt: Date;
}

export interface DiscountValidation {
  isValid: boolean;
  error?: string;
  discount?: {
    code: string;
    type: 'percentage' | 'fixed_amount';
    value: number;
    discountAmount: number; // Calculated discount in cents
  };
}

export interface CreateDiscountCodeParams {
  code: string;
  type: 'percentage' | 'fixed_amount';
  value: number;
  minOrderAmount?: number;
  maxUses?: number;
  validFrom?: Date;
  validUntil?: Date;
}

export const discountService = {
  /**
   * Validate and calculate discount for an order
   */
  async validateDiscount(
    code: string,
    orderAmount: number
  ): Promise<DiscountValidation> {
    try {
      // Normalize code (uppercase, trim)
      const normalizedCode = code.trim().toUpperCase();

      // Fetch discount code from database
      const { data, error } = await supabase
        .from('discount_codes')
        .select('*')
        .eq('code', normalizedCode)
        .single();

      if (error || !data) {
        return {
          isValid: false,
          error: 'Invalid discount code',
        };
      }

      const discount = this.mapDiscountFromDb(data);

      // Check if active
      if (!discount.isActive) {
        return {
          isValid: false,
          error: 'This discount code is no longer active',
        };
      }

      // Check valid date range
      const now = new Date();
      if (now < discount.validFrom) {
        return {
          isValid: false,
          error: 'This discount code is not yet valid',
        };
      }

      if (discount.validUntil && now > discount.validUntil) {
        return {
          isValid: false,
          error: 'This discount code has expired',
        };
      }

      // Check usage limit
      if (discount.maxUses && discount.timesUsed >= discount.maxUses) {
        return {
          isValid: false,
          error: 'This discount code has reached its usage limit',
        };
      }

      // Check minimum order amount
      if (discount.minOrderAmount && orderAmount < discount.minOrderAmount) {
        const minFormatted = (discount.minOrderAmount / 100).toFixed(2);
        return {
          isValid: false,
          error: `Minimum order amount of $${minFormatted} required`,
        };
      }

      // Calculate discount amount
      let discountAmount = 0;
      if (discount.type === 'percentage') {
        // Percentage discount (value is 0-100)
        discountAmount = Math.round((orderAmount * discount.value) / 100);
      } else {
        // Fixed amount discount (value is in cents)
        discountAmount = discount.value;
      }

      // Ensure discount doesn't exceed order amount
      if (discountAmount > orderAmount) {
        discountAmount = orderAmount;
      }

      return {
        isValid: true,
        discount: {
          code: discount.code,
          type: discount.type,
          value: discount.value,
          discountAmount,
        },
      };
    } catch (error) {
      console.error('Failed to validate discount:', error);
      return {
        isValid: false,
        error: 'Failed to validate discount code',
      };
    }
  },

  /**
   * Redeem a discount code (increment usage count)
   */
  async redeemDiscount(code: string, orderId: string): Promise<boolean> {
    try {
      const normalizedCode = code.trim().toUpperCase();

      // Increment times_used
      const { error: updateError } = await supabase.rpc('increment_discount_usage', {
        discount_code: normalizedCode,
      });

      if (updateError) throw updateError;

      // Record redemption
      const { error: redemptionError } = await supabase
        .from('discount_redemptions')
        .insert({
          discount_code: normalizedCode,
          order_id: orderId,
          redeemed_at: new Date().toISOString(),
        });

      if (redemptionError) throw redemptionError;

      return true;
    } catch (error) {
      console.error('Failed to redeem discount:', error);
      return false;
    }
  },

  /**
   * Create a new discount code (admin only)
   */
  async createDiscountCode(params: CreateDiscountCodeParams): Promise<DiscountCode> {
    try {
      // Validate value
      if (params.type === 'percentage' && (params.value < 0 || params.value > 100)) {
        throw new Error('Percentage discount must be between 0 and 100');
      }

      if (params.type === 'fixed_amount' && params.value < 0) {
        throw new Error('Fixed amount discount must be positive');
      }

      // Normalize code
      const normalizedCode = params.code.trim().toUpperCase();

      const { data, error } = await supabase
        .from('discount_codes')
        .insert({
          code: normalizedCode,
          type: params.type,
          value: params.value,
          min_order_amount: params.minOrderAmount,
          max_uses: params.maxUses,
          times_used: 0,
          valid_from: params.validFrom?.toISOString() || new Date().toISOString(),
          valid_until: params.validUntil?.toISOString(),
          is_active: true,
        })
        .select()
        .single();

      if (error) throw error;

      return this.mapDiscountFromDb(data);
    } catch (error) {
      console.error('Failed to create discount code:', error);
      throw error;
    }
  },

  /**
   * Get all discount codes (admin only)
   */
  async getAllDiscounts(): Promise<DiscountCode[]> {
    try {
      const { data, error } = await supabase
        .from('discount_codes')
        .select('*')
        .order('created_at', { ascending: false });

      if (error) throw error;

      return data?.map(this.mapDiscountFromDb) || [];
    } catch (error) {
      console.error('Failed to get discount codes:', error);
      return [];
    }
  },

  /**
   * Deactivate a discount code
   */
  async deactivateDiscount(code: string): Promise<boolean> {
    try {
      const normalizedCode = code.trim().toUpperCase();

      const { error } = await supabase
        .from('discount_codes')
        .update({ is_active: false })
        .eq('code', normalizedCode);

      if (error) throw error;

      return true;
    } catch (error) {
      console.error('Failed to deactivate discount:', error);
      return false;
    }
  },

  /**
   * Get discount usage statistics
   */
  async getDiscountStats(code: string): Promise<{
    code: string;
    timesUsed: number;
    maxUses?: number;
    remainingUses?: number;
    totalRevenue: number;
  } | null> {
    try {
      const normalizedCode = code.trim().toUpperCase();

      const { data: discount } = await supabase
        .from('discount_codes')
        .select('*')
        .eq('code', normalizedCode)
        .single();

      if (!discount) return null;

      // Get total revenue from orders using this code
      const { data: orders } = await supabase
        .from('orders')
        .select('total_amount')
        .eq('discount_code', normalizedCode);

      const totalRevenue = orders?.reduce((sum, order) => sum + order.total_amount, 0) || 0;

      const remainingUses = discount.max_uses
        ? discount.max_uses - discount.times_used
        : undefined;

      return {
        code: discount.code,
        timesUsed: discount.times_used,
        maxUses: discount.max_uses,
        remainingUses,
        totalRevenue,
      };
    } catch (error) {
      console.error('Failed to get discount stats:', error);
      return null;
    }
  },

  /**
   * Helper: Map database record to DiscountCode type
   */
  mapDiscountFromDb(data: any): DiscountCode {
    return {
      id: data.id,
      code: data.code,
      type: data.type,
      value: data.value,
      minOrderAmount: data.min_order_amount,
      maxUses: data.max_uses,
      timesUsed: data.times_used,
      validFrom: new Date(data.valid_from),
      validUntil: data.valid_until ? new Date(data.valid_until) : undefined,
      isActive: data.is_active,
      createdAt: new Date(data.created_at),
    };
  },
};
