/**
 * Review Service Tests
 *
 * Comprehensive test suite for product review functionality
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { reviewService } from '../reviewService';

// Mock Supabase
vi.mock('@/lib/supabase', () => ({
  supabase: {
    from: vi.fn(() => ({
      select: vi.fn().mockReturnThis(),
      insert: vi.fn().mockReturnThis(),
      update: vi.fn().mockReturnThis(),
      delete: vi.fn().mockReturnThis(),
      eq: vi.fn().mockReturnThis(),
      single: vi.fn(),
      order: vi.fn().mockReturnThis(),
      range: vi.fn().mockReturnThis(),
    })),
  },
}));

describe('ReviewService', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('createReview', () => {
    it('should create a review with valid data', async () => {
      const mockReview = {
        id: 'review-1',
        product_id: 'product-1',
        customer_id: 'customer-1',
        rating: 5,
        title: 'Great product!',
        text: 'Absolutely love this hoodie',
        verified_purchase: false,
        is_approved: false,
        is_featured: false,
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
      };

      const { supabase } = await import('@/lib/supabase');
      vi.mocked(supabase.from).mockReturnValue({
        insert: vi.fn().mockReturnThis(),
        select: vi.fn().mockReturnThis(),
        single: vi.fn().mockResolvedValue({ data: mockReview, error: null }),
      } as any);

      const result = await reviewService.createReview({
        productId: 'product-1',
        customerId: 'customer-1',
        rating: 5,
        title: 'Great product!',
        text: 'Absolutely love this hoodie',
      });

      expect(result).toBeDefined();
      expect(result.rating).toBe(5);
      expect(result.productId).toBe('product-1');
    });

    it('should throw error for invalid rating', async () => {
      await expect(
        reviewService.createReview({
          productId: 'product-1',
          customerId: 'customer-1',
          rating: 6, // Invalid: > 5
          text: 'Test review',
        })
      ).rejects.toThrow('Rating must be between 1 and 5');
    });

    it('should throw error for rating below 1', async () => {
      await expect(
        reviewService.createReview({
          productId: 'product-1',
          customerId: 'customer-1',
          rating: 0, // Invalid: < 1
          text: 'Test review',
        })
      ).rejects.toThrow('Rating must be between 1 and 5');
    });

    it('should mark as verified purchase when orderId provided and valid', async () => {
      const { supabase } = await import('@/lib/supabase');

      // Mock order item check
      const mockOrderItem = { id: 'item-1', order_id: 'order-1' };
      vi.mocked(supabase.from).mockReturnValueOnce({
        select: vi.fn().mockReturnThis(),
        eq: vi.fn().mockReturnThis(),
        single: vi.fn().mockResolvedValue({ data: mockOrderItem, error: null }),
      } as any);

      // Mock review creation
      const mockReview = {
        id: 'review-1',
        product_id: 'product-1',
        customer_id: 'customer-1',
        rating: 5,
        text: 'Great!',
        verified_purchase: true,
        is_approved: false,
        is_featured: false,
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
      };

      vi.mocked(supabase.from).mockReturnValueOnce({
        insert: vi.fn().mockReturnThis(),
        select: vi.fn().mockReturnThis(),
        single: vi.fn().mockResolvedValue({ data: mockReview, error: null }),
      } as any);

      const result = await reviewService.createReview({
        productId: 'product-1',
        customerId: 'customer-1',
        orderId: 'order-1',
        rating: 5,
        text: 'Great!',
      });

      expect(result.verifiedPurchase).toBe(true);
    });
  });

  describe('getProductReviews', () => {
    it('should fetch approved reviews by default', async () => {
      const mockReviews = [
        {
          id: 'review-1',
          product_id: 'product-1',
          customer_id: 'customer-1',
          rating: 5,
          text: 'Great!',
          is_approved: true,
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString(),
        },
      ];

      const { supabase } = await import('@/lib/supabase');
      vi.mocked(supabase.from).mockReturnValue({
        select: vi.fn().mockReturnThis(),
        eq: vi.fn().mockReturnThis(),
        order: vi.fn().mockReturnThis(),
        range: vi.fn().mockResolvedValue({ data: mockReviews, error: null }),
      } as any);

      const reviews = await reviewService.getProductReviews('product-1');

      expect(reviews).toHaveLength(1);
      expect(reviews[0].isApproved).toBe(true);
    });

    it('should respect pagination parameters', async () => {
      const { supabase } = await import('@/lib/supabase');
      const rangeMock = vi.fn().mockResolvedValue({ data: [], error: null });

      vi.mocked(supabase.from).mockReturnValue({
        select: vi.fn().mockReturnThis(),
        eq: vi.fn().mockReturnThis(),
        order: vi.fn().mockReturnThis(),
        range: rangeMock,
      } as any);

      await reviewService.getProductReviews('product-1', {
        limit: 5,
        offset: 10,
      });

      expect(rangeMock).toHaveBeenCalledWith(10, 14); // offset to offset+limit-1
    });

    it('should fetch unapproved reviews when approvedOnly is false', async () => {
      const mockReviews = [
        {
          id: 'review-1',
          product_id: 'product-1',
          customer_id: 'customer-1',
          rating: 5,
          text: 'Pending review',
          is_approved: false,
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString(),
        },
      ];

      const { supabase } = await import('@/lib/supabase');
      vi.mocked(supabase.from).mockReturnValue({
        select: vi.fn().mockReturnThis(),
        eq: vi.fn().mockReturnThis(),
        order: vi.fn().mockReturnThis(),
        range: vi.fn().mockResolvedValue({ data: mockReviews, error: null }),
      } as any);

      const reviews = await reviewService.getProductReviews('product-1', {
        approvedOnly: false,
      });

      expect(reviews).toHaveLength(1);
      expect(reviews[0].isApproved).toBe(false);
    });
  });

  describe('getReviewStats', () => {
    it('should calculate correct average rating', async () => {
      const mockReviews = [
        { rating: 5 },
        { rating: 4 },
        { rating: 5 },
        { rating: 3 },
        { rating: 5 },
      ];

      const { supabase } = await import('@/lib/supabase');
      vi.mocked(supabase.from).mockReturnValue({
        select: vi.fn().mockReturnThis(),
        eq: vi.fn().mockResolvedValue({ data: mockReviews, error: null }),
      } as any);

      const stats = await reviewService.getReviewStats('product-1');

      expect(stats.totalReviews).toBe(5);
      expect(stats.averageRating).toBe(4.4); // (5+4+5+3+5)/5 = 4.4
      expect(stats.ratingDistribution[5]).toBe(3);
      expect(stats.ratingDistribution[4]).toBe(1);
      expect(stats.ratingDistribution[3]).toBe(1);
    });

    it('should return zero stats for product with no reviews', async () => {
      const { supabase } = await import('@/lib/supabase');
      vi.mocked(supabase.from).mockReturnValue({
        select: vi.fn().mockReturnThis(),
        eq: vi.fn().mockResolvedValue({ data: [], error: null }),
      } as any);

      const stats = await reviewService.getReviewStats('product-1');

      expect(stats.totalReviews).toBe(0);
      expect(stats.averageRating).toBe(0);
      expect(stats.ratingDistribution).toEqual({ 5: 0, 4: 0, 3: 0, 2: 0, 1: 0 });
    });

    it('should calculate correct rating distribution', async () => {
      const mockReviews = [
        { rating: 5 },
        { rating: 5 },
        { rating: 4 },
        { rating: 3 },
        { rating: 1 },
      ];

      const { supabase } = await import('@/lib/supabase');
      vi.mocked(supabase.from).mockReturnValue({
        select: vi.fn().mockReturnThis(),
        eq: vi.fn().mockResolvedValue({ data: mockReviews, error: null }),
      } as any);

      const stats = await reviewService.getReviewStats('product-1');

      expect(stats.ratingDistribution[5]).toBe(2);
      expect(stats.ratingDistribution[4]).toBe(1);
      expect(stats.ratingDistribution[3]).toBe(1);
      expect(stats.ratingDistribution[2]).toBe(0);
      expect(stats.ratingDistribution[1]).toBe(1);
    });
  });

  describe('updateReview', () => {
    it('should update review and require re-moderation', async () => {
      const { supabase } = await import('@/lib/supabase');

      // Mock ownership check
      vi.mocked(supabase.from).mockReturnValueOnce({
        select: vi.fn().mockReturnThis(),
        eq: vi.fn().mockReturnThis(),
        single: vi.fn().mockResolvedValue({ data: { id: 'review-1' }, error: null }),
      } as any);

      // Mock update
      const mockUpdated = {
        id: 'review-1',
        product_id: 'product-1',
        customer_id: 'customer-1',
        rating: 4,
        title: 'Updated title',
        text: 'Updated text',
        is_approved: false, // Should be false after update
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
      };

      vi.mocked(supabase.from).mockReturnValueOnce({
        update: vi.fn().mockReturnThis(),
        eq: vi.fn().mockReturnThis(),
        select: vi.fn().mockReturnThis(),
        single: vi.fn().mockResolvedValue({ data: mockUpdated, error: null }),
      } as any);

      const result = await reviewService.updateReview('review-1', 'customer-1', {
        rating: 4,
        title: 'Updated title',
        text: 'Updated text',
      });

      expect(result.isApproved).toBe(false);
      expect(result.rating).toBe(4);
    });

    it('should throw error for unauthorized update', async () => {
      const { supabase } = await import('@/lib/supabase');

      vi.mocked(supabase.from).mockReturnValue({
        select: vi.fn().mockReturnThis(),
        eq: vi.fn().mockReturnThis(),
        single: vi.fn().mockResolvedValue({ data: null, error: null }),
      } as any);

      await expect(
        reviewService.updateReview('review-1', 'wrong-customer', {
          text: 'Hacked!',
        })
      ).rejects.toThrow('Review not found or unauthorized');
    });

    it('should validate rating on update', async () => {
      const { supabase } = await import('@/lib/supabase');

      vi.mocked(supabase.from).mockReturnValue({
        select: vi.fn().mockReturnThis(),
        eq: vi.fn().mockReturnThis(),
        single: vi.fn().mockResolvedValue({ data: { id: 'review-1' }, error: null }),
      } as any);

      await expect(
        reviewService.updateReview('review-1', 'customer-1', {
          rating: 6, // Invalid
        })
      ).rejects.toThrow('Rating must be between 1 and 5');
    });
  });

  describe('deleteReview', () => {
    it('should delete review successfully', async () => {
      const { supabase } = await import('@/lib/supabase');

      vi.mocked(supabase.from).mockReturnValue({
        delete: vi.fn().mockReturnThis(),
        eq: vi.fn().mockResolvedValue({ error: null }),
      } as any);

      const result = await reviewService.deleteReview('review-1', 'customer-1');

      expect(result).toBe(true);
    });

    it('should return false on delete error', async () => {
      const { supabase } = await import('@/lib/supabase');

      vi.mocked(supabase.from).mockReturnValue({
        delete: vi.fn().mockReturnThis(),
        eq: vi.fn().mockResolvedValue({ error: new Error('Not found') }),
      } as any);

      const result = await reviewService.deleteReview('review-1', 'customer-1');

      expect(result).toBe(false);
    });
  });

  describe('approveReview', () => {
    it('should approve review and set moderation fields', async () => {
      const { supabase } = await import('@/lib/supabase');

      const updateMock = vi.fn().mockReturnThis();
      const eqMock = vi.fn().mockResolvedValue({ error: null });

      vi.mocked(supabase.from).mockReturnValue({
        update: updateMock,
        eq: eqMock,
      } as any);

      const result = await reviewService.approveReview('review-1', 'admin-1');

      expect(result).toBe(true);
      expect(updateMock).toHaveBeenCalledWith(
        expect.objectContaining({
          is_approved: true,
          moderated_by: 'admin-1',
        })
      );
    });
  });

  describe('featureReview', () => {
    it('should mark review as featured', async () => {
      const { supabase } = await import('@/lib/supabase');

      vi.mocked(supabase.from).mockReturnValue({
        update: vi.fn().mockReturnThis(),
        eq: vi.fn().mockResolvedValue({ error: null }),
      } as any);

      const result = await reviewService.featureReview('review-1');

      expect(result).toBe(true);
    });
  });

  describe('getCustomerReviews', () => {
    it('should fetch all reviews by a customer', async () => {
      const mockReviews = [
        {
          id: 'review-1',
          product_id: 'product-1',
          customer_id: 'customer-1',
          rating: 5,
          text: 'Great!',
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString(),
        },
        {
          id: 'review-2',
          product_id: 'product-2',
          customer_id: 'customer-1',
          rating: 4,
          text: 'Good!',
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString(),
        },
      ];

      const { supabase } = await import('@/lib/supabase');
      vi.mocked(supabase.from).mockReturnValue({
        select: vi.fn().mockReturnThis(),
        eq: vi.fn().mockReturnThis(),
        order: vi.fn().mockResolvedValue({ data: mockReviews, error: null }),
      } as any);

      const reviews = await reviewService.getCustomerReviews('customer-1');

      expect(reviews).toHaveLength(2);
      expect(reviews[0].customerId).toBe('customer-1');
      expect(reviews[1].customerId).toBe('customer-1');
    });
  });
});
