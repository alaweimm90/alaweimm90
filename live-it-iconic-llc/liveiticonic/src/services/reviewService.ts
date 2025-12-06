/**
 * Review Service
 *
 * Handles product review creation, moderation, and retrieval.
 * Supports verified purchases and admin moderation.
 *
 * @module services/reviewService
 */

import { supabase } from '@/lib/supabase';

export interface Review {
  id: string;
  productId: string;
  customerId: string;
  orderId?: string;
  rating: number; // 1-5
  title?: string;
  text: string;
  images?: string[];
  verifiedPurchase: boolean;
  isFeatured: boolean;
  isApproved: boolean;
  createdAt: Date;
  updatedAt: Date;
}

export interface CreateReviewParams {
  productId: string;
  customerId: string;
  orderId?: string;
  rating: number;
  title?: string;
  text: string;
  images?: string[];
}

export interface ReviewStats {
  averageRating: number;
  totalReviews: number;
  ratingDistribution: {
    5: number;
    4: number;
    3: number;
    2: number;
    1: number;
  };
}

export const reviewService = {
  /**
   * Create a new product review
   */
  async createReview(params: CreateReviewParams): Promise<Review> {
    try {
      // Validate rating
      if (params.rating < 1 || params.rating > 5) {
        throw new Error('Rating must be between 1 and 5');
      }

      // Check if user purchased this product
      let verifiedPurchase = false;
      if (params.orderId) {
        const { data: orderItem } = await supabase
          .from('order_items')
          .select('id, order_id')
          .eq('order_id', params.orderId)
          .eq('product_id', params.productId)
          .single();

        verifiedPurchase = !!orderItem;
      }

      // Create review
      const { data, error } = await supabase
        .from('reviews')
        .insert({
          product_id: params.productId,
          customer_id: params.customerId,
          order_id: params.orderId,
          rating: params.rating,
          title: params.title,
          text: params.text,
          images: params.images || [],
          verified_purchase: verifiedPurchase,
          is_approved: false, // Requires moderation
          is_featured: false,
        })
        .select()
        .single();

      if (error) throw error;

      return this.mapReviewFromDb(data);
    } catch (error) {
      console.error('Failed to create review:', error);
      throw error;
    }
  },

  /**
   * Get reviews for a product
   */
  async getProductReviews(
    productId: string,
    options: {
      approvedOnly?: boolean;
      limit?: number;
      offset?: number;
    } = {}
  ): Promise<Review[]> {
    try {
      const { approvedOnly = true, limit = 10, offset = 0 } = options;

      let query = supabase
        .from('reviews')
        .select('*')
        .eq('product_id', productId)
        .order('created_at', { ascending: false })
        .range(offset, offset + limit - 1);

      if (approvedOnly) {
        query = query.eq('is_approved', true);
      }

      const { data, error } = await query;

      if (error) throw error;

      return data?.map(this.mapReviewFromDb) || [];
    } catch (error) {
      console.error('Failed to get product reviews:', error);
      return [];
    }
  },

  /**
   * Get review statistics for a product
   */
  async getReviewStats(productId: string): Promise<ReviewStats> {
    try {
      const { data, error } = await supabase
        .from('reviews')
        .select('rating')
        .eq('product_id', productId)
        .eq('is_approved', true);

      if (error) throw error;

      if (!data || data.length === 0) {
        return {
          averageRating: 0,
          totalReviews: 0,
          ratingDistribution: { 5: 0, 4: 0, 3: 0, 2: 0, 1: 0 },
        };
      }

      const ratings = data.map((r) => r.rating);
      const totalReviews = ratings.length;
      const averageRating = ratings.reduce((sum, rating) => sum + rating, 0) / totalReviews;

      const ratingDistribution = {
        5: ratings.filter((r) => r === 5).length,
        4: ratings.filter((r) => r === 4).length,
        3: ratings.filter((r) => r === 3).length,
        2: ratings.filter((r) => r === 2).length,
        1: ratings.filter((r) => r === 1).length,
      };

      return {
        averageRating: Math.round(averageRating * 10) / 10, // Round to 1 decimal
        totalReviews,
        ratingDistribution,
      };
    } catch (error) {
      console.error('Failed to get review stats:', error);
      return {
        averageRating: 0,
        totalReviews: 0,
        ratingDistribution: { 5: 0, 4: 0, 3: 0, 2: 0, 1: 0 },
      };
    }
  },

  /**
   * Get reviews by a specific customer
   */
  async getCustomerReviews(customerId: string): Promise<Review[]> {
    try {
      const { data, error } = await supabase
        .from('reviews')
        .select('*')
        .eq('customer_id', customerId)
        .order('created_at', { ascending: false });

      if (error) throw error;

      return data?.map(this.mapReviewFromDb) || [];
    } catch (error) {
      console.error('Failed to get customer reviews:', error);
      return [];
    }
  },

  /**
   * Update a review (customer can edit their own)
   */
  async updateReview(
    reviewId: string,
    customerId: string,
    updates: {
      rating?: number;
      title?: string;
      text?: string;
      images?: string[];
    }
  ): Promise<Review> {
    try {
      // Verify ownership
      const { data: existing } = await supabase
        .from('reviews')
        .select('id')
        .eq('id', reviewId)
        .eq('customer_id', customerId)
        .single();

      if (!existing) {
        throw new Error('Review not found or unauthorized');
      }

      // Validate rating if provided
      if (updates.rating && (updates.rating < 1 || updates.rating > 5)) {
        throw new Error('Rating must be between 1 and 5');
      }

      const { data, error } = await supabase
        .from('reviews')
        .update({
          ...updates,
          updated_at: new Date().toISOString(),
          is_approved: false, // Requires re-moderation after edit
        })
        .eq('id', reviewId)
        .select()
        .single();

      if (error) throw error;

      return this.mapReviewFromDb(data);
    } catch (error) {
      console.error('Failed to update review:', error);
      throw error;
    }
  },

  /**
   * Delete a review
   */
  async deleteReview(reviewId: string, customerId: string): Promise<boolean> {
    try {
      const { error } = await supabase
        .from('reviews')
        .delete()
        .eq('id', reviewId)
        .eq('customer_id', customerId);

      if (error) throw error;

      return true;
    } catch (error) {
      console.error('Failed to delete review:', error);
      return false;
    }
  },

  /**
   * Admin: Approve a review
   */
  async approveReview(reviewId: string, adminId: string): Promise<boolean> {
    try {
      const { error } = await supabase
        .from('reviews')
        .update({
          is_approved: true,
          moderated_at: new Date().toISOString(),
          moderated_by: adminId,
        })
        .eq('id', reviewId);

      if (error) throw error;

      return true;
    } catch (error) {
      console.error('Failed to approve review:', error);
      return false;
    }
  },

  /**
   * Admin: Mark review as featured
   */
  async featureReview(reviewId: string): Promise<boolean> {
    try {
      const { error } = await supabase
        .from('reviews')
        .update({ is_featured: true })
        .eq('id', reviewId);

      if (error) throw error;

      return true;
    } catch (error) {
      console.error('Failed to feature review:', error);
      return false;
    }
  },

  /**
   * Helper: Map database record to Review type
   */
  mapReviewFromDb(data: any): Review {
    return {
      id: data.id,
      productId: data.product_id,
      customerId: data.customer_id,
      orderId: data.order_id,
      rating: data.rating,
      title: data.title,
      text: data.text,
      images: data.images || [],
      verifiedPurchase: data.verified_purchase,
      isFeatured: data.is_featured,
      isApproved: data.is_approved,
      createdAt: new Date(data.created_at),
      updatedAt: new Date(data.updated_at),
    };
  },
};
