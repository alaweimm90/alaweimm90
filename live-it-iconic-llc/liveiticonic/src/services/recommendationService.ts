import { Product } from '@/types/product';

export const recommendationService = {
  async getRecommendations(userId: string, limit: number = 5): Promise<Product[]> {
    try {
      const response = await fetch(`/api/recommendations/${userId}?limit=${limit}`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        return [];
      }

      const data = await response.json();
      return data.products || [];
    } catch (error) {
      console.error('Failed to get recommendations:', error);
      return [];
    }
  },

  async getFrequentlyBought(productId: string): Promise<Product[]> {
    try {
      const response = await fetch(`/api/recommendations/frequently-bought/${productId}`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        return [];
      }

      const data = await response.json();
      return data.products || [];
    } catch (error) {
      console.error('Failed to get frequently bought products:', error);
      return [];
    }
  },

  async getRecentlyViewed(userId: string): Promise<Product[]> {
    try {
      // Get from localStorage for now
      const recentlyViewed = localStorage.getItem(`recently_viewed_${userId}`);
      if (!recentlyViewed) {
        return [];
      }

      const productIds = JSON.parse(recentlyViewed);
      // Fetch products by IDs
      const products = await Promise.all(
        productIds.map(async (id: string) => {
          const response = await fetch(`/api/products/${id}`);
          if (response.ok) {
            return await response.json();
          }
          return null;
        })
      );

      return products.filter((p): p is Product => p !== null);
    } catch (error) {
      console.error('Failed to get recently viewed:', error);
      return [];
    }
  },
};
