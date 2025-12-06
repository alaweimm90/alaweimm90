import React from 'react';
import { useRecentlyViewed } from '@/hooks/useLocalStorage';
import { useQuery } from '@tanstack/react-query';
import { productService } from '@/services/productService';
import ProductCard from '@/components/ProductCard';
import { Skeleton } from '@/components/ui/skeleton';
import { useAuth } from '@/contexts/AuthContext';

/**
 * RecentlyViewed component displays a grid of recently viewed products
 *
 * Fetches and displays products from the user's recently viewed history using localStorage
 * and useAuth context. Displays loading skeletons while products are being fetched.
 * Returns null if no recently viewed products exist.
 *
 * @component
 *
 * @example
 * <RecentlyViewed />
 *
 * @remarks
 * - Uses useRecentlyViewed hook with userId from AuthContext
 * - Fetches product data via productService
 * - Uses React Query for data fetching and caching
 * - Displays skeletons while loading
 * - Filters out null products from API responses
 */
export const RecentlyViewed: React.FC = () => {
  const { user } = useAuth();
  const userId = user?.id || 'guest';
  const { recentlyViewed } = useRecentlyViewed(userId);

  const { data: products, isLoading } = useQuery({
    queryKey: ['recentlyViewed', recentlyViewed],
    queryFn: async () => {
      const productPromises = recentlyViewed.map(id => productService.getProduct(id));
      const results = await Promise.all(productPromises);
      return results.filter((p): p is NonNullable<typeof p> => p !== null);
    },
    enabled: recentlyViewed.length > 0,
  });

  if (recentlyViewed.length === 0) {
    return null;
  }

  if (isLoading) {
    return (
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
        {Array.from({ length: 4 }).map((_, i) => (
          <Skeleton key={i} className="aspect-[3/4]" />
        ))}
      </div>
    );
  }

  if (!products || products.length === 0) {
    return null;
  }

  return (
    <div className="space-y-6">
      <h3 className="text-2xl font-display font-light text-lii-cloud">Recently Viewed</h3>
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
        {products.map(product => (
          <ProductCard key={product.id} product={product} />
        ))}
      </div>
    </div>
  );
};

export default RecentlyViewed;
