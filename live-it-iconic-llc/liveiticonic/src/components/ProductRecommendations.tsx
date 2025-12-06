import React from 'react';
import { useQuery } from '@tanstack/react-query';
import { Product } from '@/types/product';
import { recommendationService } from '@/services/recommendationService';
import ProductCard from '@/components/ProductCard';
import { Skeleton } from '@/components/ui/skeleton';
import { useAuth } from '@/contexts/AuthContext';

interface ProductRecommendationsProps {
  productId?: string;
  userId?: string;
  type?: 'recommended' | 'frequently-bought';
}

export const ProductRecommendations: React.FC<ProductRecommendationsProps> = ({
  productId,
  userId,
  type = 'recommended',
}) => {
  const { user } = useAuth();
  const effectiveUserId = userId || user?.id || 'guest';

  const { data: products, isLoading } = useQuery({
    queryKey: ['recommendations', type, productId, effectiveUserId],
    queryFn: () => {
      if (type === 'frequently-bought' && productId) {
        return recommendationService.getFrequentlyBought(productId);
      }
      return recommendationService.getRecommendations(effectiveUserId);
    },
    enabled: type === 'frequently-bought' ? !!productId : true,
  });

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
      <h3 className="text-2xl font-display font-light text-lii-cloud">
        {type === 'frequently-bought' ? 'Customers Also Bought' : 'You May Also Like'}
      </h3>
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
        {products.map(product => (
          <ProductCard key={product.id} product={product} />
        ))}
      </div>
    </div>
  );
};

export default ProductRecommendations;
