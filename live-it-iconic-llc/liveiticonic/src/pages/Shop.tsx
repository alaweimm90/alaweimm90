import React, { useState, useMemo, useCallback } from 'react';
import { useQuery } from '@tanstack/react-query';
import ProductCard from '@/components/ProductCard';
import FilterPanel from '@/components/FilterPanel';
import { Product, ProductFilter } from '@/types/product';
import SEO from '@/components/SEO';
import Navigation from '@/components/Navigation';
import Footer from '@/components/Footer';
import { Skeleton } from '@/components/ui/skeleton';
import { productService } from '@/services/productService';
import { generateOrganizationSchema } from '@/utils/seo';

const Shop: React.FC = () => {
  const [filters, setFilters] = useState<ProductFilter>({
    sortBy: 'newest',
    page: 1,
    limit: 12,
  });

  const { data: products, isLoading } = useQuery({
    queryKey: ['products', filters],
    queryFn: () => productService.getProducts(filters),
  });

  const { data: categories } = useQuery({
    queryKey: ['categories'],
    queryFn: () => productService.getCategories(),
  });

  const handleFilterChange = useCallback((newFilters: Partial<ProductFilter>) => {
    setFilters(prev => ({ ...prev, ...newFilters, page: 1 }));
  }, []);

  const categoryNames = useMemo(
    () => categories?.map(c => c.name) || [],
    [categories]
  );

  const skeletonArray = useMemo(() => Array.from({ length: 6 }), []);

  return (
    <>
      <SEO
        title="Shop Premium Apparel & Accessories"
        description="Discover our collection of precision-cut apparel inspired by automotive discipline. Premium t-shirts, hoodies, caps, and accessories for discerning individuals."
        canonical="/shop"
        ogImage="/og-shop.jpg"
        ogImageAlt="Live It Iconic Shop Collection"
        keywords="premium apparel, automotive fashion, luxury t-shirts, exclusive clothing, motorsport style"
        ogType="website"
        structuredData={generateOrganizationSchema()}
      />
      <div className="min-h-screen bg-lii-bg text-foreground font-ui overflow-x-hidden relative">
        <Navigation />
        <main className="pt-32 pb-16">
          <div className="container mx-auto px-6">
            {/* Header */}
            <div className="mb-12 text-center">
              <h1 className="text-4xl md:text-5xl font-display font-light tracking-tight text-lii-cloud mb-4">
                Shop Collection
              </h1>
              <p className="text-lii-ash font-ui max-w-2xl mx-auto">
                Explore our premium luxury automotive lifestyle merchandise. Each piece is crafted
                with precision and refined aesthetics.
              </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
              {/* Filters Sidebar */}
              <aside className="md:col-span-1">
                <FilterPanel
                  filters={filters}
                  onChange={handleFilterChange}
                  categories={categoryNames}
                />
              </aside>

              {/* Product Grid */}
              <div className="md:col-span-3">
                {isLoading ? (
                  <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
                    {skeletonArray.map((_, i) => (
                      <div key={i} className="space-y-4">
                        <Skeleton className="aspect-[3/4] rounded-lg bg-lii-charcoal/20" />
                        <Skeleton className="h-4 w-3/4 bg-lii-charcoal/20" />
                        <Skeleton className="h-4 w-1/2 bg-lii-charcoal/20" />
                      </div>
                    ))}
                  </div>
                ) : products && products.length > 0 ? (
                  <>
                    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
                      {products.map(product => (
                        <ProductCard key={product.id} product={product} />
                      ))}
                    </div>
                    {/* Pagination would go here */}
                  </>
                ) : (
                  <div className="flex flex-col items-center justify-center py-16 text-center">
                    <p className="text-lii-ash font-ui text-lg mb-4">No products found</p>
                    <p className="text-lii-ash/70 font-ui text-sm">
                      Try adjusting your filters to see more results.
                    </p>
                  </div>
                )}
              </div>
            </div>
          </div>
        </main>
        <Footer />
      </div>
    </>
  );
};

export default Shop;
