import React from 'react';
import { Product } from '@/types/product';
import ProductCard from '@/components/ProductCard';
import { Button } from '@/components/ui/button';
import { ChevronLeft, ChevronRight } from 'lucide-react';

interface RelatedProductsProps {
  products: Product[];
  onProductSelect?: (product: Product) => void;
  onToggleFavorite?: (productId: string) => void;
  title?: string;
  favoritedProducts?: string[];
}

/**
 * RelatedProducts component displays a carousel of related products
 *
 * Shows a scrollable carousel of related products with navigation buttons.
 * Useful for showing complementary items, products from same collection, or items from same category.
 *
 * @component
 * @param {RelatedProductsProps} props - Component props
 * @param {Product[]} props.products - Array of products to display
 * @param {Function} [props.onProductSelect] - Callback when product is selected
 * @param {Function} [props.onToggleFavorite] - Callback when favorite button is clicked
 * @param {string} [props.title] - Carousel title
 * @param {string[]} [props.favoritedProducts] - Array of favorited product IDs
 *
 * @example
 * <RelatedProducts
 *   products={relatedItems}
 *   title="You Might Also Like"
 *   onProductSelect={(product) => navigate(`/product/${product.slug}`)}
 * />
 */
const RelatedProducts: React.FC<RelatedProductsProps> = ({
  products,
  onProductSelect,
  onToggleFavorite,
  title = 'Related Products',
  favoritedProducts = [],
}) => {
  const [scrollPosition, setScrollPosition] = React.useState(0);
  const containerRef = React.useRef<HTMLDivElement>(null);

  if (products.length === 0) {
    return null;
  }

  const scroll = (direction: 'left' | 'right') => {
    if (containerRef.current) {
      const scrollAmount = 400;
      const newPosition =
        direction === 'left'
          ? Math.max(0, scrollPosition - scrollAmount)
          : scrollPosition + scrollAmount;

      containerRef.current.scrollTo({
        left: newPosition,
        behavior: 'smooth',
      });

      setScrollPosition(newPosition);
    }
  };

  const handleScroll = (e: React.UIEvent<HTMLDivElement>) => {
    setScrollPosition((e.target as HTMLDivElement).scrollLeft);
  };

  const canScrollLeft = scrollPosition > 0;
  const canScrollRight = containerRef.current
    ? containerRef.current.scrollLeft + containerRef.current.clientWidth <
      containerRef.current.scrollWidth
    : false;

  return (
    <section className="py-12 border-t border-lii-gold/10">
      <div className="container mx-auto px-6">
        <div className="flex items-center justify-between mb-8">
          <h2 className="text-3xl font-display font-light text-lii-cloud">{title}</h2>
          <div className="flex gap-2">
            <Button
              variant="outline"
              size="icon"
              onClick={() => scroll('left')}
              disabled={!canScrollLeft}
              className="text-lii-ash hover:text-lii-gold disabled:opacity-50"
            >
              <ChevronLeft className="w-5 h-5" />
            </Button>
            <Button
              variant="outline"
              size="icon"
              onClick={() => scroll('right')}
              disabled={!canScrollRight}
              className="text-lii-ash hover:text-lii-gold disabled:opacity-50"
            >
              <ChevronRight className="w-5 h-5" />
            </Button>
          </div>
        </div>

        <div
          ref={containerRef}
          onScroll={handleScroll}
          className="flex gap-6 overflow-x-auto scrollbar-hide"
          style={{ scrollBehavior: 'smooth' }}
        >
          {products.map((product, index) => (
            <div key={product.id} className="flex-shrink-0 w-72">
              <ProductCard
                product={{
                  id: product.id,
                  name: product.name,
                  category: product.category,
                  price: product.price,
                  description: product.description,
                }}
                index={index}
                isFavorited={favoritedProducts.includes(product.id)}
                onSelect={() => onProductSelect?.(product)}
                onToggleFavorite={(e, productId) => {
                  e.stopPropagation();
                  onToggleFavorite?.(productId);
                }}
              />
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default RelatedProducts;
