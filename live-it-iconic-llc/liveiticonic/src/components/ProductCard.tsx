import { Heart } from 'lucide-react';
import { memo } from 'react';
import ProductBirdBadge from '@/components/ProductBirdBadge';
import { getBirdType } from '@/components/utils/colorMap';

interface Product {
  id: string;
  name: string;
  category: 'Performance' | 'Recovery' | 'Training' | 'Lifestyle';
  price: number;
  description: string;
}

interface ProductCardProps {
  product: Product;
  index: number;
  isFavorited: boolean;
  onSelect: (product: Product) => void;
  onToggleFavorite: (e: React.MouseEvent<HTMLButtonElement>, productId: string) => void;
}

/**
 * ProductCard component displays a single product with image, details, and interactive controls
 *
 * Renders a clickable product card with category-specific bird badge, favorite toggle button,
 * staggered animation based on index, and product metadata. Supports product selection and
 * favorite/unfavorite actions with proper accessibility labels.
 *
 * @component
 * @param {ProductCardProps} props - Component props
 * @param {Product} props.product - Product data including name, category, price, description
 * @param {number} props.index - Card position for staggered animation delay
 * @param {boolean} props.isFavorited - Current favorited state for visual feedback
 * @param {Function} props.onSelect - Callback fired when product card is clicked
 * @param {Function} props.onToggleFavorite - Callback fired when favorite button is clicked (passes product ID)
 *
 * @example
 * <ProductCard
 *   product={productData}
 *   index={0}
 *   isFavorited={false}
 *   onSelect={(product) => navigate(`/product/${product.id}`)}
 *   onToggleFavorite={(e, id) => toggleFavorite(id)}
 * />
 */
const ProductCard = memo(({
  product,
  index,
  isFavorited,
  onSelect,
  onToggleFavorite,
}: ProductCardProps) => {
  return (
    <article className="group relative" style={{ animationDelay: `${index * 0.1}s` }}>
      <button
        onClick={() => onSelect(product)}
        className="w-full text-left cursor-pointer focus-visible:outline focus-visible:outline-2 focus-visible:outline-lii-gold focus-visible:outline-offset-2 rounded-lg"
        aria-label={`View details for ${product.name} - Premium ${product.category} in the Live It Iconic collection, priced at $${product.price}`}
      >
        {/* Image Placeholder */}
        <div
          className="aspect-[3/4] bg-gradient-to-br from-lii-charcoal/20 to-lii-ink/40 relative overflow-hidden mb-6"
          role="img"
          aria-label={`${product.name} - Premium ${product.category} apparel showcase in dark luxury styling`}
        >
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="text-6xl font-display font-extralight text-lii-gold/30">
              {product.name.charAt(0)}
            </div>
          </div>

          {/* Bird Badge */}
          <div className="absolute top-4 left-4 opacity-50 group-hover:opacity-100 transition-opacity duration-300">
            <ProductBirdBadge birdType={getBirdType(index)} size={28} />
          </div>

          {/* Corner Accent */}
          <div
            className="absolute top-4 right-4 w-12 h-12 border-r border-t border-lii-gold/20"
            aria-hidden="true"
          />
        </div>

        {/* Product Info */}
        <div className="space-y-3">
          <div className="flex items-start justify-between">
            <div>
              <h3 className="text-xl font-display font-light text-lii-cloud mb-1 group-hover:text-lii-gold transition-colors duration-300">
                {product.name}
              </h3>
              <p className="text-xs font-ui uppercase tracking-wider text-lii-cloud">
                {product.category}
              </p>
            </div>
            <button
              onClick={e => onToggleFavorite(e, product.id)}
              aria-label={
                isFavorited
                  ? `Remove ${product.name} from favorites`
                  : `Add ${product.name} to favorites`
              }
              className="min-w-[44px] min-h-[44px] text-lii-gold/40 hover:text-lii-gold transition-colors duration-300 flex items-center justify-center focus-visible:outline focus-visible:outline-2 focus-visible:outline-lii-gold focus-visible:outline-offset-2 rounded-lg flex-shrink-0"
            >
              <Heart className={`w-5 h-5 ${isFavorited ? 'fill-current' : ''}`} />
            </button>
          </div>

          <p className="text-sm font-ui font-light text-lii-cloud leading-relaxed line-clamp-2">
            {product.description}
          </p>

          {/* Price */}
          <div className="pt-3 border-t border-lii-gold/10">
            <span className="text-lg font-ui text-lii-gold font-variant-numeric tabular-nums">
              ${product.price}
            </span>
          </div>
        </div>
      </button>
    </article>
  );
});

ProductCard.displayName = 'ProductCard';

export default ProductCard;
