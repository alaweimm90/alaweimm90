import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Heart, ShoppingCart } from 'lucide-react';
import { memo } from 'react';
import { getColorHex, getAvailabilityColor } from '@/components/utils/colorMap';

interface Product {
  id: string;
  name: string;
  category: 'Performance' | 'Recovery' | 'Training' | 'Lifestyle';
  price: number;
  originalPrice?: number;
  description: string;
  materials: string;
  features: string[];
  availability: 'In Stock' | 'Limited' | 'Pre-Order' | 'Sold Out';
  colors: string[];
  sizes: string[];
}

interface ProductModalProps {
  product: Product;
  isFavorited: boolean;
  isInCart: boolean;
  onClose: () => void;
  onToggleFavorite: () => void;
  onToggleCart: () => void;
}

const ColorButton = memo(({ color }: { color: string }) => (
  <button
    aria-label={`Select color ${color}`}
    className="flex flex-col items-center gap-2 cursor-pointer group min-h-[44px] min-w-[44px]"
  >
    <div
      className="w-11 h-11 rounded-full border-2 border-lii-gold/20 group-hover:border-lii-gold/40 transition-all duration-300"
      style={{ backgroundColor: getColorHex(color) }}
    />
    <span className="text-xs text-foreground/60 font-ui">{color}</span>
  </button>
));

ColorButton.displayName = 'ColorButton';

const SizeButton = memo(({ size }: { size: string }) => (
  <button
    aria-label={`Select size ${size}`}
    className="px-4 py-3 border border-lii-gold/20 rounded-lg text-sm font-ui text-foreground/70 hover:border-lii-gold/40 hover:bg-lii-gold/5 transition-all duration-300 cursor-pointer min-h-[44px] flex items-center"
  >
    {size}
  </button>
));

SizeButton.displayName = 'SizeButton';

const FeaturesList = memo(({ features }: { features: string[] }) => (
  <div className="grid grid-cols-2 gap-2">
    {features.map((feature, i) => (
      <div key={i} className="flex items-center gap-2 text-sm text-foreground/70 font-ui">
        <div className="w-1 h-1 bg-lii-gold rounded-full" />
        {feature}
      </div>
    ))}
  </div>
));

FeaturesList.displayName = 'FeaturesList';

const ProductModal = memo(({
  product,
  isFavorited,
  isInCart,
  onClose,
  onToggleFavorite,
  onToggleCart,
}: ProductModalProps) => {
  const isSoldOut = product.availability === 'Sold Out';
  const isPreOrder = product.availability === 'Pre-Order';
  const cartButtonText = isInCart ? 'Remove from Cart' : isPreOrder ? 'Pre-Order Now' : 'Add to Cart';
  const favoriteButtonText = isFavorited ? 'Favorited' : 'Add to Favorites';

  return (
    <div className="fixed inset-0 bg-lii-black/95 backdrop-blur-xl z-50 flex items-center justify-center p-6 animate-fade-in">
      <div className="max-w-6xl w-full max-h-[90vh] overflow-y-auto">
        <div className="relative bg-lii-ink/50 rounded-3xl p-8 border border-lii-gold/20">
          {/* Close Button */}
          <Button
            variant="ghost"
            size="icon"
            onClick={onClose}
            aria-label="Close product details"
            className="absolute top-6 right-6 z-10 w-11 h-11 text-foreground hover:text-lii-gold bg-lii-black/50 backdrop-blur-sm hover:bg-lii-gold/10"
          >
            ×
          </Button>

          {/* Product Details Layout */}
          <div className="grid lg:grid-cols-2 gap-12">
            {/* Left Column */}
            <div className="relative">
              <div className="w-full aspect-[4/5] bg-gradient-to-br from-lii-gold/10 to-lii-champagne/5 rounded-2xl flex items-center justify-center mb-6">
                <span className="text-6xl font-display font-light text-lii-gold/60">
                  {product.category}
                </span>
              </div>

              {/* Features Section */}
              <div>
                <h4 className="text-lg font-display text-lii-gold mb-4 tracking-[0.1em]">
                  FEATURES
                </h4>
                <FeaturesList features={product.features} />
              </div>
            </div>

            {/* Right Column */}
            <div className="flex flex-col justify-center space-y-6">
              {/* Header with Price */}
              <div>
                <div className="flex justify-between items-start mb-4">
                  <div>
                    <h2 className="text-3xl font-display font-light text-foreground mb-2">
                      {product.name}
                    </h2>
                    <p className="text-foreground/60 font-ui font-light mb-2">
                      {product.category}
                    </p>
                    <Badge className={`text-xs ${getAvailabilityColor(product.availability)}`}>
                      {product.availability}
                    </Badge>
                  </div>
                  <div className="text-right">
                    {product.originalPrice && (
                      <div className="text-lg text-foreground/40 line-through font-ui">
                        ${product.originalPrice}
                      </div>
                    )}
                    <div className="text-3xl font-ui font-medium text-foreground">
                      ${product.price}
                    </div>
                  </div>
                </div>

                {/* Description */}
                <p className="text-lg text-foreground/80 font-ui font-light leading-relaxed mb-6">
                  {product.description}
                </p>

                {/* Materials */}
                <div className="mb-6">
                  <h4 className="text-sm font-ui font-medium text-foreground/60 uppercase tracking-wide mb-2">
                    Materials
                  </h4>
                  <p className="text-foreground/80 font-ui font-light">
                    {product.materials}
                  </p>
                </div>

                {/* Sizes */}
                <div className="mb-6">
                  <h4 className="text-sm font-ui font-medium text-foreground/60 uppercase tracking-wide mb-3">
                    Sizes Available
                  </h4>
                  <div className="flex flex-wrap gap-2">
                    {product.sizes.map(size => (
                      <SizeButton key={size} size={size} />
                    ))}
                  </div>
                </div>

                {/* Colors */}
                <div className="mb-8">
                  <h4 className="text-sm font-ui font-medium text-foreground/60 uppercase tracking-wide mb-3">
                    Available Colors
                  </h4>
                  <div className="flex gap-3">
                    {product.colors.map(color => (
                      <ColorButton key={color} color={color} />
                    ))}
                  </div>
                </div>
              </div>

              {/* Action Buttons */}
              <div className="flex flex-col sm:flex-row gap-4">
                <Button
                  onClick={onToggleFavorite}
                  variant="outline"
                  className={`flex-1 h-12 font-ui font-medium ${
                    isFavorited
                      ? 'border-lii-gold bg-lii-gold/10 text-lii-gold'
                      : 'bg-transparent border-lii-gold/20 text-foreground hover:border-lii-gold/40'
                  }`}
                >
                  <Heart className={`w-4 h-4 mr-2 ${isFavorited ? 'fill-current' : ''}`} />
                  {favoriteButtonText}
                </Button>

                <Button
                  onClick={onToggleCart}
                  disabled={isSoldOut}
                  className="flex-1 luxury-button h-12 text-lii-black font-ui font-medium tracking-wide"
                >
                  <ShoppingCart className="w-4 h-4 mr-2" />
                  {cartButtonText}
                </Button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
});

ProductModal.displayName = 'ProductModal';

export default ProductModal;
