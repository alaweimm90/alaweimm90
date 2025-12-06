import { Button } from '@/components/ui/button';
import ProductGrid from '@/components/ProductGrid';
import ProductModal from '@/components/ProductModal';
import { useState } from 'react';

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
  rating: number;
  reviews: number;
  colors: string[];
  sizes: string[];
}

/**
 * ProductShowcase component displays a grid of products with filtering and detailed modal view
 *
 * Showcases a product catalog with filtering by category (Performance, Recovery, Training, Lifestyle).
 * Each product can be clicked to open a detailed modal view with specifications, pricing, and
 * availability information. Dynamically filters product grid based on selected category.
 *
 * @component
 *
 * @example
 * <ProductShowcase />
 *
 * @remarks
 * - Product catalog is hardcoded within the component
 * - Category filter updates grid in real-time
 * - Modal displays full product details on click
 * - Categories: Performance, Recovery, Training, Lifestyle
 * - Each product includes pricing, rating, colors, and sizes
 */
const productCatalog: Product[] = [
  {
    id: 'velocity-tee',
    name: 'Velocity Performance Tee',
    category: 'Performance',
    price: 85,
    originalPrice: 110,
    description:
      'Ultra-lightweight performance tee engineered with moisture-wicking technology and anti-odor treatment.',
    materials: '92% Merino Wool, 8% Lycra',
    features: [
      'Moisture-wicking',
      'Anti-odor treatment',
      'UPF 50+ protection',
      'Seamless construction',
    ],
    availability: 'In Stock',
    rating: 4.8,
    reviews: 127,
    colors: ['Black', 'Charcoal', 'Navy', 'Stone'],
    sizes: ['XS', 'S', 'M', 'L', 'XL', 'XXL'],
  },
  {
    id: 'apex-shorts',
    name: 'Apex Training Shorts',
    category: 'Training',
    price: 125,
    originalPrice: 155,
    description: 'Professional-grade training shorts with compression liner and secure zip pocket.',
    materials: '88% Polyester, 12% Spandex',
    features: ['7-inch inseam', 'Compression liner', 'Secure zip pocket', 'Gusseted crotch'],
    availability: 'In Stock',
    rating: 4.9,
    reviews: 201,
    colors: ['Black', 'Navy', 'Charcoal'],
    sizes: ['S', 'M', 'L', 'XL'],
  },
  {
    id: 'heritage-hoodie',
    name: 'Heritage Recovery Hoodie',
    category: 'Recovery',
    price: 185,
    originalPrice: 220,
    description:
      'Premium crew sweatshirt crafted from organic cotton blend with reinforced stitching.',
    materials: '85% Organic Cotton, 15% Recycled Polyester',
    features: ['Organic cotton blend', 'Pre-shrunk', 'Reinforced stitching', 'Kangaroo pocket'],
    availability: 'In Stock',
    rating: 4.6,
    reviews: 156,
    colors: ['Ash', 'Charcoal', 'Stone'],
    sizes: ['XS', 'S', 'M', 'L', 'XL'],
  },
  {
    id: 'velocity-joggers',
    name: 'Velocity Track Joggers',
    category: 'Training',
    price: 145,
    originalPrice: 175,
    description: 'Modern joggers with tapered fit and side zip pockets for secure storage.',
    materials: '90% Cotton, 10% Elastane',
    features: ['Tapered fit', 'Side zip pockets', 'Elastic waistband', 'Reflective details'],
    availability: 'In Stock',
    rating: 4.7,
    reviews: 134,
    colors: ['Black', 'Charcoal', 'Navy'],
    sizes: ['S', 'M', 'L', 'XL', 'XXL'],
  },
  {
    id: 'heritage-jacket',
    name: 'Heritage Track Jacket',
    category: 'Lifestyle',
    price: 395,
    originalPrice: 450,
    description: 'Luxury track jacket with water-resistant coating and premium YKK zippers.',
    materials: '100% Italian Nylon, DWR coating',
    features: [
      'Water-resistant coating',
      'Premium YKK zippers',
      'Articulated sleeves',
      'Interior phone pocket',
    ],
    availability: 'Limited',
    rating: 4.7,
    reviews: 89,
    colors: ['Obsidian', 'Navy'],
    sizes: ['S', 'M', 'L', 'XL'],
  },
  {
    id: 'apex-leggings',
    name: 'Apex Performance Leggings',
    category: 'Performance',
    price: 135,
    originalPrice: 165,
    description:
      'High-performance leggings with 4-way stretch and flatlock seams for zero chafing.',
    materials: '87% Recycled Polyester, 13% Elastane',
    features: ['4-way stretch', 'Flatlock seams', 'High-rise waistband', 'Side phone pocket'],
    availability: 'In Stock',
    rating: 4.8,
    reviews: 243,
    colors: ['Black', 'Charcoal', 'Navy'],
    sizes: ['XS', 'S', 'M', 'L', 'XL'],
  },
];

const ProductShowcase = () => {
  const [selectedProduct, setSelectedProduct] = useState<Product | null>(null);
  const [cart, setCart] = useState<string[]>([]);
  const [favorites, setFavorites] = useState<string[]>([]);

  const filteredProducts = productCatalog.slice(0, 3);

  const handleToggleFavorite = (e: React.MouseEvent<HTMLButtonElement>, productId: string) => {
    e.stopPropagation();
    setFavorites(prev =>
      prev.includes(productId) ? prev.filter(id => id !== productId) : [...prev, productId]
    );
  };

  const handleToggleCart = () => {
    if (!selectedProduct) return;
    setCart(prev =>
      prev.includes(selectedProduct.id)
        ? prev.filter(id => id !== selectedProduct.id)
        : [...prev, selectedProduct.id]
    );
  };

  return (
    <section id="collection" className="py-16 md:py-20 bg-lii-ink relative overflow-hidden">
      {/* Background Orb */}
      <div className="absolute inset-0 opacity-10">
        <div className="absolute top-1/2 right-1/3 w-64 h-64 bg-lii-gold rounded-full blur-[120px] animate-[luxuryPulse_10s_ease-in-out_infinite]" />
      </div>

      <div className="container mx-auto px-6 relative z-10">
        {/* Header */}
        <div className="text-center mb-10 max-w-4xl mx-auto">
          <div className="backdrop-blur-xl bg-lii-bg/40 border border-lii-gold/20 rounded-2xl p-6 md:p-8 shadow-[0_8px_32px_rgba(193,160,96,0.1)]">
            <h2 className="text-3xl sm:text-4xl md:text-5xl font-display tracking-tight mb-3">
              <span className="bg-gradient-to-r from-lii-cloud via-lii-gold to-lii-cloud bg-clip-text text-transparent">
                The Collection
              </span>
            </h2>
            <div className="w-16 h-px bg-gradient-to-r from-transparent via-lii-gold to-transparent mx-auto mb-3" />
            <p className="text-base sm:text-lg font-ui text-lii-ash">
              Engineered for movement. Refined for life.
            </p>
          </div>
        </div>

        {/* Product Grid */}
        <div className="max-w-6xl mx-auto">
          <ProductGrid
            products={filteredProducts}
            favorites={favorites}
            onSelectProduct={setSelectedProduct}
            onToggleFavorite={handleToggleFavorite}
          />

          {/* View All CTA */}
          <div className="text-center mt-16">
            <Button variant="secondary" size="lg" className="font-ui font-medium">
              View Full Collection
            </Button>
          </div>
        </div>
      </div>

      {/* Product Detail Modal */}
      {selectedProduct && (
        <ProductModal
          product={selectedProduct}
          isFavorited={favorites.includes(selectedProduct.id)}
          isInCart={cart.includes(selectedProduct.id)}
          onClose={() => setSelectedProduct(null)}
          onToggleFavorite={() => handleToggleFavorite({} as React.MouseEvent<HTMLButtonElement>, selectedProduct.id)}
          onToggleCart={handleToggleCart}
        />
      )}
    </section>
  );
};

export default ProductShowcase;
