import { memo } from 'react';
import ProductCard from '@/components/ProductCard';

interface Product {
  id: string;
  name: string;
  category: 'Performance' | 'Recovery' | 'Training' | 'Lifestyle';
  price: number;
  description: string;
}

interface ProductGridProps {
  products: Product[];
  favorites: string[];
  onSelectProduct: (product: Product) => void;
  onToggleFavorite: (e: React.MouseEvent<HTMLButtonElement>, productId: string) => void;
}

const ProductGrid = memo(({
  products,
  favorites,
  onSelectProduct,
  onToggleFavorite,
}: ProductGridProps) => {
  return (
    <div className="grid md:grid-cols-3 gap-8">
      {products.map((product, index) => (
        <ProductCard
          key={product.id}
          product={product}
          index={index}
          isFavorited={favorites.includes(product.id)}
          onSelect={onSelectProduct}
          onToggleFavorite={onToggleFavorite}
        />
      ))}
    </div>
  );
});

ProductGrid.displayName = 'ProductGrid';

export default ProductGrid;
