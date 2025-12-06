import type { Meta, StoryObj } from '@storybook/react';
import { ProductCard } from './ProductCard';
import { useState } from 'react';

const meta: Meta<typeof ProductCard> = {
  title: 'Components/ProductCard',
  component: ProductCard,
  tags: ['autodocs'],
  parameters: {
    backgrounds: {
      default: 'dark',
    },
  },
  decorators: [
    (Story) => (
      <div className="max-w-sm">
        <Story />
      </div>
    ),
  ],
};

export default meta;
type Story = StoryObj<typeof ProductCard>;

const mockProduct = {
  id: 'prod-001',
  name: 'Heritage Polo',
  category: 'Lifestyle' as const,
  price: 89.99,
  description: 'Premium heritage polo shirt with luxury craftsmanship',
};

export const Default: Story = {
  render: () => {
    const [isFavorited, setIsFavorited] = useState(false);
    
    return (
      <ProductCard
        product={mockProduct}
        index={0}
        isFavorited={isFavorited}
        onSelect={(product) => console.log('Selected:', product)}
        onToggleFavorite={(e, id) => {
          e.stopPropagation();
          setIsFavorited(!isFavorited);
        }}
      />
    );
  },
};

export const Favorited: Story = {
  render: () => (
    <ProductCard
      product={mockProduct}
      index={0}
      isFavorited={true}
      onSelect={(product) => console.log('Selected:', product)}
      onToggleFavorite={() => console.log('Toggle favorite')}
    />
  ),
};

export const Performance: Story = {
  render: () => {
    const perfProduct = {
      ...mockProduct,
      name: 'Performance Tank',
      category: 'Performance' as const,
    };
    
    return (
      <ProductCard
        product={perfProduct}
        index={0}
        isFavorited={false}
        onSelect={(product) => console.log('Selected:', product)}
        onToggleFavorite={() => console.log('Toggle favorite')}
      />
    );
  },
};

export const Grid: Story = {
  render: () => {
    const products = [
      { ...mockProduct, id: 'p1', name: 'Heritage Polo' },
      { ...mockProduct, id: 'p2', name: 'Performance Tank', category: 'Performance' as const },
      { ...mockProduct, id: 'p3', name: 'Recovery Hoodie', category: 'Recovery' as const },
    ];
    
    return (
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
        {products.map((product, index) => (
          <ProductCard
            key={product.id}
            product={product}
            index={index}
            isFavorited={false}
            onSelect={(p) => console.log('Selected:', p)}
            onToggleFavorite={() => console.log('Toggle favorite')}
          />
        ))}
      </div>
    );
  },
};
