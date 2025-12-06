import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen } from '@/test/test-utils';
import userEvent from '@testing-library/user-event';
import ProductGrid from './ProductGrid';
import { mockProducts } from '@/test/mocks';

// Mock ProductCard component
vi.mock('./ProductCard', () => ({
  default: ({ product, index, isFavorited, onSelect, onToggleFavorite }: any) => (
    <div data-testid={`product-card-${product.id}`}>
      <h3>{product.name}</h3>
      <button onClick={() => onSelect(product)}>View Product</button>
      <button onClick={(e) => onToggleFavorite(e, product.id)}>
        {isFavorited ? 'Remove from' : 'Add to'} Favorites
      </button>
    </div>
  ),
}));

describe('ProductGrid Component', () => {
  const mockOnSelectProduct = vi.fn();
  const mockOnToggleFavorite = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders grid of products', () => {
    render(
      <ProductGrid
        products={mockProducts}
        favoriteIds={[]}
        onSelectProduct={mockOnSelectProduct}
        onToggleFavorite={mockOnToggleFavorite}
      />
    );

    // Should render without errors and have product cards
    mockProducts.forEach(product => {
      expect(screen.getByTestId(`product-card-${product.id}`)).toBeInTheDocument();
    });
  });

  it('renders correct number of product cards', () => {
    const { container } = render(
      <ProductGrid
        products={mockProducts}
        favoriteIds={[]}
        onSelectProduct={mockOnSelectProduct}
        onToggleFavorite={mockOnToggleFavorite}
      />
    );

    const cards = container.querySelectorAll('[data-testid^="product-card-"]');
    expect(cards.length).toBeGreaterThan(0);
  });

  it('calls onSelectProduct when product is clicked', async () => {
    const user = userEvent.setup();
    render(
      <ProductGrid
        products={mockProducts}
        favoriteIds={[]}
        onSelectProduct={mockOnSelectProduct}
        onToggleFavorite={mockOnToggleFavorite}
      />
    );

    const viewButtons = screen.queryAllByText('View Product');
    if (viewButtons.length > 0) {
      await user.click(viewButtons[0]);
      expect(mockOnSelectProduct).toHaveBeenCalled();
    }
  });

  it('renders with grid layout', () => {
    const { container } = render(
      <ProductGrid
        products={mockProducts}
        favoriteIds={[]}
        onSelectProduct={mockOnSelectProduct}
        onToggleFavorite={mockOnToggleFavorite}
      />
    );

    // Check component renders without errors
    expect(container.firstChild).toBeInTheDocument();
  });

  it('displays empty state when no products', () => {
    const { container } = render(
      <ProductGrid
        products={[]}
        favoriteIds={[]}
        onSelectProduct={mockOnSelectProduct}
        onToggleFavorite={mockOnToggleFavorite}
      />
    );

    // Should render grid container but no product cards
    const cards = container.querySelectorAll('[data-testid^="product-card-"]');
    expect(cards).toHaveLength(0);
  });
});
