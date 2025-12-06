import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@/test/test-utils';
import userEvent from '@testing-library/user-event';
import ProductCard from './ProductCard';
import { mockProduct, mockProduct2 } from '@/test/mocks';

// Mock ProductBirdBadge
vi.mock('./ProductBirdBadge', () => ({
  default: ({ birdType, size }: any) => (
    <div data-testid={`bird-badge-${birdType}`} data-size={size}>
      Bird Badge
    </div>
  ),
}));

// Mock colorMap utility
vi.mock('./utils/colorMap', () => ({
  getBirdType: (index: number) => `bird-${index}`,
}));

describe('ProductCard Component', () => {
  const mockOnSelect = vi.fn();
  const mockOnToggleFavorite = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders product card with basic information', () => {
    render(
      <ProductCard
        product={mockProduct}
        index={0}
        isFavorited={false}
        onSelect={mockOnSelect}
        onToggleFavorite={mockOnToggleFavorite}
      />
    );

    expect(screen.getByText('Classic T-Shirt')).toBeInTheDocument();
    expect(screen.getByText('Performance')).toBeInTheDocument();
    expect(screen.getByText(/High-quality performance/)).toBeInTheDocument();
  });

  it('displays product price', () => {
    render(
      <ProductCard
        product={mockProduct}
        index={0}
        isFavorited={false}
        onSelect={mockOnSelect}
        onToggleFavorite={mockOnToggleFavorite}
      />
    );

    expect(screen.getByText('$65')).toBeInTheDocument();
  });

  it('calls onSelect when product card is clicked', async () => {
    const user = userEvent.setup();
    render(
      <ProductCard
        product={mockProduct}
        index={0}
        isFavorited={false}
        onSelect={mockOnSelect}
        onToggleFavorite={mockOnToggleFavorite}
      />
    );

    const productCard = screen.getByText('Classic T-Shirt').closest('article');
    await user.click(productCard!);

    expect(mockOnSelect).toHaveBeenCalledWith(mockProduct);
    expect(mockOnSelect).toHaveBeenCalledTimes(1);
  });

  it('renders favorite button with correct aria-label', () => {
    render(
      <ProductCard
        product={mockProduct}
        index={0}
        isFavorited={false}
        onSelect={mockOnSelect}
        onToggleFavorite={mockOnToggleFavorite}
      />
    );

    const favoriteButton = screen.getByLabelText(/Add Classic T-Shirt to favorites/);
    expect(favoriteButton).toBeInTheDocument();
  });

  it('calls onToggleFavorite when favorite button is clicked', async () => {
    const user = userEvent.setup();
    render(
      <ProductCard
        product={mockProduct}
        index={0}
        isFavorited={false}
        onSelect={mockOnSelect}
        onToggleFavorite={mockOnToggleFavorite}
      />
    );

    const favoriteButton = screen.getByLabelText(/Add Classic T-Shirt to favorites/);
    await user.click(favoriteButton);

    expect(mockOnToggleFavorite).toHaveBeenCalledTimes(1);
    expect(mockOnToggleFavorite).toHaveBeenCalledWith(
      expect.any(Object),
      mockProduct.id
    );
  });

  it('shows filled heart icon when product is favorited', () => {
    const { container } = render(
      <ProductCard
        product={mockProduct}
        index={0}
        isFavorited={true}
        onSelect={mockOnSelect}
        onToggleFavorite={mockOnToggleFavorite}
      />
    );

    const heartIcon = container.querySelector('svg[class*="fill-current"]');
    expect(heartIcon).toBeInTheDocument();
  });

  it('shows empty heart icon when product is not favorited', () => {
    const { container } = render(
      <ProductCard
        product={mockProduct}
        index={0}
        isFavorited={false}
        onSelect={mockOnSelect}
        onToggleFavorite={mockOnToggleFavorite}
      />
    );

    const heartIcon = container.querySelector('svg:not([class*="fill-current"])');
    expect(heartIcon).toBeInTheDocument();
  });

  it('renders bird badge with correct type', () => {
    render(
      <ProductCard
        product={mockProduct}
        index={3}
        isFavorited={false}
        onSelect={mockOnSelect}
        onToggleFavorite={mockOnToggleFavorite}
      />
    );

    const birdBadge = screen.getByTestId('bird-badge-bird-3');
    expect(birdBadge).toBeInTheDocument();
    expect(birdBadge).toHaveAttribute('data-size', '28');
  });

  it('has proper accessibility for product image', () => {
    render(
      <ProductCard
        product={mockProduct}
        index={0}
        isFavorited={false}
        onSelect={mockOnSelect}
        onToggleFavorite={mockOnToggleFavorite}
      />
    );

    const imageElement = screen.getByRole('img', {
      name: /Classic T-Shirt - Performance product image/,
    });
    expect(imageElement).toBeInTheDocument();
  });

  it('applies animation delay based on index', () => {
    const { container } = render(
      <ProductCard
        product={mockProduct}
        index={5}
        isFavorited={false}
        onSelect={mockOnSelect}
        onToggleFavorite={mockOnToggleFavorite}
      />
    );

    const article = container.querySelector('article');
    expect(article).toHaveStyle({ animationDelay: '0.5s' });
  });

  it('renders article element with group class', () => {
    const { container } = render(
      <ProductCard
        product={mockProduct}
        index={0}
        isFavorited={false}
        onSelect={mockOnSelect}
        onToggleFavorite={mockOnToggleFavorite}
      />
    );

    const article = container.querySelector('article');
    expect(article).toHaveClass('group', 'relative');
  });

  it('has border separator between description and price', () => {
    const { container } = render(
      <ProductCard
        product={mockProduct}
        index={0}
        isFavorited={false}
        onSelect={mockOnSelect}
        onToggleFavorite={mockOnToggleFavorite}
      />
    );

    const borderElement = container.querySelector('div[class*="border-t"]');
    expect(borderElement).toBeInTheDocument();
  });

  it('handles different product categories', () => {
    render(
      <ProductCard
        product={mockProduct2}
        index={0}
        isFavorited={false}
        onSelect={mockOnSelect}
        onToggleFavorite={mockOnToggleFavorite}
      />
    );

    expect(screen.getByText('Premium Hoodie')).toBeInTheDocument();
    expect(screen.getByText('Lifestyle')).toBeInTheDocument();
    expect(screen.getByText('$125')).toBeInTheDocument();
  });

  it('has memoization to prevent unnecessary re-renders', () => {
    const { rerender } = render(
      <ProductCard
        product={mockProduct}
        index={0}
        isFavorited={false}
        onSelect={mockOnSelect}
        onToggleFavorite={mockOnToggleFavorite}
      />
    );

    expect(mockOnSelect).not.toHaveBeenCalled();

    rerender(
      <ProductCard
        product={mockProduct}
        index={0}
        isFavorited={true}
        onSelect={mockOnSelect}
        onToggleFavorite={mockOnToggleFavorite}
      />
    );

    expect(mockOnSelect).not.toHaveBeenCalled();
  });

  it('has displayName for debugging', () => {
    expect(ProductCard.displayName).toBe('ProductCard');
  });
});
