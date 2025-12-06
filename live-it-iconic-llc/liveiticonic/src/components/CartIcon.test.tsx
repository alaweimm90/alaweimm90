import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@/test/test-utils';
import userEvent from '@testing-library/user-event';
import CartIcon from './CartIcon';

describe('CartIcon Component', () => {
  const mockOnClick = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders cart icon button', () => {
    render(<CartIcon onClick={mockOnClick} />);

    const button = screen.getByRole('button');
    expect(button).toBeInTheDocument();
  });

  it('calls onClick when clicked', async () => {
    const user = userEvent.setup();
    render(<CartIcon onClick={mockOnClick} />);

    const button = screen.getByRole('button');
    await user.click(button);

    expect(mockOnClick).toHaveBeenCalledTimes(1);
  });

  it('has proper accessibility attributes', () => {
    render(<CartIcon onClick={mockOnClick} />);

    const button = screen.getByRole('button');
    expect(button).toHaveAttribute('aria-label');
  });

  it('has minimum touch target size', () => {
    render(<CartIcon onClick={mockOnClick} />);

    const button = screen.getByRole('button');
    expect(button).toHaveClass('min-w-[44px]', 'min-h-[44px]');
  });

  it('has hover effects', () => {
    const { container } = render(<CartIcon onClick={mockOnClick} />);

    const button = screen.getByRole('button');
    expect(button).toHaveClass('hover:text-lii-gold');
  });

  it('has transition animation', () => {
    const { container } = render(<CartIcon onClick={mockOnClick} />);

    const button = screen.getByRole('button');
    expect(button).toHaveClass('transition-all', 'duration-micro');
  });

  it('renders shopping cart icon', () => {
    const { container } = render(<CartIcon onClick={mockOnClick} />);

    const svg = container.querySelector('svg');
    expect(svg).toBeInTheDocument();
  });

  it('supports custom className', () => {
    const { container } = render(
      <CartIcon onClick={mockOnClick} className="custom-class" />
    );

    const button = screen.getByRole('button');
    expect(button).toHaveClass('custom-class');
  });

  it('is keyboard accessible', async () => {
    const user = userEvent.setup();
    render(<CartIcon onClick={mockOnClick} />);

    const button = screen.getByRole('button');
    await user.tab();

    expect(button).toHaveFocus();
  });

  it('handles rapid clicks', async () => {
    const user = userEvent.setup();
    render(<CartIcon onClick={mockOnClick} />);

    const button = screen.getByRole('button');
    await user.click(button);
    await user.click(button);
    await user.click(button);

    expect(mockOnClick).toHaveBeenCalledTimes(3);
  });
});
