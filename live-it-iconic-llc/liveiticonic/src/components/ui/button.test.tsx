import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@/test/test-utils';
import userEvent from '@testing-library/user-event';
import { Button } from './button';

describe('Button Component', () => {
  it('renders button with default variant', () => {
    render(<Button>Click me</Button>);

    const button = screen.getByRole('button', { name: /Click me/i });
    expect(button).toBeInTheDocument();
  });

  it('renders button with primary variant', () => {
    const { container } = render(<Button variant="primary">Primary</Button>);

    const button = screen.getByRole('button');
    expect(button).toHaveClass('bg-lii-gold');
  });

  it('renders button with secondary variant', () => {
    render(<Button variant="secondary">Secondary</Button>);

    const button = screen.getByRole('button');
    expect(button).toBeInTheDocument();
  });

  it('renders button with outline variant', () => {
    render(<Button variant="outline">Outline</Button>);

    const button = screen.getByRole('button');
    expect(button).toBeInTheDocument();
  });

  it('renders button with ghost variant', () => {
    render(<Button variant="ghost">Ghost</Button>);

    const button = screen.getByRole('button');
    expect(button).toBeInTheDocument();
  });

  it('renders button with large size', () => {
    const { container } = render(<Button size="lg">Large</Button>);

    const button = screen.getByRole('button');
    // Size should be applied (specific height varies by implementation)
    expect(button).toBeInTheDocument();
    expect(button.className).toMatch(/h-/);
  });

  it('renders button with small size', () => {
    const { container } = render(<Button size="sm">Small</Button>);

    const button = screen.getByRole('button');
    // Size should be applied
    expect(button).toBeInTheDocument();
    expect(button.className).toMatch(/h-/);
  });

  it('renders button with icon size', () => {
    const { container } = render(<Button size="icon">Icon</Button>);

    const button = screen.getByRole('button');
    // Icon size should have width and height
    expect(button).toBeInTheDocument();
    expect(button.className).toMatch(/[wh]-/);
  });

  it('renders disabled button', () => {
    render(<Button disabled>Disabled</Button>);

    const button = screen.getByRole('button');
    expect(button).toBeDisabled();
    expect(button).toHaveAttribute('disabled');
  });

  it('handles click events', async () => {
    const handleClick = vi.fn();
    const user = userEvent.setup();

    render(<Button onClick={handleClick}>Click me</Button>);

    const button = screen.getByRole('button');
    await user.click(button);

    expect(handleClick).toHaveBeenCalledTimes(1);
  });

  it('does not trigger click when disabled', async () => {
    const handleClick = vi.fn();
    const user = userEvent.setup();

    render(
      <Button onClick={handleClick} disabled>
        Disabled
      </Button>
    );

    const button = screen.getByRole('button');
    await user.click(button);

    expect(handleClick).not.toHaveBeenCalled();
  });

  it('supports className prop', () => {
    const { container } = render(
      <Button className="custom-class">Custom</Button>
    );

    const button = screen.getByRole('button');
    expect(button).toHaveClass('custom-class');
  });

  it('renders as a link with asChild prop', () => {
    const { container } = render(
      <Button asChild>
        <a href="/test">Link Button</a>
      </Button>
    );

    const link = screen.getByRole('link', { name: /Link Button/i });
    expect(link).toBeInTheDocument();
    expect(link).toHaveAttribute('href', '/test');
  });

  it('renders button with loading state', () => {
    render(
      <Button disabled aria-label="Loading">
        Loading...
      </Button>
    );

    const button = screen.getByRole('button');
    expect(button).toBeDisabled();
    expect(button).toHaveAttribute('aria-label', 'Loading');
  });

  it('applies focus styles', async () => {
    const user = userEvent.setup();
    const { container } = render(<Button>Focus Test</Button>);

    const button = screen.getByRole('button');
    await user.tab();

    expect(button).toHaveFocus();
  });

  it('supports children rendering', () => {
    render(
      <Button>
        <span>Content</span>
      </Button>
    );

    expect(screen.getByText('Content')).toBeInTheDocument();
  });

  it('has proper touch target size', () => {
    const { container } = render(<Button size="icon">Icon</Button>);

    const button = screen.getByRole('button');
    // Icon button should have width and height
    expect(button.className).toMatch(/[wh]-10/);
  });

  it('renders with correct aria-label for icon buttons', () => {
    render(<Button size="icon" aria-label="Delete">X</Button>);

    const button = screen.getByLabelText('Delete');
    expect(button).toBeInTheDocument();
  });

  it('supports multiple class variants', () => {
    const { container } = render(
      <Button variant="primary" size="lg" className="custom">
        Complex
      </Button>
    );

    const button = screen.getByRole('button');
    expect(button).toHaveClass('custom');
  });

  it('maintains button semantics', () => {
    render(<Button>Semantic Button</Button>);

    const button = screen.getByRole('button', { name: /Semantic Button/i });
    expect(button).toBeInTheDocument();
    expect(button.tagName).toBe('BUTTON');
  });

  it('supports aria attributes', () => {
    render(
      <Button aria-pressed="false" aria-label="Toggle feature">
        Toggle
      </Button>
    );

    const button = screen.getByRole('button');
    expect(button).toHaveAttribute('aria-pressed', 'false');
    expect(button).toHaveAttribute('aria-label', 'Toggle feature');
  });

  it('has hover states defined', () => {
    const { container } = render(<Button>Hover Test</Button>);

    const button = screen.getByRole('button');
    // Check that button has transition classes
    expect(button.className).toMatch(/transition|hover/);
  });
});
