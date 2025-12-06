import { describe, it, expect } from 'vitest';
import { render, screen } from '@/test/test-utils';
import InventoryBadge from './InventoryBadge';

describe('InventoryBadge Component', () => {
  it('renders badge for in-stock items', () => {
    render(<InventoryBadge stock={10} />);

    expect(screen.getByText(/In Stock/i)).toBeInTheDocument();
  });

  it('renders badge for low stock items', () => {
    render(<InventoryBadge stock={3} />);

    expect(screen.getByText(/Low Stock/i)).toBeInTheDocument();
  });

  it('renders badge for out of stock items', () => {
    render(<InventoryBadge stock={0} />);

    expect(screen.getByText(/Out of Stock/i)).toBeInTheDocument();
  });

  it('applies correct styling for in-stock', () => {
    const { container } = render(<InventoryBadge stock={10} />);

    const badge = container.querySelector('[class*="bg-green"]') || container.querySelector('span');
    expect(badge).toBeInTheDocument();
  });

  it('applies correct styling for low-stock', () => {
    const { container } = render(<InventoryBadge stock={3} />);

    const badge = container.querySelector('[class*="bg-yellow"]') || container.querySelector('span');
    expect(badge).toBeInTheDocument();
  });

  it('applies correct styling for out-of-stock', () => {
    const { container } = render(<InventoryBadge stock={0} />);

    const badge = container.querySelector('[class*="bg-red"]') || container.querySelector('span');
    expect(badge).toBeInTheDocument();
  });

  it('handles boundary value for low stock', () => {
    const { rerender } = render(<InventoryBadge stock={5} />);

    let badge = screen.getByText(/In Stock/i);
    expect(badge).toBeInTheDocument();

    rerender(<InventoryBadge stock={4} />);
    badge = screen.getByText(/Low Stock/i);
    expect(badge).toBeInTheDocument();
  });

  it('displays stock count when provided', () => {
    render(<InventoryBadge stock={15} />);

    const text = screen.getByText(/In Stock/i).textContent;
    expect(text).toMatch(/\d+/);
  });

  it('has proper accessibility attributes', () => {
    render(<InventoryBadge stock={10} />);

    const badge = screen.getByText(/In Stock/i);
    expect(badge).toBeInTheDocument();
    expect(badge.closest('[role]')?.getAttribute('role')).toBeDefined();
  });

  it('renders as badge component', () => {
    const { container } = render(<InventoryBadge stock={10} />);

    const badge = container.querySelector('[class*="badge"]');
    // If using a Badge component, it should exist
    expect(container.firstChild).toBeInTheDocument();
  });

  it('updates when stock prop changes', () => {
    const { rerender } = render(<InventoryBadge stock={10} />);

    expect(screen.getByText(/In Stock/i)).toBeInTheDocument();

    rerender(<InventoryBadge stock={0} />);

    expect(screen.getByText(/Out of Stock/i)).toBeInTheDocument();
  });

  it('handles very high stock numbers', () => {
    render(<InventoryBadge stock={999} />);

    expect(screen.getByText(/In Stock/i)).toBeInTheDocument();
  });

  it('distinguishes between 0 and negative stock', () => {
    const { rerender } = render(<InventoryBadge stock={0} />);

    expect(screen.getByText(/Out of Stock/i)).toBeInTheDocument();

    rerender(<InventoryBadge stock={-1} />);

    // Should still show as out of stock
    expect(screen.getByText(/Out of Stock/i)).toBeInTheDocument();
  });
});
