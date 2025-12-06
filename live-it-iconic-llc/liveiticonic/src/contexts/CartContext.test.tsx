import { describe, it, expect, beforeEach, vi } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import { CartProvider, useCart } from './CartContext';
import { mockCartItem, mockProduct, mockProduct2 } from '@/test/mocks';

describe('CartContext', () => {
  beforeEach(() => {
    localStorage.clear();
  });

  const wrapper = ({ children }: { children: React.ReactNode }) => (
    <CartProvider>{children}</CartProvider>
  );

  it('provides initial empty cart state', () => {
    const { result } = renderHook(() => useCart(), { wrapper });

    expect(result.current.items).toEqual([]);
    expect(result.current.total).toBe(0);
    expect(result.current.itemCount).toBe(0);
    expect(result.current.isOpen).toBe(false);
  });

  it('adds item to cart', () => {
    const { result } = renderHook(() => useCart(), { wrapper });

    act(() => {
      result.current.addItem({
        id: mockProduct.id,
        name: mockProduct.name,
        price: mockProduct.price,
        image: 'https://example.com/image.jpg',
      });
    });

    expect(result.current.items).toHaveLength(1);
    expect(result.current.items[0]).toMatchObject({
      id: mockProduct.id,
      name: mockProduct.name,
      price: mockProduct.price,
      quantity: 1,
    });
  });

  it('increments quantity when adding existing item', () => {
    const { result } = renderHook(() => useCart(), { wrapper });

    const item = {
      id: mockProduct.id,
      name: mockProduct.name,
      price: mockProduct.price,
      image: 'https://example.com/image.jpg',
    };

    act(() => {
      result.current.addItem(item);
      result.current.addItem(item);
    });

    expect(result.current.items).toHaveLength(1);
    expect(result.current.items[0].quantity).toBe(2);
  });

  it('calculates correct total when adding items', () => {
    const { result } = renderHook(() => useCart(), { wrapper });

    act(() => {
      result.current.addItem({
        id: mockProduct.id,
        name: mockProduct.name,
        price: mockProduct.price,
        image: 'https://example.com/image.jpg',
      });
      result.current.addItem({
        id: mockProduct2.id,
        name: mockProduct2.name,
        price: mockProduct2.price,
        image: 'https://example.com/image.jpg',
      });
    });

    const expectedTotal = mockProduct.price + mockProduct2.price;
    expect(result.current.total).toBe(expectedTotal);
  });

  it('updates item count correctly', () => {
    const { result } = renderHook(() => useCart(), { wrapper });

    act(() => {
      result.current.addItem({
        id: mockProduct.id,
        name: mockProduct.name,
        price: mockProduct.price,
        image: 'https://example.com/image.jpg',
      });
      result.current.addItem({
        id: mockProduct.id,
        name: mockProduct.name,
        price: mockProduct.price,
        image: 'https://example.com/image.jpg',
      });
    });

    expect(result.current.itemCount).toBe(2);
  });

  it('removes item from cart', () => {
    const { result } = renderHook(() => useCart(), { wrapper });

    act(() => {
      result.current.addItem({
        id: mockProduct.id,
        name: mockProduct.name,
        price: mockProduct.price,
        image: 'https://example.com/image.jpg',
      });
      result.current.addItem({
        id: mockProduct2.id,
        name: mockProduct2.name,
        price: mockProduct2.price,
        image: 'https://example.com/image.jpg',
      });
    });

    expect(result.current.items).toHaveLength(2);

    act(() => {
      result.current.removeItem(mockProduct.id);
    });

    expect(result.current.items).toHaveLength(1);
    expect(result.current.items[0].id).toBe(mockProduct2.id);
  });

  it('recalculates total when removing item', () => {
    const { result } = renderHook(() => useCart(), { wrapper });

    act(() => {
      result.current.addItem({
        id: mockProduct.id,
        name: mockProduct.name,
        price: mockProduct.price,
        image: 'https://example.com/image.jpg',
      });
      result.current.addItem({
        id: mockProduct2.id,
        name: mockProduct2.name,
        price: mockProduct2.price,
        image: 'https://example.com/image.jpg',
      });
    });

    const totalBefore = result.current.total;

    act(() => {
      result.current.removeItem(mockProduct2.id);
    });

    expect(result.current.total).toBe(totalBefore - mockProduct2.price);
  });

  it('updates item quantity', () => {
    const { result } = renderHook(() => useCart(), { wrapper });

    act(() => {
      result.current.addItem({
        id: mockProduct.id,
        name: mockProduct.name,
        price: mockProduct.price,
        image: 'https://example.com/image.jpg',
      });
    });

    act(() => {
      result.current.updateQuantity(mockProduct.id, 5);
    });

    expect(result.current.items[0].quantity).toBe(5);
  });

  it('removes item when quantity is set to 0', () => {
    const { result } = renderHook(() => useCart(), { wrapper });

    act(() => {
      result.current.addItem({
        id: mockProduct.id,
        name: mockProduct.name,
        price: mockProduct.price,
        image: 'https://example.com/image.jpg',
      });
    });

    act(() => {
      result.current.updateQuantity(mockProduct.id, 0);
    });

    expect(result.current.items).toHaveLength(0);
  });

  it('removes item when quantity is set to negative', () => {
    const { result } = renderHook(() => useCart(), { wrapper });

    act(() => {
      result.current.addItem({
        id: mockProduct.id,
        name: mockProduct.name,
        price: mockProduct.price,
        image: 'https://example.com/image.jpg',
      });
    });

    act(() => {
      result.current.updateQuantity(mockProduct.id, -5);
    });

    expect(result.current.items).toHaveLength(0);
  });

  it('recalculates total when updating quantity', () => {
    const { result } = renderHook(() => useCart(), { wrapper });

    act(() => {
      result.current.addItem({
        id: mockProduct.id,
        name: mockProduct.name,
        price: mockProduct.price,
        image: 'https://example.com/image.jpg',
      });
    });

    const initialTotal = result.current.total;

    act(() => {
      result.current.updateQuantity(mockProduct.id, 3);
    });

    expect(result.current.total).toBe(initialTotal * 3);
  });

  it('clears entire cart', () => {
    const { result } = renderHook(() => useCart(), { wrapper });

    act(() => {
      result.current.addItem({
        id: mockProduct.id,
        name: mockProduct.name,
        price: mockProduct.price,
        image: 'https://example.com/image.jpg',
      });
      result.current.addItem({
        id: mockProduct2.id,
        name: mockProduct2.name,
        price: mockProduct2.price,
        image: 'https://example.com/image.jpg',
      });
    });

    expect(result.current.items.length).toBeGreaterThan(0);

    act(() => {
      result.current.clearCart();
    });

    expect(result.current.items).toHaveLength(0);
    expect(result.current.total).toBe(0);
    expect(result.current.itemCount).toBe(0);
  });

  it('sets isOpen state', () => {
    const { result } = renderHook(() => useCart(), { wrapper });

    expect(result.current.isOpen).toBe(false);

    act(() => {
      result.current.setIsOpen(true);
    });

    expect(result.current.isOpen).toBe(true);

    act(() => {
      result.current.setIsOpen(false);
    });

    expect(result.current.isOpen).toBe(false);
  });

  it('saves cart to localStorage', async () => {
    const { result } = renderHook(() => useCart(), { wrapper });

    act(() => {
      result.current.addItem({
        id: mockProduct.id,
        name: mockProduct.name,
        price: mockProduct.price,
        image: 'https://example.com/image.jpg',
      });
    });

    // Wait for localStorage to be updated
    await new Promise(resolve => setTimeout(resolve, 0));

    const savedCart = localStorage.getItem('liveiticonic-cart');
    expect(savedCart).toBeTruthy();

    const cartData = JSON.parse(savedCart!);
    expect(cartData).toHaveLength(1);
    expect(cartData[0].id).toBe(mockProduct.id);
  });

  it('loads cart from localStorage on mount', () => {
    const cartData = [
      {
        id: mockProduct.id,
        name: mockProduct.name,
        price: mockProduct.price,
        quantity: 2,
        image: 'https://example.com/image.jpg',
      },
    ];

    localStorage.setItem('liveiticonic-cart', JSON.stringify(cartData));

    const { result } = renderHook(() => useCart(), { wrapper });

    expect(result.current.items).toHaveLength(1);
    expect(result.current.items[0].id).toBe(mockProduct.id);
    expect(result.current.items[0].quantity).toBe(2);
  });

  it('throws error when useCart is called outside provider', () => {
    // Suppress console.error for this test
    const consoleSpy = vi.spyOn(console, 'error').mockImplementation(() => {});

    expect(() => {
      renderHook(() => useCart());
    }).toThrow('useCart must be used within a CartProvider');

    consoleSpy.mockRestore();
  });

  it('correctly calculates totals with multiple items and quantities', () => {
    const { result } = renderHook(() => useCart(), { wrapper });

    act(() => {
      result.current.addItem({
        id: mockProduct.id,
        name: mockProduct.name,
        price: mockProduct.price,
        image: 'https://example.com/image.jpg',
      });
      result.current.updateQuantity(mockProduct.id, 2);

      result.current.addItem({
        id: mockProduct2.id,
        name: mockProduct2.name,
        price: mockProduct2.price,
        image: 'https://example.com/image.jpg',
      });
      result.current.updateQuantity(mockProduct2.id, 3);
    });

    const expectedTotal = mockProduct.price * 2 + mockProduct2.price * 3;
    const expectedItemCount = 2 + 3;

    expect(result.current.total).toBe(expectedTotal);
    expect(result.current.itemCount).toBe(expectedItemCount);
  });

  it('maintains consistency between items length and itemCount', () => {
    const { result } = renderHook(() => useCart(), { wrapper });

    act(() => {
      result.current.addItem({
        id: mockProduct.id,
        name: mockProduct.name,
        price: mockProduct.price,
        image: 'https://example.com/image.jpg',
      });
      result.current.addItem({
        id: mockProduct.id,
        name: mockProduct.name,
        price: mockProduct.price,
        image: 'https://example.com/image.jpg',
      });
    });

    // 2 items, but 1 unique item with quantity 2
    expect(result.current.items.length).toBe(1);
    expect(result.current.itemCount).toBe(2);
  });
});
