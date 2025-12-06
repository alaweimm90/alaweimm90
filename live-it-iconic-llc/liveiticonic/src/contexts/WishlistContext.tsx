import React, { createContext, useContext, useState, useEffect } from 'react';

interface WishlistItem {
  productId: string;
  addedAt: Date;
}

interface WishlistContextType {
  items: WishlistItem[];
  count: number;
  addToWishlist: (productId: string) => void;
  removeFromWishlist: (productId: string) => void;
  toggleWishlist: (productId: string) => void;
  isInWishlist: (productId: string) => boolean;
  clearWishlist: () => void;
  getWishlistUrl: () => string;
}

const WishlistContext = createContext<WishlistContextType | undefined>(undefined);

const WISHLIST_STORAGE_KEY = 'lii-wishlist';

/**
 * WishlistProvider component that manages wishlist state and persistence
 *
 * Persists wishlist to localStorage and provides methods for managing wishlist items.
 * Automatically syncs with localStorage on mount and whenever items change.
 *
 * @component
 * @example
 * <WishlistProvider>
 *   <App />
 * </WishlistProvider>
 */
export const WishlistProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [items, setItems] = useState<WishlistItem[]>([]);

  // Load from localStorage on mount
  useEffect(() => {
    const saved = localStorage.getItem(WISHLIST_STORAGE_KEY);
    if (saved) {
      try {
        const parsed = JSON.parse(saved);
        setItems(parsed.map((item: any) => ({ ...item, addedAt: new Date(item.addedAt) })));
      } catch (e) {
        console.error('Failed to load wishlist from localStorage', e);
      }
    }
  }, []);

  // Save to localStorage whenever items change
  useEffect(() => {
    localStorage.setItem(WISHLIST_STORAGE_KEY, JSON.stringify(items));
  }, [items]);

  const addToWishlist = (productId: string) => {
    setItems(prev => {
      if (prev.some(item => item.productId === productId)) {
        return prev;
      }
      return [...prev, { productId, addedAt: new Date() }];
    });
  };

  const removeFromWishlist = (productId: string) => {
    setItems(prev => prev.filter(item => item.productId !== productId));
  };

  const toggleWishlist = (productId: string) => {
    if (isInWishlist(productId)) {
      removeFromWishlist(productId);
    } else {
      addToWishlist(productId);
    }
  };

  const isInWishlist = (productId: string) => {
    return items.some(item => item.productId === productId);
  };

  const clearWishlist = () => {
    setItems([]);
  };

  const getWishlistUrl = () => {
    const ids = items.map(item => item.productId).join(',');
    const baseUrl = `${window.location.origin}/wishlist`;
    return ids ? `${baseUrl}?items=${ids}` : baseUrl;
  };

  const value: WishlistContextType = {
    items,
    count: items.length,
    addToWishlist,
    removeFromWishlist,
    toggleWishlist,
    isInWishlist,
    clearWishlist,
    getWishlistUrl,
  };

  return <WishlistContext.Provider value={value}>{children}</WishlistContext.Provider>;
};

/**
 * Hook to use wishlist context
 *
 * @example
 * const { addToWishlist, isInWishlist, count } = useWishlist();
 */
export const useWishlist = () => {
  const context = useContext(WishlistContext);
  if (!context) {
    throw new Error('useWishlist must be used within a WishlistProvider');
  }
  return context;
};

export default WishlistContext;
