/**
 * useLocalStorage Hook
 *
 * Synchronizes component state with localStorage, providing persistent state
 * across page reloads and browser sessions. Handles serialization/deserialization
 * automatically with error handling.
 *
 * @module useLocalStorage
 *
 * @example
 * // Basic usage with string value
 * const [theme, setTheme] = useLocalStorage('theme', 'dark');
 *
 * // Update value (saves to localStorage)
 * setTheme('light');
 *
 * @example
 * // With complex objects
 * const [user, setUser] = useLocalStorage<User>('currentUser', defaultUser);
 *
 * // With updater function
 * setUser(prev => ({ ...prev, theme: 'light' }));
 *
 * @example
 * // With arrays
 * const [cart, setCart] = useLocalStorage<CartItem[]>('cart', []);
 * setCart(prev => [...prev, newItem]);
 */

import { useState, useEffect } from 'react';

/**
 * Hook for managing state synced with localStorage
 *
 * Provides a way to persist component state across page reloads and browser
 * sessions using localStorage. Handles JSON serialization, parse errors,
 * and updates both state and storage simultaneously.
 *
 * @template T - Type of value being stored
 * @param {string} key - localStorage key for storing the value
 * @param {T} initialValue - Default value if key doesn't exist in localStorage
 *
 * @returns {[T, (value: T | ((val: T) => T)) => void]} Tuple containing:
 *   - value: Current value from localStorage or initialValue
 *   - setValue: Function to update value (also updates localStorage)
 *
 * @throws {Error} Silently catches and logs localStorage errors (quota exceeded, disabled)
 *
 * @example
 * const [name, setName] = useLocalStorage('name', 'John');
 * // Reads from localStorage.name, updates on setName()
 *
 * @example
 * // With initial complex value
 * const [preferences, setPreferences] = useLocalStorage<Preferences>(
 *   'userPreferences',
 *   { theme: 'dark', language: 'en' }
 * );
 *
 * @example
 * // Using updater function (like useState)
 * setPreferences(prev => ({
 *   ...prev,
 *   theme: prev.theme === 'dark' ? 'light' : 'dark'
 * }));
 *
 * @see {@link useRecentlyViewed} for example of hook composition
 */
export function useLocalStorage<T>(
  key: string,
  initialValue: T
): [T, (value: T | ((val: T) => T)) => void] {
  // State to store our value
  const [storedValue, setStoredValue] = useState<T>(() => {
    try {
      const item = window.localStorage.getItem(key);
      return item ? JSON.parse(item) : initialValue;
    } catch (error) {
      console.error('Error reading from localStorage:', error);
      return initialValue;
    }
  });

  // Return a wrapped version of useState's setter function that persists the new value to localStorage
  /**
   * Update stored value and sync with localStorage
   *
   * Supports both direct values and updater functions (like useState).
   * Automatically serializes to JSON and saves to localStorage.
   *
   * @param {T | Function} value - New value or updater function
   * @returns {void}
   *
   * @example
   * // Direct value
   * setValue('new value');
   *
   * // Updater function
   * setValue(prev => ({ ...prev, field: 'updated' }));
   */
  const setValue = (value: T | ((val: T) => T)) => {
    try {
      // Allow value to be a function so we have same API as useState
      const valueToStore = value instanceof Function ? value(storedValue) : value;
      setStoredValue(valueToStore);
      window.localStorage.setItem(key, JSON.stringify(valueToStore));
    } catch (error) {
      console.error('Error saving to localStorage:', error);
    }
  };

  return [storedValue, setValue];
}

/**
 * Hook for tracking recently viewed products
 *
 * Maintains a list of recently viewed products in localStorage,
 * limited to the 10 most recent items.
 *
 * @param {string} [userId] - Optional user ID for personalization
 *
 * @returns {Object} Object containing:
 *   - recentlyViewed: Array of product IDs viewed recently
 *   - addToRecentlyViewed: Function to add product to recently viewed
 *
 * @example
 * const { recentlyViewed, addToRecentlyViewed } = useRecentlyViewed(userId);
 *
 * // Add product when viewed
 * useEffect(() => {
 *   addToRecentlyViewed(productId);
 * }, [productId]);
 *
 * // Get products
 * const products = await productService.getByIds(recentlyViewed);
 */

// Helper hook for recently viewed products
export function useRecentlyViewed(userId?: string) {
  const [recentlyViewed, setRecentlyViewed] = useLocalStorage<string[]>(
    `recently_viewed_${userId || 'guest'}`,
    []
  );

  const addToRecentlyViewed = (productId: string) => {
    setRecentlyViewed(prev => {
      const filtered = prev.filter(id => id !== productId);
      return [productId, ...filtered].slice(0, 10); // Keep last 10
    });
  };

  return { recentlyViewed, addToRecentlyViewed };
}
