import { describe, it, expect, beforeEach, vi } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import { useLocalStorage, useRecentlyViewed } from './useLocalStorage';

describe('useLocalStorage Hook', () => {
  beforeEach(() => {
    localStorage.clear();
    vi.clearAllMocks();
  });

  describe('useLocalStorage', () => {
    it('returns initial value when localStorage is empty', () => {
      const { result } = renderHook(() => useLocalStorage('testKey', 'initialValue'));

      expect(result.current[0]).toBe('initialValue');
    });

    it('reads existing value from localStorage', () => {
      localStorage.setItem('testKey', JSON.stringify('existingValue'));

      const { result } = renderHook(() => useLocalStorage('testKey', 'initialValue'));

      expect(result.current[0]).toBe('existingValue');
    });

    it('stores simple string values', () => {
      const { result } = renderHook(() => useLocalStorage('name', 'John'));

      act(() => {
        result.current[1]('Jane');
      });

      expect(result.current[0]).toBe('Jane');
      expect(localStorage.getItem('name')).toBe(JSON.stringify('Jane'));
    });

    it('stores and retrieves objects', () => {
      const initialObject = { id: 1, name: 'Test' };

      const { result } = renderHook(() =>
        useLocalStorage('object', initialObject)
      );

      expect(result.current[0]).toEqual(initialObject);
    });

    it('updates object in localStorage', () => {
      const initialObject = { id: 1, name: 'Test' };

      const { result } = renderHook(() =>
        useLocalStorage('object', initialObject)
      );

      const updatedObject = { id: 1, name: 'Updated' };

      act(() => {
        result.current[1](updatedObject);
      });

      expect(result.current[0]).toEqual(updatedObject);
      expect(JSON.parse(localStorage.getItem('object')!)).toEqual(updatedObject);
    });

    it('supports updater function like useState', () => {
      const { result } = renderHook(() =>
        useLocalStorage('counter', 0)
      );

      act(() => {
        result.current[1](prev => prev + 1);
      });

      expect(result.current[0]).toBe(1);

      act(() => {
        result.current[1](prev => prev + 1);
      });

      expect(result.current[0]).toBe(2);
    });

    it('stores and retrieves arrays', () => {
      const initialArray = [1, 2, 3];

      const { result } = renderHook(() =>
        useLocalStorage('array', initialArray)
      );

      expect(result.current[0]).toEqual(initialArray);
    });

    it('updates arrays with spread operator', () => {
      const { result } = renderHook(() =>
        useLocalStorage('items', [] as number[])
      );

      act(() => {
        result.current[1](prev => [...prev, 1]);
      });

      expect(result.current[0]).toEqual([1]);

      act(() => {
        result.current[1](prev => [...prev, 2]);
      });

      expect(result.current[0]).toEqual([1, 2]);
    });

    it('handles complex nested objects', () => {
      const initialValue = {
        user: {
          name: 'John',
          preferences: {
            theme: 'dark',
            language: 'en',
          },
        },
      };

      const { result } = renderHook(() =>
        useLocalStorage('complex', initialValue)
      );

      expect(result.current[0]).toEqual(initialValue);
      expect(result.current[0].user.preferences.theme).toBe('dark');
    });

    it('persists across hook unmount and remount', () => {
      const { result, unmount } = renderHook(() =>
        useLocalStorage('persistent', 'value1')
      );

      act(() => {
        result.current[1]('value2');
      });

      unmount();

      const { result: result2 } = renderHook(() =>
        useLocalStorage('persistent', 'defaultValue')
      );

      expect(result2.current[0]).toBe('value2');
    });

    it('handles localStorage errors gracefully', () => {
      const consoleSpy = vi.spyOn(console, 'error').mockImplementation(() => {});

      // Mock localStorage to throw an error
      const getItemSpy = vi
        .spyOn(Storage.prototype, 'getItem')
        .mockImplementationOnce(() => {
          throw new Error('Storage error');
        });

      const { result } = renderHook(() =>
        useLocalStorage('testKey', 'defaultValue')
      );

      expect(result.current[0]).toBe('defaultValue');

      getItemSpy.mockRestore();
      consoleSpy.mockRestore();
    });

    it('handles setItem errors gracefully', () => {
      const consoleSpy = vi.spyOn(console, 'error').mockImplementation(() => {});

      const setItemSpy = vi
        .spyOn(Storage.prototype, 'setItem')
        .mockImplementationOnce(() => {
          throw new Error('Storage full');
        });

      const { result } = renderHook(() =>
        useLocalStorage('testKey', 'initialValue')
      );

      act(() => {
        result.current[1]('newValue');
      });

      // Should still update state even if storage fails
      expect(result.current[0]).toBe('newValue');

      setItemSpy.mockRestore();
      consoleSpy.mockRestore();
    });

    it('stores boolean values', () => {
      const { result } = renderHook(() =>
        useLocalStorage('toggle', false)
      );

      act(() => {
        result.current[1](true);
      });

      expect(result.current[0]).toBe(true);
      expect(localStorage.getItem('toggle')).toBe('true');
    });

    it('stores null values', () => {
      const { result } = renderHook(() =>
        useLocalStorage<string | null>('nullable', null)
      );

      expect(result.current[0]).toBe(null);
    });

    it('stores numeric values', () => {
      const { result } = renderHook(() =>
        useLocalStorage('number', 42)
      );

      act(() => {
        result.current[1](100);
      });

      expect(result.current[0]).toBe(100);
      expect(localStorage.getItem('number')).toBe('100');
    });

    it('handles different keys independently', () => {
      const { result: result1 } = renderHook(() =>
        useLocalStorage('key1', 'value1')
      );

      const { result: result2 } = renderHook(() =>
        useLocalStorage('key2', 'value2')
      );

      expect(result1.current[0]).toBe('value1');
      expect(result2.current[0]).toBe('value2');

      act(() => {
        result1.current[1]('updated1');
      });

      expect(result1.current[0]).toBe('updated1');
      expect(result2.current[0]).toBe('value2'); // Unchanged
    });
  });

  describe('useRecentlyViewed', () => {
    it('returns empty array initially', () => {
      const { result } = renderHook(() => useRecentlyViewed());

      expect(result.current.recentlyViewed).toEqual([]);
    });

    it('adds product to recently viewed', () => {
      const { result } = renderHook(() => useRecentlyViewed());

      act(() => {
        result.current.addToRecentlyViewed('product-1');
      });

      expect(result.current.recentlyViewed).toEqual(['product-1']);
    });

    it('moves recently viewed product to front', () => {
      const { result } = renderHook(() => useRecentlyViewed());

      act(() => {
        result.current.addToRecentlyViewed('product-1');
        result.current.addToRecentlyViewed('product-2');
        result.current.addToRecentlyViewed('product-1'); // View again
      });

      expect(result.current.recentlyViewed).toEqual(['product-1', 'product-2']);
    });

    it('limits recently viewed to 10 items', () => {
      const { result } = renderHook(() => useRecentlyViewed());

      act(() => {
        for (let i = 1; i <= 15; i++) {
          result.current.addToRecentlyViewed(`product-${i}`);
        }
      });

      expect(result.current.recentlyViewed).toHaveLength(10);
      expect(result.current.recentlyViewed[0]).toBe('product-15'); // Most recent
      expect(result.current.recentlyViewed[9]).toBe('product-6'); // Oldest kept
    });

    it('uses userId in localStorage key', () => {
      const { result } = renderHook(() => useRecentlyViewed('user-123'));

      act(() => {
        result.current.addToRecentlyViewed('product-1');
      });

      expect(localStorage.getItem('recently_viewed_user-123')).toBeTruthy();
    });

    it('uses guest as default userId', () => {
      const { result } = renderHook(() => useRecentlyViewed());

      act(() => {
        result.current.addToRecentlyViewed('product-1');
      });

      expect(localStorage.getItem('recently_viewed_guest')).toBeTruthy();
    });

    it('maintains separate lists for different users', () => {
      const { result: result1 } = renderHook(() => useRecentlyViewed('user-1'));
      const { result: result2 } = renderHook(() => useRecentlyViewed('user-2'));

      act(() => {
        result1.current.addToRecentlyViewed('product-a');
        result2.current.addToRecentlyViewed('product-b');
      });

      expect(result1.current.recentlyViewed).toEqual(['product-a']);
      expect(result2.current.recentlyViewed).toEqual(['product-b']);
    });

    it('persists recently viewed across page reloads', () => {
      const { result, unmount } = renderHook(() => useRecentlyViewed('user-test'));

      act(() => {
        result.current.addToRecentlyViewed('product-1');
        result.current.addToRecentlyViewed('product-2');
      });

      unmount();

      const { result: result2 } = renderHook(() => useRecentlyViewed('user-test'));

      expect(result2.current.recentlyViewed).toEqual(['product-2', 'product-1']);
    });

    it('removes duplicates keeping most recent', () => {
      const { result } = renderHook(() => useRecentlyViewed());

      act(() => {
        result.current.addToRecentlyViewed('product-1');
        result.current.addToRecentlyViewed('product-2');
        result.current.addToRecentlyViewed('product-3');
        result.current.addToRecentlyViewed('product-1'); // Duplicate
      });

      // product-1 should be moved to front, not duplicated
      expect(result.current.recentlyViewed).toEqual(['product-1', 'product-3', 'product-2']);
      expect(result.current.recentlyViewed).toHaveLength(3);
    });
  });
});
