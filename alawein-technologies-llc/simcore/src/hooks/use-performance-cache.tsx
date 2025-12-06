/**
 * Performance-optimized caching hook for expensive calculations
 */

import { useRef, useCallback, useMemo } from 'react';

export interface CacheOptions {
  maxSize?: number;
  maxAge?: number; // in milliseconds
  keyFn?: (...args: any[]) => string;
}

interface CacheEntry<T> {
  value: T;
  timestamp: number;
  accessCount: number;
  lastAccess: number;
}

export function usePerformanceCache<T>(options: CacheOptions = {}) {
  const {
    maxSize = 100,
    maxAge = 5 * 60 * 1000, // 5 minutes default
    keyFn = (...args) => JSON.stringify(args)
  } = options;

  const cache = useRef<Map<string, CacheEntry<T>>>(new Map());

  const cleanupExpired = useCallback(() => {
    const now = Date.now();
    for (const [key, entry] of cache.current) {
      if (now - entry.timestamp > maxAge) {
        cache.current.delete(key);
      }
    }
  }, [maxAge]);

  const evictLeastUsed = useCallback(() => {
    if (cache.current.size <= maxSize) return;

    // Sort by access count and last access time
    const entries = Array.from(cache.current.entries()).sort((a, b) => {
      const [, entryA] = a;
      const [, entryB] = b;
      
      // First by access count, then by last access time
      if (entryA.accessCount !== entryB.accessCount) {
        return entryA.accessCount - entryB.accessCount;
      }
      return entryA.lastAccess - entryB.lastAccess;
    });

    // Remove oldest entries
    const toRemove = entries.slice(0, entries.length - maxSize);
    toRemove.forEach(([key]) => cache.current.delete(key));
  }, [maxSize]);

  const get = useCallback((key: string): T | undefined => {
    cleanupExpired();
    
    const entry = cache.current.get(key);
    if (!entry) return undefined;

    // Update access statistics
    entry.accessCount++;
    entry.lastAccess = Date.now();
    
    return entry.value;
  }, [cleanupExpired]);

  const set = useCallback((key: string, value: T) => {
    const now = Date.now();
    
    cache.current.set(key, {
      value,
      timestamp: now,
      accessCount: 1,
      lastAccess: now
    });

    evictLeastUsed();
  }, [evictLeastUsed]);

  const memoize = useCallback(<Args extends any[], Result>(
    fn: (...args: Args) => Result
  ) => {
    return (...args: Args): Result => {
      const key = keyFn(...args);
      const cached = get(key);
      
      if (cached !== undefined) {
        return cached as unknown as Result;
      }

      const result = fn(...args);
      set(key, result as unknown as T);
      return result;
    };
  }, [get, set, keyFn]);

  const memoizeAsync = useCallback(<Args extends any[], Result>(
    fn: (...args: Args) => Promise<Result>
  ) => {
    const pendingPromises = new Map<string, Promise<Result>>();
    
    return async (...args: Args): Promise<Result> => {
      const key = keyFn(...args);
      const cached = get(key);
      
      if (cached !== undefined) {
        return cached as unknown as Result;
      }

      // Check if computation is already in progress
      const pending = pendingPromises.get(key);
      if (pending) return pending;

      // Start new computation
      const promise = fn(...args).then(result => {
        set(key, result as unknown as T);
        pendingPromises.delete(key);
        return result;
      }).catch(error => {
        pendingPromises.delete(key);
        throw error;
      });

      pendingPromises.set(key, promise);
      return promise;
    };
  }, [get, set, keyFn]);

  const clear = useCallback(() => {
    cache.current.clear();
  }, []);

  const stats = useMemo(() => ({
    size: cache.current.size,
    maxSize,
    maxAge,
    hitRate: 0 // Could track this with additional counters
  }), [maxSize, maxAge]);

  return {
    get,
    set,
    memoize,
    memoizeAsync,
    clear,
    stats
  };
}

export default usePerformanceCache;