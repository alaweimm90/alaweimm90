/**
 * Memory management hook for large datasets and calculations
 */

import { useEffect, useRef, useCallback } from 'react';

export interface MemoryStats {
  usedJSHeapSize: number;
  totalJSHeapSize: number;
  jsHeapSizeLimit: number;
  percentage: number;
}

export interface MemoryManagerOptions {
  thresholdPercent?: number;
  checkInterval?: number;
  onThresholdExceeded?: (stats: MemoryStats) => void;
}

export function useMemoryManagement(options: MemoryManagerOptions = {}) {
  const {
    thresholdPercent = 80,
    checkInterval = 5000,
    onThresholdExceeded
  } = options;

  const cleanupCallbacks = useRef<(() => void)[]>([]);
  const intervalRef = useRef<NodeJS.Timeout>();

  const getMemoryStats = useCallback((): MemoryStats | null => {
    if ('memory' in performance) {
      const memory = (performance as any).memory;
      return {
        usedJSHeapSize: memory.usedJSHeapSize,
        totalJSHeapSize: memory.totalJSHeapSize,
        jsHeapSizeLimit: memory.jsHeapSizeLimit,
        percentage: (memory.usedJSHeapSize / memory.totalJSHeapSize) * 100
      };
    }
    return null;
  }, []);

  const registerCleanup = useCallback((callback: () => void) => {
    cleanupCallbacks.current.push(callback);
    
    // Return unregister function
    return () => {
      const index = cleanupCallbacks.current.indexOf(callback);
      if (index > -1) {
        cleanupCallbacks.current.splice(index, 1);
      }
    };
  }, []);

  const forceCleanup = useCallback(() => {
    // Execute all registered cleanup callbacks
    cleanupCallbacks.current.forEach(callback => {
      try {
        callback();
      } catch (error) {
        console.warn('Memory cleanup callback failed:', error);
      }
    });

    // Suggest garbage collection (if available)
    if ('gc' in window && typeof (window as any).gc === 'function') {
      (window as any).gc();
    }
  }, []);

  const checkMemoryUsage = useCallback(() => {
    const stats = getMemoryStats();
    if (stats && stats.percentage > thresholdPercent) {
      onThresholdExceeded?.(stats);
      forceCleanup();
    }
  }, [getMemoryStats, thresholdPercent, onThresholdExceeded, forceCleanup]);

  useEffect(() => {
    if (checkInterval > 0) {
      intervalRef.current = setInterval(checkMemoryUsage, checkInterval);
    }

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, [checkMemoryUsage, checkInterval]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      forceCleanup();
    };
  }, [forceCleanup]);

  return {
    getMemoryStats,
    registerCleanup,
    forceCleanup,
    checkMemoryUsage
  };
}

// Buffer pool for reusing typed arrays
export class TypedArrayPool {
  private pools = new Map<string, Float32Array[]>();
  private maxPoolSize: number;

  constructor(maxPoolSize: number = 10) {
    this.maxPoolSize = maxPoolSize;
  }

  get(type: 'Float32Array', size: number): Float32Array {
    const key = `${type}_${size}`;
    const pool = this.pools.get(key);
    
    if (pool && pool.length > 0) {
      return pool.pop()!;
    }
    
    return new Float32Array(size);
  }

  release(array: Float32Array): void {
    const key = `Float32Array_${array.length}`;
    let pool = this.pools.get(key);
    
    if (!pool) {
      pool = [];
      this.pools.set(key, pool);
    }

    if (pool.length < this.maxPoolSize) {
      // Clear the array before returning to pool
      array.fill(0);
      pool.push(array);
    }
  }

  clear(): void {
    this.pools.clear();
  }

  getStats() {
    const stats = new Map<string, number>();
    for (const [key, pool] of this.pools) {
      stats.set(key, pool.length);
    }
    return stats;
  }
}

// Global typed array pool instance
export const typedArrayPool = new TypedArrayPool();

export default useMemoryManagement;