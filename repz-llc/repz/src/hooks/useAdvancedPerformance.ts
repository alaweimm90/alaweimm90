import { useState, useEffect, useCallback, useRef } from 'react';
import { logger } from '@/utils/logger';
import { performanceMonitor } from '@/utils/performance';

// ============================================================================
// PHASE 3: ADVANCED PERFORMANCE OPTIMIZATION
// Intelligent caching, lazy loading, and performance monitoring
// ============================================================================

interface CacheEntry<T> {
  data: T;
  timestamp: number;
  ttl: number;
  hits: number;
}

class IntelligentCache<T = unknown> {
  private cache = new Map<string, CacheEntry<T>>();
  private maxSize: number;
  private defaultTTL: number;

  constructor(maxSize = 100, defaultTTL = 300000) { // 5 minutes default
    this.maxSize = maxSize;
    this.defaultTTL = defaultTTL;
  }

  set(key: string, data: T, ttl = this.defaultTTL): void {
    // If cache is full, remove least recently used
    if (this.cache.size >= this.maxSize) {
      const oldestKey = Array.from(this.cache.entries())
        .sort(([,a], [,b]) => a.timestamp - b.timestamp)[0][0];
      this.cache.delete(oldestKey);
    }

    this.cache.set(key, {
      data,
      timestamp: Date.now(),
      ttl,
      hits: 0
    });
  }

  get(key: string): T | null {
    const entry = this.cache.get(key);
    if (!entry) return null;

    // Check if expired
    if (Date.now() - entry.timestamp > entry.ttl) {
      this.cache.delete(key);
      return null;
    }

    // Update hit count and timestamp for LRU
    entry.hits++;
    entry.timestamp = Date.now();
    return entry.data;
  }

  clear(): void {
    this.cache.clear();
  }

  getStats() {
    return {
      size: this.cache.size,
      totalHits: Array.from(this.cache.values()).reduce((sum, entry) => sum + entry.hits, 0)
    };
  }
}

// Global intelligent cache instances
export const apiCache = new IntelligentCache(50, 300000); // 5 min for API calls
export const imageCache = new IntelligentCache(100, 3600000); // 1 hour for images
export const componentCache = new IntelligentCache(25, 600000); // 10 min for component data

// Advanced lazy loading hook with intersection observer
export const useAdvancedLazyLoading = <T>(
  loadFunction: () => Promise<T>,
  options: {
    threshold?: number;
    rootMargin?: string;
    triggerOnce?: boolean;
    enabled?: boolean;
    cacheKey?: string;
  } = {}
) => {
  const {
    threshold = 0.1,
    rootMargin = '50px',
    triggerOnce = true,
    enabled = true,
    cacheKey
  } = options;

  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const [inView, setInView] = useState(false);
  
  const targetRef = useRef<HTMLElement | null>(null);
  const hasLoadedRef = useRef(false);

  const load = useCallback(async () => {
    if (hasLoadedRef.current && triggerOnce) return;
    if (!enabled) return;

    try {
      setLoading(true);
      setError(null);

      // Check cache first
      if (cacheKey) {
        const cached = apiCache.get(cacheKey);
        if (cached) {
          setData(cached);
          setLoading(false);
          hasLoadedRef.current = true;
          return;
        }
      }

      const startTime = performance.now();
      const result = await loadFunction();
      const loadTime = performance.now() - startTime;

        // Log performance
        if (performanceMonitor && 'recordMetric' in performanceMonitor) {
          (performanceMonitor as { recordMetric: (name: string, value: number, tags?: Record<string, unknown>) => void }).recordMetric('lazy_load_time', loadTime, {
            cacheKey: cacheKey || 'unknown',
            cached: false
          });
        }

      // Cache the result
      if (cacheKey) {
        apiCache.set(cacheKey, result);
      }

      setData(result);
      hasLoadedRef.current = true;
      
    } catch (err) {
      const error = err instanceof Error ? err : new Error('Unknown error');
      setError(error);
      logger.error('Lazy loading failed', { error: error.message, cacheKey });
    } finally {
      setLoading(false);
    }
  }, [loadFunction, triggerOnce, enabled, cacheKey]);

  useEffect(() => {
    if (!targetRef.current || !enabled) return;

    const observer = new IntersectionObserver(
      ([entry]) => {
        const isIntersecting = entry.isIntersecting;
        setInView(isIntersecting);
        
        if (isIntersecting) {
          load();
        }
      },
      { threshold, rootMargin }
    );

    observer.observe(targetRef.current);

    return () => observer.disconnect();
  }, [load, threshold, rootMargin, enabled]);

  return {
    targetRef,
    data,
    loading,
    error,
    inView,
    reload: load
  };
};

// Intelligent preloading hook
export const useIntelligentPreloader = () => {
  const [preloadQueue, setPreloadQueue] = useState<Array<{
    key: string;
    loader: () => Promise<unknown>;
    priority: 'high' | 'medium' | 'low';
  }>>([]);

  const [isPreloading, setIsPreloading] = useState(false);
  const processingRef = useRef(false);

  const addToQueue = useCallback((
    key: string,
    loader: () => Promise<unknown>,
    priority: 'high' | 'medium' | 'low' = 'medium'
  ) => {
    setPreloadQueue(prev => {
      // Don't add duplicates
      if (prev.some(item => item.key === key)) return prev;
      
      const newItem = { key, loader, priority };
      
      // Insert based on priority
      const highPriorityIndex = prev.findIndex(item => item.priority !== 'high');
      const mediumPriorityIndex = prev.findIndex(item => item.priority === 'low');
      
      if (priority === 'high') {
        return [newItem, ...prev];
      } else if (priority === 'medium') {
        const insertIndex = highPriorityIndex === -1 ? prev.length : 
          (mediumPriorityIndex === -1 ? prev.length : mediumPriorityIndex);
        return [...prev.slice(0, insertIndex), newItem, ...prev.slice(insertIndex)];
      } else {
        return [...prev, newItem];
      }
    });
  }, []);

  const processQueue = useCallback(async () => {
    if (processingRef.current || preloadQueue.length === 0) return;
    
    processingRef.current = true;
    setIsPreloading(true);

    try {
        // Process items in batches based on network quality
        const connection = navigator.connection;
        const batchSize = connection ?
          connection.effectiveType === '4g' ? 3 : 1 : 2;

      while (preloadQueue.length > 0) {
        const batch = preloadQueue.slice(0, batchSize);
        setPreloadQueue(prev => prev.slice(batchSize));

        await Promise.allSettled(
          batch.map(async ({ key, loader }) => {
            try {
              // Check if already cached
              if (apiCache.get(key)) return;
              
              const result = await loader();
              apiCache.set(key, result);
              
              logger.info('Preloaded successfully', { key });
            } catch (error) {
              logger.warn('Preload failed', { key, error });
            }
          })
        );

        // Small delay between batches to avoid overwhelming the system
        await new Promise(resolve => setTimeout(resolve, 100));
      }
    } finally {
      processingRef.current = false;
      setIsPreloading(false);
    }
  }, [preloadQueue]);

  useEffect(() => {
    // Start processing when queue has items and we're idle
    if (preloadQueue.length > 0 && !processingRef.current) {
      // Use requestIdleCallback if available, otherwise setTimeout
      if (window.requestIdleCallback) {
        window.requestIdleCallback(processQueue);
      } else {
        setTimeout(processQueue, 100);
      }
    }
  }, [preloadQueue, processQueue]);

  return {
    addToQueue,
    queueLength: preloadQueue.length,
    isPreloading
  };
};

// Performance-aware data fetching hook
export const usePerformantFetch = <T>(
  url: string | null,
  options: {
    enabled?: boolean;
    cacheKey?: string;
    cacheTTL?: number;
    retries?: number;
    timeout?: number;
    onSuccess?: (data: T) => void;
    onError?: (error: Error) => void;
  } = {}
) => {
  const {
    enabled = true,
    cacheKey = url || '',
    cacheTTL = 300000, // 5 minutes
    retries = 3,
    timeout = 10000,
    onSuccess,
    onError
  } = options;

  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  
  const abortControllerRef = useRef<AbortController | null>(null);

  const fetchData = useCallback(async () => {
    if (!url || !enabled) return;

    // Check cache first
    const cached = apiCache.get(cacheKey);
    if (cached) {
      setData(cached);
      onSuccess?.(cached);
      return;
    }

    setLoading(true);
    setError(null);

    // Cancel previous request
    abortControllerRef.current?.abort();
    abortControllerRef.current = new AbortController();

    let attempt = 0;
    const startTime = performance.now();

    while (attempt < retries) {
      try {
        const timeoutId = setTimeout(() => {
          abortControllerRef.current?.abort();
        }, timeout);

        const response = await fetch(url, {
          signal: abortControllerRef.current.signal,
          headers: {
            'Cache-Control': 'max-age=300', // 5 minutes browser cache
          }
        });

        clearTimeout(timeoutId);

        if (!response.ok) {
          throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        const result = await response.json();
        const fetchTime = performance.now() - startTime;

        // Cache the result
        apiCache.set(cacheKey, result, cacheTTL);

        // Log performance
        if (performanceMonitor && 'recordMetric' in performanceMonitor) {
          (performanceMonitor as { recordMetric: (name: string, value: number, tags?: Record<string, unknown>) => void }).recordMetric('fetch_time', fetchTime, {
            url,
            cached: false,
            attempt: attempt + 1
          });
        }

        setData(result);
        onSuccess?.(result);
        break;

      } catch (err) {
        attempt++;
        
        if (err instanceof Error && err.name === 'AbortError') {
          break; // Don't retry aborted requests
        }

        if (attempt >= retries) {
          const error = err instanceof Error ? err : new Error('Fetch failed');
          setError(error);
          onError?.(error);
          
          logger.error('Fetch failed after retries', {
            url,
            attempts: retries,
            error: error.message
          });
        } else {
          // Exponential backoff
          await new Promise(resolve => setTimeout(resolve, Math.pow(2, attempt) * 1000));
        }
      }
    }

    setLoading(false);
  }, [url, enabled, cacheKey, cacheTTL, retries, timeout, onSuccess, onError]);

  useEffect(() => {
    fetchData();

    return () => {
      abortControllerRef.current?.abort();
    };
  }, [fetchData]);

  const refetch = useCallback(() => {
    // Clear cache and refetch
    apiCache.set(cacheKey, null, 0);
    fetchData();
  }, [cacheKey, fetchData]);

  return {
    data,
    loading,
    error,
    refetch
  };
};

// Memory management utilities
export const useMemoryManagement = () => {
  const [memoryUsage, setMemoryUsage] = useState(0);
  const [isLowMemory, setIsLowMemory] = useState(false);

  useEffect(() => {
    const checkMemoryUsage = () => {
      if ('memory' in performance) {
        const memory = performance.memory;
        const usedMB = memory.usedJSHeapSize / 1048576;
        const limitMB = memory.jsHeapSizeLimit / 1048576;
        
        setMemoryUsage(usedMB);
        setIsLowMemory(usedMB / limitMB > 0.8); // Warning at 80% usage
        
        if (isLowMemory) {
          logger.warn('High memory usage detected', { 
            used: usedMB, 
            limit: limitMB,
            percentage: Math.round((usedMB / limitMB) * 100)
          });
          
          // Clear caches when memory is low
          apiCache.clear();
          imageCache.clear();
          componentCache.clear();
        }
      }
    };

    const interval = setInterval(checkMemoryUsage, 10000); // Check every 10 seconds
    checkMemoryUsage(); // Initial check

    return () => clearInterval(interval);
  }, [isLowMemory]);

  const clearCaches = useCallback(() => {
    apiCache.clear();
    imageCache.clear();
    componentCache.clear();
    logger.info('Caches cleared manually');
  }, []);

  return {
    memoryUsage,
    isLowMemory,
    clearCaches
  };
};