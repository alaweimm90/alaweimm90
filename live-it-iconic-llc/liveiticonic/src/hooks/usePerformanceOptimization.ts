/**
 * Performance Optimization Hooks
 * Utilities for improving React component performance
 * Helps achieve INP target by reducing interaction latency
 */

import { useCallback, useRef, useEffect, useMemo, DependencyList } from 'react';

/**
 * Debounce hook for expensive operations
 * Useful for search inputs, resize handlers, etc.
 * Helps improve INP by debouncing rapid interactions
 */
export function useDebounce<T extends (...args: unknown[]) => unknown>(
  callback: T,
  delay: number = 300,
  deps: DependencyList = []
) {
  const timeoutRef = useRef<NodeJS.Timeout>();

  const debouncedCallback = useCallback(
    (...args: Parameters<T>) => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }

      timeoutRef.current = setTimeout(() => {
        callback(...args);
      }, delay);
    },
    [callback, delay, ...deps]
  );

  useEffect(() => {
    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
    };
  }, []);

  return debouncedCallback as T;
}

/**
 * Throttle hook for frequent event handlers
 * Useful for scroll handlers, window resize, etc.
 * Prevents excessive re-renders and calculations
 */
export function useThrottle<T extends (...args: unknown[]) => unknown>(
  callback: T,
  delay: number = 300,
  deps: DependencyList = []
) {
  const timeoutRef = useRef<NodeJS.Timeout>();
  const lastRunRef = useRef<number>(0);

  useEffect(() => {
    lastRunRef.current = Date.now();
  }, []);

  const throttledCallback = useCallback(
    (...args: Parameters<T>) => {
      const now = Date.now();
      const timeSinceLastRun = now - lastRunRef.current;

      if (timeSinceLastRun >= delay) {
        callback(...args);
        lastRunRef.current = now;
      } else {
        if (timeoutRef.current) {
          clearTimeout(timeoutRef.current);
        }

        timeoutRef.current = setTimeout(() => {
          callback(...args);
          lastRunRef.current = Date.now();
        }, delay - timeSinceLastRun);
      }
    },
    [callback, delay, ...deps]
  );

  useEffect(() => {
    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
    };
  }, []);

  return throttledCallback as T;
}

/**
 * Request idle callback hook
 * Schedules work for when the browser is idle
 * Helps prevent blocking the main thread
 */
export function useIdleCallback(
  callback: () => void,
  options?: IdleRequestOptions,
  deps: DependencyList = []
) {
  useEffect(() => {
    if (!('requestIdleCallback' in window)) {
      // Fallback to setTimeout for browsers that don't support requestIdleCallback
      const timeoutId = setTimeout(callback, 0);
      return () => clearTimeout(timeoutId);
    }
    const w = window as typeof window & {
      requestIdleCallback?: (cb: IdleRequestCallback, opts?: IdleRequestOptions) => number;
      cancelIdleCallback?: (id: number) => void;
    };
    const id = w.requestIdleCallback!(callback, options);
    return () => w.cancelIdleCallback?.(id);
  }, deps);
}

/**
 * Request animation frame hook
 * Schedules work for the next animation frame
 * Useful for smooth animations without blocking main thread
 */
export function useAnimationFrame(
  callback: (deltaTime: number) => void,
  enabled: boolean = true,
  deps: DependencyList = []
) {
  const rafRef = useRef<number>();
  const lastTimeRef = useRef<number>(0);

  useEffect(() => {
    if (!enabled) return;

    const animate = (currentTime: number) => {
      const deltaTime = currentTime - lastTimeRef.current;
      lastTimeRef.current = currentTime;
      callback(deltaTime);
      rafRef.current = requestAnimationFrame(animate);
    };

    rafRef.current = requestAnimationFrame(animate);

    return () => {
      if (rafRef.current) {
        cancelAnimationFrame(rafRef.current);
      }
    };
  }, [enabled, callback, ...deps]);
}

/**
 * Intersection Observer hook
 * For lazy loading images and components
 * Improves LCP by deferring below-the-fold content
 */
export function useIntersectionObserver(
  ref: React.RefObject<HTMLElement>,
  options?: IntersectionObserverInit & { onChange?: (isVisible: boolean) => void }
) {
  const [isVisible, setIsVisible] = React.useState(false);
  const { onChange, ...observerOptions } = options || {};

  useEffect(() => {
    if (!ref.current) return;

    const observer = new IntersectionObserver(([entry]) => {
      setIsVisible(entry.isIntersecting);
      onChange?.(entry.isIntersecting);
    }, observerOptions);

    observer.observe(ref.current);

    return () => {
      observer.disconnect();
    };
  }, [ref, onChange, observerOptions]);

  return isVisible;
}

/**
 * Use memo callback with automatic deps cleanup
 * Helps prevent unnecessary re-renders of list items
 */
export function useMemoCallback<T extends (...args: unknown[]) => unknown>(
  callback: T,
  deps: DependencyList
) {
  return useCallback(callback, deps);
}

/**
 * Use computed value that only updates when deps change
 * Similar to useMemo but with cleaner API
 */
export function useComputed<T>(fn: () => T, deps: DependencyList) {
  return useMemo(fn, deps);
}

/**
 * Performance measurement hook
 * Logs component render time in development
 */
export function useRenderTime(componentName: string, enabled: boolean = import.meta.env.DEV) {
  const renderStartTime = useRef<number>(0);

  useEffect(() => {
    if (!enabled) return;

    renderStartTime.current = performance.now();

    return () => {
      const renderTime = performance.now() - renderStartTime.current;
      if (renderTime > 16.67) {
        // Warn if render takes longer than one frame (60fps)
        console.warn(
          `[Performance] ${componentName} took ${renderTime.toFixed(2)}ms to render (exceeds 16.67ms frame budget)`
        );
      }
    };
  }, [componentName, enabled]);
}

/**
 * Async data fetching hook with caching
 * Prevents unnecessary re-renders and redundant fetches
 */
export function useAsyncData<T>(
  asyncFn: () => Promise<T>,
  deps: DependencyList,
  cacheKey?: string
) {
  const [data, setData] = React.useState<T | null>(null);
  const [loading, setLoading] = React.useState(true);
  const [error, setError] = React.useState<Error | null>(null);
  const cacheRef = useRef<{ key: string; data: T; timestamp: number }[]>([]);

  useEffect(() => {
    let mounted = true;

    // Check cache
    const cached = cacheKey
      ? cacheRef.current.find(
          (c) => c.key === cacheKey && Date.now() - c.timestamp < 5 * 60 * 1000
        )
      : null;

    if (cached) {
      queueMicrotask(() => {
        if (!mounted) return;
        setData(cached.data);
        setLoading(false);
      });
      return;
    }

    queueMicrotask(() => {
      if (!mounted) return;
      setLoading(true);
    });
    asyncFn()
      .then((result) => {
        if (!mounted) return;
        setData(result);
        setError(null);

        // Cache result
        if (cacheKey) {
          cacheRef.current = [
            ...cacheRef.current.filter((c) => c.key !== cacheKey),
            { key: cacheKey, data: result, timestamp: Date.now() },
          ];
        }
      })
      .catch((err) => {
        if (!mounted) return;
        setError(err);
        setData(null);
      })
      .finally(() => {
        if (mounted) {
          setLoading(false);
        }
      });

    return () => {
      mounted = false;
    };
  }, [asyncFn, cacheKey, ...deps]);

  return { data, loading, error };
}

/**
 * Measure interaction latency
 * Helps track INP (Interaction to Next Paint) metric
 */
export function measureInteraction(label: string): () => void {
  const startTime = performance.now();

  return () => {
    const duration = performance.now() - startTime;
    console.log(`[Interaction] ${label}: ${duration.toFixed(2)}ms`);

    // Warn if interaction takes too long (200ms target)
    if (duration > 200) {
      console.warn(`[Performance] Interaction "${label}" took ${duration.toFixed(2)}ms (target: <200ms)`);
    }
  };
}
