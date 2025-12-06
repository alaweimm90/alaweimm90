import { useState, useCallback, useRef, useMemo } from 'react';
import { useDebounce } from 'use-debounce';

// Optimized state hook with debouncing and batching
export const useOptimizedState = <T>(
  initialState: T,
  debounceMs = 300
) => {
  const [state, setState] = useState<T>(initialState);
  const [debouncedState] = useDebounce(state, debounceMs);
  const updateCount = useRef(0);

  const optimizedSetState = useCallback((newState: T | ((prev: T) => T)) => {
    updateCount.current += 1;
    setState(newState);
  }, []);

  const batchedUpdate = useCallback((updates: Partial<T>) => {
    setState(prev => ({ ...prev, ...updates }));
  }, []);

  return {
    state,
    debouncedState,
    setState: optimizedSetState,
    batchedUpdate,
    updateCount: updateCount.current
  };
};

// Memoized computation hook
export const useMemoizedComputation = <T, D extends readonly unknown[]>(
  computation: () => T,
  deps: D,
  options: { cacheSize?: number } = {}
) => {
  const { cacheSize = 10 } = options;
  const cache = useRef<Map<string, T>>(new Map());

  return useMemo(() => {
    const key = JSON.stringify(deps);
    
    if (cache.current.has(key)) {
      return cache.current.get(key)!;
    }

    const result = computation();
    
    // Implement LRU cache
    if (cache.current.size >= cacheSize) {
      const firstKey = cache.current.keys().next().value;
      cache.current.delete(firstKey);
    }

    cache.current.set(key, result);
    return result;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [cacheSize, computation, ...deps]);
};

// Optimized event handler with automatic cleanup
export const useOptimizedEventHandler = <T extends (...args: any[]) => void>(
  handler: T,
  throttleMs = 16 // 60fps
) => {
  const lastCall = useRef(0);
  const timeoutRef = useRef<NodeJS.Timeout>();

  const throttledHandler = useCallback((...args: Parameters<T>) => {
    const now = Date.now();
    
    if (now - lastCall.current >= throttleMs) {
      lastCall.current = now;
      handler(...args);
    } else {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
      
      timeoutRef.current = setTimeout(() => {
        lastCall.current = Date.now();
        handler(...args);
      }, throttleMs - (now - lastCall.current));
    }
  }, [handler, throttleMs]);

  // Cleanup on unmount
  return throttledHandler;
};

// Batch state updates for better performance
export const useBatchedUpdates = () => {
  const pendingUpdates = useRef<Array<() => void>>([]);
  const isScheduled = useRef(false);

  const scheduleUpdate = useCallback((updateFn: () => void) => {
    pendingUpdates.current.push(updateFn);
    
    if (!isScheduled.current) {
      isScheduled.current = true;
      
      // Use scheduler API if available, otherwise fallback to setTimeout
      const schedule = (callback: () => void) => {
        if ('scheduler' in window && window.scheduler && 'postTask' in window.scheduler) {
          window.scheduler.postTask(callback, { priority: 'user-blocking' });
        } else {
          setTimeout(callback, 0);
        }
      };

      schedule(() => {
        const updates = pendingUpdates.current.splice(0);
        isScheduled.current = false;
        
        updates.forEach(update => update());
      });
    }
  }, []);

  return { scheduleUpdate };
};