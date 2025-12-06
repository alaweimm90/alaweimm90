import { useEffect, useCallback, useRef } from 'react';
import { trackError, usePerformanceTracking } from '../utils/monitoring';
import { Analytics } from '../utils/analytics';

interface PerformanceConfig {
  trackRender?: boolean;
  trackInteractions?: boolean;
  trackAsyncOperations?: boolean;
  slowRenderThreshold?: number;
}

export const usePerformanceMonitor = (
  componentName: string, 
  config: PerformanceConfig = {}
) => {
  const {
    trackRender = true,
    trackInteractions = true,
    trackAsyncOperations = true,
    slowRenderThreshold = 16 // 60fps = ~16ms per frame
  } = config;

  const { trackOperation } = usePerformanceTracking(componentName);
  const renderCountRef = useRef(0);
  const lastRenderTimeRef = useRef(0);

  // Track render performance
  useEffect(() => {
    if (!trackRender) return;

    renderCountRef.current += 1;
    const renderTime = performance.now() - lastRenderTimeRef.current;
    
    if (renderCountRef.current > 1 && renderTime > slowRenderThreshold) {
      console.warn(`[Performance] Slow render in ${componentName}: ${renderTime.toFixed(2)}ms`);
    }
    
    lastRenderTimeRef.current = performance.now();
  });

  // Track user interactions with performance impact
  const trackInteraction = useCallback(<T>(actionName: string, fn: () => T): T => {
    if (!trackInteractions) return fn();

    return trackOperation(`interaction_${actionName}`, fn);
  }, [trackOperation, trackInteractions]);

  // Track async operations
  const trackAsyncOperation = useCallback(async <T>(operationName: string, asyncFn: () => Promise<T>): Promise<T> => {
    if (!trackAsyncOperations) return asyncFn();
    
    try {
      return await trackOperation(`async_${operationName}`, asyncFn);
    } catch (error) {
      trackError(error as Error, {
        component: componentName,
        action: operationName
      });
      throw error;
    }
  }, [trackOperation, trackAsyncOperations, componentName]);

  // Monitor memory usage (for development)
  const checkMemoryUsage = useCallback(() => {
    if (process.env.NODE_ENV !== 'development') return;

    const memory = performance.memory;
    if (memory) {
      const used = Math.round(memory.usedJSHeapSize / 1048576 * 100) / 100;
      const total = Math.round(memory.totalJSHeapSize / 1048576 * 100) / 100;

      if (used > 50) { // Alert if using more than 50MB
        console.warn(`[Memory] High usage in ${componentName}: ${used}MB / ${total}MB`);
      }
    }
  }, [componentName]);

  // Check memory usage periodically in development
  useEffect(() => {
    if (process.env.NODE_ENV !== 'development') return;
    
    const interval = setInterval(checkMemoryUsage, 30000); // Every 30 seconds
    return () => clearInterval(interval);
  }, [checkMemoryUsage]);

  return {
    trackInteraction,
    trackAsyncOperation,
    renderCount: renderCountRef.current,
    checkMemoryUsage
  };
};

// Hook for monitoring Core Web Vitals - simplified for production readiness
export const useCoreWebVitals = () => {
  useEffect(() => {
    if (typeof window === 'undefined') return;

    // Simple performance observer for core metrics
    try {
      // Check if PerformanceObserver is available
      if (typeof PerformanceObserver === 'undefined') {
        console.warn('[Performance] PerformanceObserver not available');
        return;
      }

      const isProd = (import.meta as any)?.env?.MODE === 'production';
      const observer = new PerformanceObserver((list) => {
        for (const entry of list.getEntries()) {
          const value = (entry as any).value ?? entry.duration ?? entry.startTime;
          if (isProd) {
            Analytics.trackCustom('web_vital', {
              name: entry.name,
              value: typeof value === 'number' ? Number(value.toFixed(2)) : value
            });
          } else {
            console.log(`[Performance] ${entry.name}: ${typeof value === 'number' ? value.toFixed(2) : value}`);
          }
        }
      });
      
      // Observe navigation and paint timing
      observer.observe({ 
        entryTypes: ['navigation', 'paint', 'largest-contentful-paint'] 
      });
      
      return () => observer.disconnect();
    } catch (error) {
      console.warn('Performance observer not supported:', error);
    }
  }, []);
};

// Hook for monitoring API performance
export const useAPIPerformanceMonitor = () => {
  const trackAPICall = useCallback(async (
    endpoint: string,
    method: string,
    requestFn: () => Promise<any>
  ) => {
    const startTime = performance.now();
    
    try {
      const result = await requestFn();
      const duration = performance.now() - startTime;
      
      console.log(`[API] ${method} ${endpoint}: ${duration.toFixed(2)}ms`);
      
      // Track slow API calls
      if (duration > 2000) {
        console.warn(`[API] Slow request: ${method} ${endpoint} took ${duration.toFixed(2)}ms`);
      }
      
      return result;
    } catch (error) {
      const duration = performance.now() - startTime;
      trackError(error as Error, {
        component: 'API',
        action: `${method} ${endpoint}`,
        metadata: { duration: duration.toFixed(2) }
      });
      throw error;
    }
  }, []);

  return { trackAPICall };
};