import { useEffect, useCallback } from 'react';
import * as Sentry from '@sentry/react';

/**
 * Track an error with context
 */
export const trackError = (error: Error, context?: Record<string, any>) => {
  console.error('[MONITOR] Error:', {
    message: error.message,
    stack: error.stack,
    context,
    timestamp: new Date().toISOString(),
    url: window.location.href
  });

  // In production, send to Sentry
  if (import.meta.env.MODE === 'production' && import.meta.env.VITE_SENTRY_DSN) {
    Sentry.captureException(error, {
      extra: context,
      tags: {
        component: context?.component || 'unknown',
        severity: context?.severity || 'error',
      },
    });
  }
};

/**
 * Hook for tracking performance of operations
 */
export const usePerformanceTracking = (scope: string) => {
  const trackOperation = useCallback(<T>(operationName: string, fn: () => T): T => {
    const startTime = performance.now();
    try {
      const result = fn();
      const duration = performance.now() - startTime;

      if (duration > 100) {
        console.warn(`[MONITOR] Slow operation: ${scope}/${operationName} took ${duration.toFixed(2)}ms`);
      }

      return result;
    } catch (error) {
      const duration = performance.now() - startTime;
      trackError(error as Error, {
        scope,
        operation: operationName,
        duration: duration.toFixed(2)
      });
      throw error;
    }
  }, [scope]);

  return { trackOperation };
};

// Initialize basic error tracking and performance monitoring
export const initializeMonitoring = () => {
  // Initialize Sentry for production error tracking
  if (import.meta.env.VITE_SENTRY_DSN) {
    Sentry.init({
      dsn: import.meta.env.VITE_SENTRY_DSN,
      environment: import.meta.env.MODE,
      tracesSampleRate: import.meta.env.MODE === 'production' ? 0.1 : 1.0,
      replaysSessionSampleRate: 0.1,
      replaysOnErrorSampleRate: 1.0,
      integrations: [
        Sentry.browserTracingIntegration(),
        Sentry.replayIntegration({
          maskAllText: false,
          blockAllMedia: false,
        }),
      ],
      beforeSend(event, hint) {
        // Don't send errors in development
        if (import.meta.env.MODE === 'development') {
          console.log('[SENTRY] Would send:', event);
          return null;
        }
        return event;
      },
    });
    console.log('[MONITOR] Sentry initialized');
  }

  // Basic error tracking
  window.addEventListener('error', (event) => {
    console.error('[MONITOR] JavaScript Error:', {
      message: event.message,
      filename: event.filename,
      lineno: event.lineno,
      colno: event.colno,
      stack: event.error?.stack,
      timestamp: new Date().toISOString(),
      url: window.location.href,
      userAgent: navigator.userAgent
    });
  });

  // Promise rejection tracking
  window.addEventListener('unhandledrejection', (event) => {
    console.error('[MONITOR] Unhandled Promise Rejection:', {
      reason: event.reason,
      timestamp: new Date().toISOString(),
      url: window.location.href
    });
  });

  // Basic performance tracking
  if ('performance' in window) {
    const observer = new PerformanceObserver((list) => {
      list.getEntries().forEach((entry) => {
        if (entry.entryType === 'navigation') {
          console.log('[MONITOR] Page Load Time:', entry.duration);
        }
      });
    });

    try {
      observer.observe({ entryTypes: ['navigation'] });
    } catch (e) {
      // Fallback for browsers that don't support this
      console.warn('[MONITOR] Performance observer not supported');
    }
  }
};

// Component performance tracking hook
export const useComponentPerformance = (componentName: string) => {
  useEffect(() => {
    const startTime = performance.now();
    
    return () => {
      const endTime = performance.now();
      const renderTime = endTime - startTime;
      
      if (renderTime > 100) { // Log slow renders
        console.warn(`[MONITOR] Slow render detected: ${componentName} took ${renderTime.toFixed(2)}ms`);
      }
    };
  }, [componentName]);
};

// Route navigation error tracking
export const trackNavigationError = (attemptedPath: string, error?: string) => {
  console.error('[MONITOR] Navigation Error:', {
    attemptedPath,
    error,
    currentPath: window.location.pathname,
    timestamp: new Date().toISOString(),
    userAgent: navigator.userAgent
  });
};

// Performance metrics tracking
export const trackPerformanceMetric = (name: string, value: number, context?: Record<string, unknown>) => {
  console.log('[MONITOR] Performance Metric:', {
    name,
    value,
    context,
    timestamp: new Date().toISOString(),
    url: window.location.href
  });
};

// Initialize monitoring when the module loads
if (typeof window !== 'undefined') {
  initializeMonitoring();
}