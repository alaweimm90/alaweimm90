import { useEffect } from 'react';

interface PerformanceMetrics {
  lcp?: number;
  fid?: number;
  cls?: number;
  fcp?: number;
  ttfb?: number;
}

interface LayoutShiftEntry extends PerformanceEntry {
  value: number;
  hadRecentInput: boolean;
}

interface FirstInputEntry extends PerformanceEventTiming {
  processingStart?: number;
}

export const PerformanceMonitor = () => {
  useEffect(() => {
    // Web Vitals monitoring
    const observer = new PerformanceObserver((list) => {
      list.getEntries().forEach((entry) => {
        const metrics: PerformanceMetrics = {};
        
        switch (entry.entryType) {
          case 'largest-contentful-paint':
            metrics.lcp = entry.startTime;
            break;
            case 'first-input': {
              const e = entry as FirstInputEntry;
              metrics.fid = (e.processingStart ?? e.startTime) - e.startTime;
              break;
            }
          case 'layout-shift': {
            const layoutEntry = entry as LayoutShiftEntry;
            if (!layoutEntry.hadRecentInput) {
              metrics.cls = layoutEntry.value;
            }
            break;
          }
        }
        
        // Send metrics to monitoring service
        if (Object.keys(metrics).length > 0) {
          sendMetrics(metrics);
        }
      });
    });
    
    // Observe different entry types
    try {
      observer.observe({ entryTypes: ['largest-contentful-paint', 'first-input', 'layout-shift'] });
    } catch (e) {
      // Fallback for unsupported browsers
      console.warn('Performance Observer not supported');
    }
    
    // Navigation timing
    window.addEventListener('load', () => {
      const navigation = performance.getEntriesByType('navigation')[0] as PerformanceNavigationTiming;
      const metrics: PerformanceMetrics = {
        ttfb: navigation.responseStart - navigation.requestStart,
        fcp: performance.getEntriesByName('first-contentful-paint')[0]?.startTime
      };
      
      sendMetrics(metrics);
    });
    
    return () => {
      observer.disconnect();
    };
  }, []);
  
  const sendMetrics = async (metrics: PerformanceMetrics) => {
    try {
      // In production, send to analytics service
      await fetch('/api/metrics', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ...metrics,
          timestamp: Date.now(),
          userAgent: navigator.userAgent,
          url: window.location.href
        })
      });
    } catch (error) {
      console.warn('Failed to send performance metrics:', error);
    }
  };
  
  return null; // This is a monitoring component, no UI
};

// Hook for component-level performance monitoring
export const usePerformanceTracking = (componentName: string) => {
  useEffect(() => {
    const startTime = performance.now();
    
    return () => {
      const endTime = performance.now();
      const renderTime = endTime - startTime;
      
      // Track slow components (>100ms)
      if (renderTime > 100) {
        console.warn(`Slow component render: ${componentName} took ${renderTime.toFixed(2)}ms`);
        
        // Send slow render metric
        fetch('/api/metrics', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            type: 'slow-render',
            component: componentName,
            renderTime,
            timestamp: Date.now()
          })
        }).catch(() => {}); // Silent fail
      }
    };
  }, [componentName]);
};
