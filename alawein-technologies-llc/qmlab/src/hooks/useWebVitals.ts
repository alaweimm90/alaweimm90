import { useEffect, useRef } from 'react';
import { trackQuantumEvents } from '@/lib/analytics';
import { ExtendedPerformanceEventTiming } from '@/lib/simple-stubs';
import { logger } from '@/lib/logger';

interface WebVitalsMetrics {
  lcp?: number;
  fid?: number;
  cls?: number;
  fcp?: number;
  ttfb?: number;
  inp?: number;
}

export const useWebVitals = () => {
  const metricsRef = useRef<WebVitalsMetrics>({});
  const observerRef = useRef<PerformanceObserver | null>(null);

  useEffect(() => {
    // Track Core Web Vitals
    const trackVital = (name: string, value: number, id: string) => {
      metricsRef.current = {
        ...metricsRef.current,
        [name.toLowerCase()]: value
      };

      // Send to analytics
      trackQuantumEvents.pageLoadTime(value, {
        metric_name: name,
        metric_id: id,
        ...metricsRef.current
      });
    };

    // LCP Observer
    if ('PerformanceObserver' in window) {
      try {
        const lcpObserver = new PerformanceObserver((entryList) => {
          const entries = entryList.getEntries();
          const lastEntry = entries[entries.length - 1] as ExtendedPerformanceEventTiming;
          if (lastEntry) {
            trackVital('LCP', lastEntry.startTime, lastEntry.id || 'lcp');
          }
        });
        lcpObserver.observe({ entryTypes: ['largest-contentful-paint'] });

        // CLS Observer
        const clsObserver = new PerformanceObserver((entryList) => {
          let clsValue = 0;
          for (const entry of entryList.getEntries()) {
            if (!(entry as any).hadRecentInput) {
              clsValue += (entry as any).value;
            }
          }
          if (clsValue > 0) {
            trackVital('CLS', clsValue, 'cls');
          }
        });
        clsObserver.observe({ entryTypes: ['layout-shift'] });

        // FID/INP Observer
        const responsiveObserver = new PerformanceObserver((entryList) => {
          for (const entry of entryList.getEntries()) {
            const eventEntry = entry as ExtendedPerformanceEventTiming;
            if (eventEntry.name === 'first-input') {
              trackVital('FID', eventEntry.processingStart - eventEntry.startTime, 'fid');
            }
            // Track INP (Interaction to Next Paint)
            if (eventEntry.interactionId) {
              const inp = eventEntry.processingStart - eventEntry.startTime;
              trackVital('INP', inp, `inp-${eventEntry.interactionId}`);
            }
          }
        });
        
        try {
          responsiveObserver.observe({ entryTypes: ['first-input', 'event'] });
        } catch {
          // Fallback for browsers without 'event' entry type
          responsiveObserver.observe({ entryTypes: ['first-input'] });
        }

        observerRef.current = lcpObserver;

        // Navigation timing
        window.addEventListener('load', () => {
          const navigation = performance.getEntriesByType('navigation')[0] as PerformanceNavigationTiming;
          if (navigation) {
            const ttfb = navigation.responseStart - navigation.requestStart;
            trackVital('TTFB', ttfb, 'ttfb');
          }

          // FCP
          const paintEntries = performance.getEntriesByType('paint');
          const fcpEntry = paintEntries.find(entry => entry.name === 'first-contentful-paint');
          if (fcpEntry) {
            trackVital('FCP', fcpEntry.startTime, 'fcp');
          }
        });

      } catch (error) {
        logger.warn('Performance Observer not fully supported', { error });
      }
    }

    return () => {
      if (observerRef.current) {
        observerRef.current.disconnect();
      }
    };
  }, []);

  return metricsRef.current;
};

// Real User Monitoring for quantum-specific interactions
export const useQuantumMetrics = () => {
  const trackQuantumPerformance = (operation: string, startTime: number, metadata?: Record<string, any>) => {
    const duration = performance.now() - startTime;
    
    // Track quantum operation performance
    trackQuantumEvents.componentLoad(`quantum-${operation}`, duration);
    
    // Log performance data for analysis
    logger.debug(`Quantum ${operation} performance`, { duration, operation, ...metadata });
    
    return duration;
  };

  const measureQuantumOperation = async <T>(
    operation: string,
    fn: () => Promise<T> | T,
    metadata?: Record<string, any>
  ): Promise<T> => {
    const startTime = performance.now();
    
    try {
      const result = await fn();
      trackQuantumPerformance(operation, startTime, { ...metadata, success: true });
      return result;
    } catch (error) {
      trackQuantumPerformance(operation, startTime, { ...metadata, success: false, error: (error as Error).message });
      throw error;
    }
  };

  return {
    trackQuantumPerformance,
    measureQuantumOperation
  };
};