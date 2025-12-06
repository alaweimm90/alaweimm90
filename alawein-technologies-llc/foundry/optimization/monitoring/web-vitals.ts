/**
 * Core Web Vitals Monitoring
 * Tracks and reports performance metrics for all applications
 */

import { onCLS, onFCP, onFID, onLCP, onTTFB, onINP, Metric } from 'web-vitals';

export interface PerformanceMetric {
  name: string;
  value: number;
  rating: 'good' | 'needs-improvement' | 'poor';
  delta: number;
  entries: any[];
  id: string;
  navigationType: string;
  timestamp: number;
  url: string;
  userAgent: string;
  connection?: {
    effectiveType: string;
    downlink: number;
    rtt: number;
  };
}

/**
 * Performance thresholds based on Core Web Vitals
 */
export const PERFORMANCE_THRESHOLDS = {
  LCP: {
    good: 2500,
    needsImprovement: 4000
  },
  FID: {
    good: 100,
    needsImprovement: 300
  },
  CLS: {
    good: 0.1,
    needsImprovement: 0.25
  },
  FCP: {
    good: 1800,
    needsImprovement: 3000
  },
  TTFB: {
    good: 800,
    needsImprovement: 1800
  },
  INP: {
    good: 200,
    needsImprovement: 500
  }
};

/**
 * Get rating based on metric value
 */
function getRating(name: string, value: number): 'good' | 'needs-improvement' | 'poor' {
  const thresholds = PERFORMANCE_THRESHOLDS[name as keyof typeof PERFORMANCE_THRESHOLDS];
  if (!thresholds) return 'good';

  if (value <= thresholds.good) return 'good';
  if (value <= thresholds.needsImprovement) return 'needs-improvement';
  return 'poor';
}

/**
 * Get connection information
 */
function getConnectionInfo() {
  const nav = navigator as any;
  const connection = nav.connection || nav.mozConnection || nav.webkitConnection;

  if (connection) {
    return {
      effectiveType: connection.effectiveType,
      downlink: connection.downlink,
      rtt: connection.rtt
    };
  }

  return undefined;
}

/**
 * Format metric for reporting
 */
function formatMetric(metric: Metric): PerformanceMetric {
  return {
    name: metric.name,
    value: Math.round(metric.value),
    rating: getRating(metric.name, metric.value),
    delta: Math.round(metric.delta),
    entries: metric.entries,
    id: metric.id,
    navigationType: metric.navigationType,
    timestamp: Date.now(),
    url: window.location.href,
    userAgent: navigator.userAgent,
    connection: getConnectionInfo()
  };
}

/**
 * Send metrics to analytics endpoint
 */
async function sendToAnalytics(metrics: PerformanceMetric[]) {
  const endpoint = process.env.NEXT_PUBLIC_ANALYTICS_ENDPOINT || '/api/metrics';

  try {
    await fetch(endpoint, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        metrics,
        session: {
          id: getSessionId(),
          timestamp: Date.now(),
          viewport: {
            width: window.innerWidth,
            height: window.innerHeight
          },
          screen: {
            width: screen.width,
            height: screen.height,
            colorDepth: screen.colorDepth
          }
        }
      }),
    });
  } catch (error) {
    console.error('Failed to send metrics:', error);
    // Store in local storage for retry
    storeMetricsLocally(metrics);
  }
}

/**
 * Get or create session ID
 */
function getSessionId(): string {
  let sessionId = sessionStorage.getItem('performance-session-id');
  if (!sessionId) {
    sessionId = `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
    sessionStorage.setItem('performance-session-id', sessionId);
  }
  return sessionId;
}

/**
 * Store metrics locally for retry
 */
function storeMetricsLocally(metrics: PerformanceMetric[]) {
  try {
    const stored = localStorage.getItem('pending-metrics');
    const pending = stored ? JSON.parse(stored) : [];
    pending.push(...metrics);

    // Keep only last 100 metrics
    const trimmed = pending.slice(-100);
    localStorage.setItem('pending-metrics', JSON.stringify(trimmed));
  } catch (error) {
    console.error('Failed to store metrics locally:', error);
  }
}

/**
 * Retry sending stored metrics
 */
async function retryStoredMetrics() {
  try {
    const stored = localStorage.getItem('pending-metrics');
    if (stored) {
      const metrics = JSON.parse(stored);
      if (metrics.length > 0) {
        await sendToAnalytics(metrics);
        localStorage.removeItem('pending-metrics');
      }
    }
  } catch (error) {
    console.error('Failed to retry stored metrics:', error);
  }
}

/**
 * Custom metrics tracking
 */
export class CustomMetrics {
  private marks: Map<string, number> = new Map();
  private measures: Map<string, number> = new Map();

  /**
   * Mark a point in time
   */
  mark(name: string) {
    this.marks.set(name, performance.now());
    performance.mark(name);
  }

  /**
   * Measure between two marks
   */
  measure(name: string, startMark: string, endMark?: string) {
    const start = this.marks.get(startMark);
    const end = endMark ? this.marks.get(endMark) : performance.now();

    if (start && end) {
      const duration = end - start;
      this.measures.set(name, duration);

      if (endMark) {
        performance.measure(name, startMark, endMark);
      } else {
        performance.measure(name, startMark);
      }

      return duration;
    }

    return null;
  }

  /**
   * Track component render time
   */
  trackComponentRender(componentName: string, renderTime: number) {
    const metric: PerformanceMetric = {
      name: `component-render-${componentName}`,
      value: renderTime,
      rating: renderTime < 16 ? 'good' : renderTime < 50 ? 'needs-improvement' : 'poor',
      delta: 0,
      entries: [],
      id: `${Date.now()}-${Math.random()}`,
      navigationType: 'component',
      timestamp: Date.now(),
      url: window.location.href,
      userAgent: navigator.userAgent,
      connection: getConnectionInfo()
    };

    sendToAnalytics([metric]);
  }

  /**
   * Track API call performance
   */
  trackAPICall(endpoint: string, duration: number, status: number) {
    const metric: PerformanceMetric = {
      name: `api-call`,
      value: duration,
      rating: duration < 200 ? 'good' : duration < 1000 ? 'needs-improvement' : 'poor',
      delta: 0,
      entries: [{ endpoint, status }],
      id: `${Date.now()}-${Math.random()}`,
      navigationType: 'api',
      timestamp: Date.now(),
      url: window.location.href,
      userAgent: navigator.userAgent,
      connection: getConnectionInfo()
    };

    sendToAnalytics([metric]);
  }

  /**
   * Get all measures
   */
  getMeasures() {
    return Array.from(this.measures.entries()).map(([name, value]) => ({
      name,
      value
    }));
  }

  /**
   * Clear all marks and measures
   */
  clear() {
    this.marks.clear();
    this.measures.clear();
  }
}

/**
 * Initialize Web Vitals tracking
 */
export function initWebVitals() {
  const metrics: PerformanceMetric[] = [];
  let metricsBuffer: PerformanceMetric[] = [];
  let bufferTimer: NodeJS.Timeout | null = null;

  // Buffer metrics and send in batches
  const bufferMetric = (metric: PerformanceMetric) => {
    metricsBuffer.push(metric);

    if (bufferTimer) {
      clearTimeout(bufferTimer);
    }

    // Send metrics after 5 seconds of inactivity or when buffer reaches 10 metrics
    if (metricsBuffer.length >= 10) {
      sendToAnalytics(metricsBuffer);
      metricsBuffer = [];
    } else {
      bufferTimer = setTimeout(() => {
        if (metricsBuffer.length > 0) {
          sendToAnalytics(metricsBuffer);
          metricsBuffer = [];
        }
      }, 5000);
    }
  };

  // Core Web Vitals
  onLCP((metric) => {
    const formatted = formatMetric(metric);
    metrics.push(formatted);
    bufferMetric(formatted);
    console.log('LCP:', formatted);
  });

  onFID((metric) => {
    const formatted = formatMetric(metric);
    metrics.push(formatted);
    bufferMetric(formatted);
    console.log('FID:', formatted);
  });

  onCLS((metric) => {
    const formatted = formatMetric(metric);
    metrics.push(formatted);
    bufferMetric(formatted);
    console.log('CLS:', formatted);
  });

  // Additional metrics
  onFCP((metric) => {
    const formatted = formatMetric(metric);
    metrics.push(formatted);
    bufferMetric(formatted);
    console.log('FCP:', formatted);
  });

  onTTFB((metric) => {
    const formatted = formatMetric(metric);
    metrics.push(formatted);
    bufferMetric(formatted);
    console.log('TTFB:', formatted);
  });

  onINP((metric) => {
    const formatted = formatMetric(metric);
    metrics.push(formatted);
    bufferMetric(formatted);
    console.log('INP:', formatted);
  });

  // Retry stored metrics on page load
  if (typeof window !== 'undefined') {
    retryStoredMetrics();
  }

  // Send remaining metrics on page unload
  if (typeof window !== 'undefined') {
    window.addEventListener('beforeunload', () => {
      if (metricsBuffer.length > 0) {
        // Use sendBeacon for reliability
        const endpoint = process.env.NEXT_PUBLIC_ANALYTICS_ENDPOINT || '/api/metrics';
        navigator.sendBeacon(endpoint, JSON.stringify({
          metrics: metricsBuffer,
          session: {
            id: getSessionId(),
            timestamp: Date.now()
          }
        }));
      }
    });
  }

  return metrics;
}

// Export singleton instance
export const customMetrics = new CustomMetrics();