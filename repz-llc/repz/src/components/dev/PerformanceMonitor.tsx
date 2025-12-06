import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/ui/molecules/Card';
import { Badge } from '@/ui/atoms/Badge';
import { Button } from '@/ui/atoms/Button';
import { Alert, AlertDescription } from '@/ui/molecules/Alert';
import { Activity, Zap, Clock, Database, Gauge, RefreshCw } from 'lucide-react';

interface PerformanceMetrics {
  // Core Web Vitals
  lcp: number | null; // Largest Contentful Paint
  fid: number | null; // First Input Delay
  cls: number | null; // Cumulative Layout Shift
  fcp: number | null; // First Contentful Paint
  ttfb: number | null; // Time to First Byte
  
  // Custom metrics
  memoryUsage: number | null;
  loadTime: number | null;
  renderCount: number;
  bundleSize: number | null;
  
  // Real-time metrics
  fps: number | null;
  timestamp: number;
}

const PerformanceMonitor: React.FC = () => {
  const [metrics, setMetrics] = useState<PerformanceMetrics>({
    lcp: null,
    fid: null,
    cls: null,
    fcp: null,
    ttfb: null,
    memoryUsage: null,
    loadTime: null,
    renderCount: 0,
    bundleSize: null,
    fps: null,
    timestamp: Date.now()
  });

  const [isVisible, setIsVisible] = useState(process.env.NODE_ENV === 'development');
  const [autoRefresh, setAutoRefresh] = useState(true);

  // Collect performance metrics
  const collectMetrics = () => {
    const newMetrics: PerformanceMetrics = {
      ...metrics,
      timestamp: Date.now(),
      renderCount: metrics.renderCount + 1
    };

    // Memory usage (Chrome only)
    if ('memory' in performance) {
      const memory = (performance as { memory?: { usedJSHeapSize: number } }).memory;
      if (memory) {
        newMetrics.memoryUsage = Math.round(memory.usedJSHeapSize / 1048576 * 100) / 100; // MB
      }
    }

    // Navigation timing
    const navEntries = performance.getEntriesByType('navigation') as PerformanceNavigationTiming[];
    if (navEntries.length > 0) {
      const nav = navEntries[0];
      newMetrics.loadTime = Math.round(nav.loadEventEnd - nav.startTime);
      newMetrics.ttfb = Math.round(nav.responseStart - nav.startTime);
    }

    // Paint timing
    const paintEntries = performance.getEntriesByType('paint');
    paintEntries.forEach((entry) => {
      if (entry.name === 'first-contentful-paint') {
        newMetrics.fcp = Math.round(entry.startTime);
      }
    });

    // LCP
    if ('PerformanceObserver' in window) {
      try {
        const lcpObserver = new PerformanceObserver((list) => {
          const entries = list.getEntries();
          const lastEntry = entries[entries.length - 1] as PerformanceEntry & { startTime: number };
          if (lastEntry) {
            setMetrics(prev => ({
              ...prev,
              lcp: Math.round(lastEntry.startTime)
            }));
          }
        });
        lcpObserver.observe({ type: 'largest-contentful-paint', buffered: true });
      } catch (error) {
        console.warn('LCP observer failed:', error);
      }
    }

    // Estimate bundle size from script tags
    const scripts = document.querySelectorAll('script[src]');
    let estimatedSize = 0;
    scripts.forEach(script => {
      const src = script.getAttribute('src');
      if (src && src.includes('assets/')) {
        // This is a rough estimation - in production you'd have actual size data
        estimatedSize += 200; // KB estimate per chunk
      }
    });
    newMetrics.bundleSize = estimatedSize;

    setMetrics(newMetrics);
  };

  // FPS monitoring
  useEffect(() => {
    let frameCount = 0;
    let lastTime = performance.now();
    let animationId: number;

    const measureFPS = () => {
      frameCount++;
      const currentTime = performance.now();
      
      if (currentTime >= lastTime + 1000) {
        const fps = Math.round((frameCount * 1000) / (currentTime - lastTime));
        setMetrics(prev => ({ ...prev, fps }));
        frameCount = 0;
        lastTime = currentTime;
      }
      
      animationId = requestAnimationFrame(measureFPS);
    };

    if (isVisible) {
      measureFPS();
    }

    return () => {
      if (animationId) {
        cancelAnimationFrame(animationId);
      }
    };
  }, [isVisible]);

  // Auto refresh metrics
  useEffect(() => {
    if (!autoRefresh || !isVisible) return;

    const interval = setInterval(collectMetrics, 2000);
    return () => clearInterval(interval);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [autoRefresh, isVisible]);

  // Initial metrics collection
  useEffect(() => {
    if (isVisible) {
      collectMetrics();
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isVisible]);

  const getMetricStatus = (value: number | null, thresholds: { good: number; poor: number }): 'good' | 'needs-improvement' | 'poor' => {
    if (value === null) return 'good';
    if (value <= thresholds.good) return 'good';
    if (value <= thresholds.poor) return 'needs-improvement';
    return 'poor';
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'good': return 'bg-green-100 text-green-800';
      case 'needs-improvement': return 'bg-yellow-100 text-yellow-800';
      case 'poor': return 'bg-red-100 text-red-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  if (!isVisible) {
    return (
      <Button
        variant="outline"
        size="sm"
        onClick={() => setIsVisible(true)}
        className="fixed bottom-4 right-4 z-50"
      >
        <Activity className="h-4 w-4 mr-2" />
        Show Performance
      </Button>
    );
  }

  return (
    <div className="fixed bottom-4 right-4 z-50 w-96 max-h-96 overflow-y-auto">
      <Card className="shadow-lg border-2">
        <CardHeader className="pb-2">
          <div className="flex items-center justify-between">
            <CardTitle className="text-sm flex items-center gap-2">
              <Gauge className="h-4 w-4" />
              Performance Monitor
            </CardTitle>
            <div className="flex gap-1">
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setAutoRefresh(!autoRefresh)}
              >
                <RefreshCw className={`h-3 w-3 ${autoRefresh ? 'animate-spin' : ''}`} />
              </Button>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setIsVisible(false)}
              >
                ×
              </Button>
            </div>
          </div>
        </CardHeader>
        
        <CardContent className="space-y-4 text-xs">
          {/* Core Web Vitals */}
          <div>
            <h4 className="font-semibold mb-2 flex items-center gap-1">
              <Zap className="h-3 w-3" />
              Core Web Vitals
            </h4>
            <div className="grid grid-cols-2 gap-2">
              <div className="flex justify-between items-center">
                <span>LCP:</span>
                <Badge className={getStatusColor(getMetricStatus(metrics.lcp, { good: 2500, poor: 4000 }))}>
                  {metrics.lcp ? `${metrics.lcp}ms` : 'N/A'}
                </Badge>
              </div>
              <div className="flex justify-between items-center">
                <span>FCP:</span>
                <Badge className={getStatusColor(getMetricStatus(metrics.fcp, { good: 1800, poor: 3000 }))}>
                  {metrics.fcp ? `${metrics.fcp}ms` : 'N/A'}
                </Badge>
              </div>
              <div className="flex justify-between items-center">
                <span>TTFB:</span>
                <Badge className={getStatusColor(getMetricStatus(metrics.ttfb, { good: 800, poor: 1800 }))}>
                  {metrics.ttfb ? `${metrics.ttfb}ms` : 'N/A'}
                </Badge>
              </div>
              <div className="flex justify-between items-center">
                <span>CLS:</span>
                <Badge className={getStatusColor(getMetricStatus(metrics.cls, { good: 0.1, poor: 0.25 }))}>
                  {metrics.cls ? metrics.cls.toFixed(3) : 'N/A'}
                </Badge>
              </div>
            </div>
          </div>

          {/* Runtime Metrics */}
          <div>
            <h4 className="font-semibold mb-2 flex items-center gap-1">
              <Clock className="h-3 w-3" />
              Runtime Metrics
            </h4>
            <div className="space-y-1">
              <div className="flex justify-between">
                <span>FPS:</span>
                <Badge className={metrics.fps && metrics.fps < 30 ? 'bg-red-100 text-red-800' : 'bg-green-100 text-green-800'}>
                  {metrics.fps ? `${metrics.fps}` : 'N/A'}
                </Badge>
              </div>
              <div className="flex justify-between">
                <span>Memory:</span>
                <Badge className={metrics.memoryUsage && metrics.memoryUsage > 50 ? 'bg-yellow-100 text-yellow-800' : 'bg-green-100 text-green-800'}>
                  {metrics.memoryUsage ? `${metrics.memoryUsage}MB` : 'N/A'}
                </Badge>
              </div>
              <div className="flex justify-between">
                <span>Renders:</span>
                <Badge variant="outline">{metrics.renderCount}</Badge>
              </div>
            </div>
          </div>

          {/* Bundle Info */}
          <div>
            <h4 className="font-semibold mb-2 flex items-center gap-1">
              <Database className="h-3 w-3" />
              Bundle Info
            </h4>
            <div className="space-y-1">
              <div className="flex justify-between">
                <span>Est. Size:</span>
                <Badge variant="outline">
                  {metrics.bundleSize ? `~${metrics.bundleSize}KB` : 'N/A'}
                </Badge>
              </div>
              <div className="flex justify-between">
                <span>Load Time:</span>
                <Badge className={metrics.loadTime && metrics.loadTime > 3000 ? 'bg-red-100 text-red-800' : 'bg-green-100 text-green-800'}>
                  {metrics.loadTime ? `${metrics.loadTime}ms` : 'N/A'}
                </Badge>
              </div>
            </div>
          </div>

          {/* Performance Tips */}
          {(metrics.memoryUsage && metrics.memoryUsage > 50) && (
            <Alert>
              <AlertDescription className="text-xs">
                High memory usage detected. Consider optimizing component re-renders.
              </AlertDescription>
            </Alert>
          )}

          {(metrics.fps && metrics.fps < 30) && (
            <Alert>
              <AlertDescription className="text-xs">
                Low FPS detected. Check for expensive operations in render cycle.
              </AlertDescription>
            </Alert>
          )}
        </CardContent>
      </Card>
    </div>
  );
};

export default PerformanceMonitor;