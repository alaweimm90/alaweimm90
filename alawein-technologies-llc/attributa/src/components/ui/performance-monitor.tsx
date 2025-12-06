import React, { useEffect, useState } from 'react';
import { Badge } from '@/components/ui/badge';
import { cn } from '@/lib/utils';

interface PerformanceMetrics {
  lcp: number | null;
  fcp: number | null;
  cls: number | null;
  fid: number | null;
}

interface PerformanceMonitorProps {
  className?: string;
  showBadge?: boolean;
}

export function PerformanceMonitor({ className, showBadge = true }: PerformanceMonitorProps) {
  const [metrics, setMetrics] = useState<PerformanceMetrics>({
    lcp: null,
    fcp: null,
    cls: null,
    fid: null,
  });
  const [isVisible, setIsVisible] = useState(false);

  useEffect(() => {
    let observer: PerformanceObserver;

    if ('PerformanceObserver' in window) {
      observer = new PerformanceObserver((list) => {
        const entries = list.getEntries();
        
        entries.forEach((entry) => {
          switch (entry.entryType) {
            case 'largest-contentful-paint':
              setMetrics(prev => ({ ...prev, lcp: entry.startTime }));
              break;
            case 'first-contentful-paint':
              setMetrics(prev => ({ ...prev, fcp: entry.startTime }));
              break;
            case 'layout-shift':
              if (!('hadRecentInput' in entry && entry.hadRecentInput)) {
                setMetrics(prev => ({
                  ...prev,
                  cls: (prev.cls || 0) + ('value' in entry && typeof entry.value === 'number' ? entry.value : 0)
                }));
              }
              break;
            case 'first-input':
              setMetrics(prev => ({ ...prev, fid: ('processingStart' in entry && typeof entry.processingStart === 'number' ? entry.processingStart : entry.startTime) - entry.startTime }));
              break;
          }
        });
      });

      try {
        observer.observe({ entryTypes: ['largest-contentful-paint', 'first-contentful-paint', 'layout-shift', 'first-input'] });
      } catch (e) {
        console.warn('Performance monitoring not fully supported');
      }
    }

    // Show performance badge after page load
    const timer = setTimeout(() => setIsVisible(true), 2000);

    return () => {
      observer?.disconnect();
      clearTimeout(timer);
    };
  }, []);

  const getScoreColor = (metric: string, value: number | null) => {
    if (value === null) return 'text-muted-foreground';
    
    switch (metric) {
      case 'lcp':
        return value <= 2500 ? 'text-success' : value <= 4000 ? 'text-warning' : 'text-destructive';
      case 'fcp':
        return value <= 1800 ? 'text-success' : value <= 3000 ? 'text-warning' : 'text-destructive';
      case 'cls':
        return value <= 0.1 ? 'text-success' : value <= 0.25 ? 'text-warning' : 'text-destructive';
      case 'fid':
        return value <= 100 ? 'text-success' : value <= 300 ? 'text-warning' : 'text-destructive';
      default:
        return 'text-muted-foreground';
    }
  };

  const overallScore = React.useMemo(() => {
    const { lcp, fcp, cls, fid } = metrics;
    let score = 0;
    let count = 0;

    if (lcp !== null) {
      score += lcp <= 2500 ? 100 : lcp <= 4000 ? 75 : 50;
      count++;
    }
    if (fcp !== null) {
      score += fcp <= 1800 ? 100 : fcp <= 3000 ? 75 : 50;
      count++;
    }
    if (cls !== null) {
      score += cls <= 0.1 ? 100 : cls <= 0.25 ? 75 : 50;
      count++;
    }
    if (fid !== null) {
      score += fid <= 100 ? 100 : fid <= 300 ? 75 : 50;
      count++;
    }

    return count > 0 ? Math.round(score / count) : null;
  }, [metrics]);

  if (!showBadge || !isVisible || overallScore === null) {
    return null;
  }

  return (
    <div className={cn("fixed bottom-4 right-4 z-50", className)}>
      <Badge 
        variant={overallScore >= 90 ? "default" : overallScore >= 75 ? "secondary" : "destructive"}
        className="backdrop-blur-sm bg-background/90 border shadow-lg animate-fade-in"
      >
        <span className="text-xs font-mono">
          Perf: {overallScore}
        </span>
        {metrics.lcp && (
          <span className={cn("ml-2 text-xs", getScoreColor('lcp', metrics.lcp))}>
            LCP {Math.round(metrics.lcp)}ms
          </span>
        )}
      </Badge>
    </div>
  );
}

// Hook for components to track custom performance metrics
export function usePerformanceMetric(name: string, startTime?: number) {
  const [duration, setDuration] = useState<number | null>(null);

  const markStart = React.useCallback(() => {
    performance.mark(`${name}-start`);
  }, [name]);

  const markEnd = React.useCallback(() => {
    performance.mark(`${name}-end`);
    performance.measure(name, `${name}-start`, `${name}-end`);
    
    const measure = performance.getEntriesByName(name, 'measure')[0];
    if (measure) {
      setDuration(measure.duration);
    }
  }, [name]);

  React.useEffect(() => {
    if (startTime) {
      const currentTime = performance.now();
      setDuration(currentTime - startTime);
    }
  }, [startTime]);

  return { duration, markStart, markEnd };
}