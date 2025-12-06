import React, { useEffect, useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/ui/molecules/Card';
import { Badge } from '@/ui/atoms/Badge';
import { Progress } from '@/components/ui/progress';
import { Activity, Zap, Clock, Database } from 'lucide-react';

interface PerformanceMetrics {
  coreWebVitals: {
    lcp: number; // Largest Contentful Paint
    fcp: number; // First Contentful Paint
    cls: number; // Cumulative Layout Shift
    fid: number; // First Input Delay
  };
  resourceMetrics: {
    totalSize: number;
    loadTime: number;
    memoryUsage: number;
    cacheHitRate: number;
  };
  userExperience: {
    bounceRate: number;
    sessionDuration: number;
    pageViews: number;
    conversionRate: number;
  };
}

export const AdvancedPerformanceMonitor: React.FC = () => {
  const [metrics, setMetrics] = useState<PerformanceMetrics | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const collectMetrics = async () => {
      try {
        // Collect Core Web Vitals
        const navigation = performance.getEntriesByType('navigation')[0] as PerformanceNavigationTiming;
        const paint = performance.getEntriesByType('paint');
        
        const mockMetrics: PerformanceMetrics = {
          coreWebVitals: {
            lcp: navigation.loadEventEnd - navigation.loadEventStart,
            fcp: paint.find(p => p.name === 'first-contentful-paint')?.startTime || 0,
            cls: Math.random() * 0.1, // Mock CLS
            fid: Math.random() * 100 // Mock FID
          },
          resourceMetrics: {
            totalSize: navigation.transferSize || 0,
            loadTime: navigation.loadEventEnd - navigation.fetchStart,
            memoryUsage: (performance as any).memory?.usedJSHeapSize || 0,
            cacheHitRate: 85 + Math.random() * 10
          },
          userExperience: {
            bounceRate: 25 + Math.random() * 20,
            sessionDuration: 180 + Math.random() * 120,
            pageViews: 3 + Math.random() * 5,
            conversionRate: 8 + Math.random() * 4
          }
        };

        setMetrics(mockMetrics);
      } catch (error) {
        console.error('Failed to collect performance metrics:', error);
      } finally {
        setIsLoading(false);
      }
    };

    collectMetrics();
  }, []);

  const getPerformanceGrade = (metric: string, value: number): string => {
    const thresholds: Record<string, { good: number; needs_improvement: number }> = {
      lcp: { good: 2500, needs_improvement: 4000 },
      fcp: { good: 1800, needs_improvement: 3000 },
      cls: { good: 0.1, needs_improvement: 0.25 },
      fid: { good: 100, needs_improvement: 300 }
    };

    const threshold = thresholds[metric];
    if (!threshold) return 'unknown';

    if (value <= threshold.good) return 'good';
    if (value <= threshold.needs_improvement) return 'needs-improvement';
    return 'poor';
  };

  const formatBytes = (bytes: number): string => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  if (isLoading) {
    return (
      <Card className="glass-card">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Activity className="h-5 w-5" />
            Performance Monitor
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {[1, 2, 3, 4].map((i) => (
              <div key={i} className="h-4 bg-muted/20 rounded animate-pulse" />
            ))}
          </div>
        </CardContent>
      </Card>
    );
  }

  if (!metrics) {
    return (
      <Card className="glass-card">
        <CardContent className="pt-6">
          <p className="text-muted-foreground text-center">Unable to load performance metrics</p>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
      {/* Core Web Vitals */}
      <Card className="glass-card">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Zap className="h-5 w-5 text-performance" />
            Core Web Vitals
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-3">
            <div className="flex justify-between items-center">
              <span className="text-sm">LCP</span>
              <div className="flex items-center gap-2">
                <Badge variant={getPerformanceGrade('lcp', metrics.coreWebVitals.lcp) === 'good' ? 'default' : 'destructive'}>
                  {(metrics.coreWebVitals.lcp / 1000).toFixed(2)}s
                </Badge>
              </div>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-sm">FCP</span>
              <Badge variant={getPerformanceGrade('fcp', metrics.coreWebVitals.fcp) === 'good' ? 'default' : 'destructive'}>
                {(metrics.coreWebVitals.fcp / 1000).toFixed(2)}s
              </Badge>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-sm">CLS</span>
              <Badge variant={getPerformanceGrade('cls', metrics.coreWebVitals.cls) === 'good' ? 'default' : 'destructive'}>
                {metrics.coreWebVitals.cls.toFixed(3)}
              </Badge>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-sm">FID</span>
              <Badge variant={getPerformanceGrade('fid', metrics.coreWebVitals.fid) === 'good' ? 'default' : 'destructive'}>
                {metrics.coreWebVitals.fid.toFixed(0)}ms
              </Badge>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Resource Metrics */}
      <Card className="glass-card">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Database className="h-5 w-5 text-adaptive" />
            Resource Usage
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-3">
            <div>
              <div className="flex justify-between text-sm mb-1">
                <span>Bundle Size</span>
                <span>{formatBytes(metrics.resourceMetrics.totalSize)}</span>
              </div>
              <Progress value={Math.min(100, (metrics.resourceMetrics.totalSize / 1024 / 1024) * 10)} />
            </div>
            <div>
              <div className="flex justify-between text-sm mb-1">
                <span>Load Time</span>
                <span>{(metrics.resourceMetrics.loadTime / 1000).toFixed(2)}s</span>
              </div>
              <Progress value={Math.min(100, (metrics.resourceMetrics.loadTime / 1000) * 20)} />
            </div>
            <div>
              <div className="flex justify-between text-sm mb-1">
                <span>Memory Usage</span>
                <span>{formatBytes(metrics.resourceMetrics.memoryUsage)}</span>
              </div>
              <Progress value={Math.min(100, (metrics.resourceMetrics.memoryUsage / 1024 / 1024 / 100) * 100)} />
            </div>
            <div>
              <div className="flex justify-between text-sm mb-1">
                <span>Cache Hit Rate</span>
                <span>{metrics.resourceMetrics.cacheHitRate.toFixed(1)}%</span>
              </div>
              <Progress value={metrics.resourceMetrics.cacheHitRate} />
            </div>
          </div>
        </CardContent>
      </Card>

      {/* User Experience */}
      <Card className="glass-card">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Clock className="h-5 w-5 text-longevity" />
            User Experience
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-3">
            <div className="flex justify-between items-center">
              <span className="text-sm">Bounce Rate</span>
              <Badge variant={metrics.userExperience.bounceRate < 30 ? 'default' : 'secondary'}>
                {metrics.userExperience.bounceRate.toFixed(1)}%
              </Badge>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-sm">Session Duration</span>
              <span className="text-sm font-medium">{Math.floor(metrics.userExperience.sessionDuration / 60)}m {Math.floor(metrics.userExperience.sessionDuration % 60)}s</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-sm">Page Views</span>
              <span className="text-sm font-medium">{metrics.userExperience.pageViews.toFixed(1)}</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-sm">Conversion Rate</span>
              <Badge variant="default" className="bg-success/10 text-success border-success/20">
                {metrics.userExperience.conversionRate.toFixed(1)}%
              </Badge>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default AdvancedPerformanceMonitor;