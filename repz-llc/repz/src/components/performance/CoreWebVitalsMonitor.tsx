import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "@/ui/molecules/Card";
import { Badge } from "@/ui/atoms/Badge";
import { Progress } from "@/components/ui/progress";
import { Button } from "@/ui/atoms/Button";
import { 
  Activity, 
  Zap, 
  Clock, 
  TrendingUp,
  AlertTriangle,
  CheckCircle,
  RefreshCw
} from 'lucide-react';

interface CoreWebVitalsMetric {
  name: string;
  value: number;
  threshold: number;
  unit: string;
  status: 'good' | 'needs-improvement' | 'poor';
  description: string;
}

interface PerformanceEntry {
  name: string;
  entryType: string;
  startTime: number;
  duration: number;
}

export function CoreWebVitalsMonitor() {
  const [isMonitoring, setIsMonitoring] = useState(false);
  const [metrics, setMetrics] = useState<CoreWebVitalsMetric[]>([
    {
      name: 'Largest Contentful Paint (LCP)',
      value: 0,
      threshold: 2500,
      unit: 'ms',
      status: 'good',
      description: 'Time until the largest content element becomes visible'
    },
    {
      name: 'First Input Delay (FID)',
      value: 0,
      threshold: 100,
      unit: 'ms', 
      status: 'good',
      description: 'Time from first user interaction to browser response'
    },
    {
      name: 'Cumulative Layout Shift (CLS)',
      value: 0,
      threshold: 0.1,
      unit: '',
      status: 'good',
      description: 'Measure of visual stability during page load'
    },
    {
      name: 'First Contentful Paint (FCP)',
      value: 0,
      threshold: 1800,
      unit: 'ms',
      status: 'good',
      description: 'Time until first content appears on screen'
    },
    {
      name: 'Time to First Byte (TTFB)',
      value: 0,
      threshold: 600,
      unit: 'ms',
      status: 'good',
      description: 'Time from request to first byte of response'
    }
  ]);

  const [performanceScore, setPerformanceScore] = useState(0);

  useEffect(() => {
    if (isMonitoring) {
      collectCoreWebVitals();
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isMonitoring]);

  const collectCoreWebVitals = async () => {
    try {
      const updatedMetrics = [...metrics];

      // Collect performance entries
      const observer = new PerformanceObserver((list) => {
        for (const entry of list.getEntries()) {
          if (entry.entryType === 'largest-contentful-paint') {
            updateMetric(updatedMetrics, 'Largest Contentful Paint (LCP)', entry.startTime);
          }
          if (entry.entryType === 'first-input') {
            const fidEntry = entry as PerformanceEntry & { processingStart?: number };
            updateMetric(updatedMetrics, 'First Input Delay (FID)', (fidEntry.processingStart || entry.startTime) - entry.startTime);
          }
          if (entry.entryType === 'layout-shift') {
            const clsEntry = entry as PerformanceEntry & { value?: number; hadRecentInput?: boolean };
            if (!clsEntry.hadRecentInput) {
              updateMetric(updatedMetrics, 'Cumulative Layout Shift (CLS)', clsEntry.value || 0);
            }
          }
        }
      });

      // Observe different entry types
      try {
        observer.observe({ entryTypes: ['largest-contentful-paint', 'first-input', 'layout-shift'] });
      } catch (e) {
        console.warn('Some performance observers not supported');
      }

      // Collect paint timing
      const paintEntries = performance.getEntriesByType('paint');
      const fcp = paintEntries.find(entry => entry.name === 'first-contentful-paint');
      if (fcp) {
        updateMetric(updatedMetrics, 'First Contentful Paint (FCP)', fcp.startTime);
      }

      // Collect navigation timing for TTFB
      const navigation = performance.getEntriesByType('navigation')[0] as PerformanceNavigationTiming;
      if (navigation) {
        const ttfb = navigation.responseStart - navigation.requestStart;
        updateMetric(updatedMetrics, 'Time to First Byte (TTFB)', ttfb);
      }

      setMetrics(updatedMetrics);
      calculatePerformanceScore(updatedMetrics);

      // Clean up observer after collection
      setTimeout(() => {
        observer.disconnect();
      }, 5000);

    } catch (error) {
      console.error('Error collecting Core Web Vitals:', error);
    }
  };

  const updateMetric = (metricsArray: CoreWebVitalsMetric[], name: string, value: number) => {
    const metricIndex = metricsArray.findIndex(m => m.name === name);
    if (metricIndex !== -1) {
      const metric = metricsArray[metricIndex];
      metric.value = Math.round(value * 100) / 100;
      
      // Determine status based on thresholds
      if (name.includes('CLS')) {
        metric.status = value <= 0.1 ? 'good' : value <= 0.25 ? 'needs-improvement' : 'poor';
      } else {
        const ratio = value / metric.threshold;
        metric.status = ratio <= 1 ? 'good' : ratio <= 1.5 ? 'needs-improvement' : 'poor';
      }
    }
  };

  const calculatePerformanceScore = (metricsArray: CoreWebVitalsMetric[]) => {
    const goodMetrics = metricsArray.filter(m => m.status === 'good').length;
    const totalMetrics = metricsArray.length;
    const score = Math.round((goodMetrics / totalMetrics) * 100);
    setPerformanceScore(score);
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'good': return <CheckCircle className="h-4 w-4 text-success" />;
      case 'needs-improvement': return <AlertTriangle className="h-4 w-4 text-warning" />;
      case 'poor': return <AlertTriangle className="h-4 w-4 text-destructive" />;
      default: return <Activity className="h-4 w-4" />;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'good': return 'text-success';
      case 'needs-improvement': return 'text-warning';
      case 'poor': return 'text-destructive';
      default: return 'text-muted-foreground';
    }
  };

  const startMonitoring = () => {
    setIsMonitoring(true);
    collectCoreWebVitals();
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold">Core Web Vitals Monitor</h2>
          <p className="text-muted-foreground">Google PageSpeed Insights Metrics</p>
        </div>
        <Button 
          onClick={startMonitoring}
          disabled={isMonitoring}
          className="flex items-center gap-2"
        >
          <RefreshCw className={`h-4 w-4 ${isMonitoring ? 'animate-spin' : ''}`} />
          {isMonitoring ? 'Monitoring...' : 'Start Monitoring'}
        </Button>
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <TrendingUp className="h-5 w-5" />
            Performance Score
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div>
              <div className="flex items-center justify-between mb-2">
                <span className="text-3xl font-bold">{performanceScore}%</span>
                <Badge variant={performanceScore >= 90 ? "default" : performanceScore >= 70 ? "secondary" : "destructive"}>
                  {performanceScore >= 90 ? "Excellent" : performanceScore >= 70 ? "Good" : "Needs Work"}
                </Badge>
              </div>
              <Progress value={performanceScore} className="h-3" />
            </div>
            <p className="text-sm text-muted-foreground">
              Based on Core Web Vitals performance thresholds
            </p>
          </div>
        </CardContent>
      </Card>

      <div className="grid gap-4">
        {metrics.map((metric, index) => (
          <Card key={index}>
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div className="space-y-2">
                  <div className="flex items-center gap-3">
                    {metric.name.includes('LCP') && <Clock className="h-4 w-4" />}
                    {metric.name.includes('FID') && <Zap className="h-4 w-4" />}
                    {metric.name.includes('CLS') && <Activity className="h-4 w-4" />}
                    {metric.name.includes('FCP') && <Clock className="h-4 w-4" />}
                    {metric.name.includes('TTFB') && <Activity className="h-4 w-4" />}
                    <h3 className="font-semibold">{metric.name}</h3>
                    {getStatusIcon(metric.status)}
                  </div>
                  <p className="text-sm text-muted-foreground">
                    {metric.description}
                  </p>
                </div>
                <div className="text-right">
                  <div className={`text-2xl font-bold ${getStatusColor(metric.status)}`}>
                    {metric.value === 0 ? '-' : `${metric.value}${metric.unit}`}
                  </div>
                  <div className="text-sm text-muted-foreground">
                    Target: {metric.threshold}{metric.unit}
                  </div>
                </div>
              </div>
              {metric.value > 0 && (
                <div className="mt-4">
                  <Progress 
                    value={Math.min((metric.threshold / Math.max(metric.value, metric.threshold)) * 100, 100)} 
                    className="h-2"
                  />
                </div>
              )}
            </CardContent>
          </Card>
        ))}
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Performance Recommendations</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            <div className="flex items-start gap-3">
              <CheckCircle className="h-4 w-4 text-success mt-0.5" />
              <div>
                <h4 className="font-medium">Image Optimization</h4>
                <p className="text-sm text-muted-foreground">Use WebP format and lazy loading for images</p>
              </div>
            </div>
            <div className="flex items-start gap-3">
              <CheckCircle className="h-4 w-4 text-success mt-0.5" />
              <div>
                <h4 className="font-medium">Code Splitting</h4>
                <p className="text-sm text-muted-foreground">Implement route-based and component-based code splitting</p>
              </div>
            </div>
            <div className="flex items-start gap-3">
              <CheckCircle className="h-4 w-4 text-success mt-0.5" />
              <div>
                <h4 className="font-medium">Critical CSS</h4>
                <p className="text-sm text-muted-foreground">Inline critical CSS and defer non-critical styles</p>
              </div>
            </div>
            <div className="flex items-start gap-3">
              <AlertTriangle className="h-4 w-4 text-warning mt-0.5" />
              <div>
                <h4 className="font-medium">Layout Stability</h4>
                <p className="text-sm text-muted-foreground">Reserve space for images and ads to prevent layout shifts</p>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}