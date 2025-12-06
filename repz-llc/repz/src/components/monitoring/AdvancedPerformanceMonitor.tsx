import React, { useState, useEffect } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/ui/molecules/Card';
import { Badge } from '@/ui/atoms/Badge';
import { Button } from '@/ui/atoms/Button';
import { Progress } from '@/components/ui/progress';
import { 
  Activity, 
  Clock, 
  MemoryStick, 
  Zap, 
  AlertTriangle,
  CheckCircle,
  TrendingUp,
  TrendingDown
} from 'lucide-react';
import { useMemoryMonitor } from '@/hooks/useMemoryMonitor';
import { usePerformanceMonitor } from '@/hooks/usePerformanceMonitor';

interface PerformanceMetric {
  name: string;
  value: number;
  unit: string;
  status: 'good' | 'warning' | 'critical';
  threshold: number;
  icon: React.ComponentType<{ className?: string }>;
}

export const AdvancedPerformanceMonitor = () => {
  const [metrics, setMetrics] = useState<PerformanceMetric[]>([]);
  const [isMonitoring, setIsMonitoring] = useState(false);
  const [memoryHistory, setMemoryHistory] = useState<number[]>([]);

  const { trackInteraction } = usePerformanceMonitor('AdvancedPerformanceMonitor');
  
  const { getCurrentStats, forceGarbageCollection, isSupported } = useMemoryMonitor({
    threshold: 75,
    onThresholdExceeded: (stats) => {
      console.warn('Memory threshold exceeded:', stats);
    },
    onMemoryLeak: (stats) => {
      console.error('Memory leak detected:', stats);
    }
  });

  const collectMetrics = () => {
    trackInteraction('collect_metrics', () => {
      const memoryStats = getCurrentStats();
      const navigationTiming = performance.getEntriesByType('navigation')[0] as PerformanceNavigationTiming;
      
      const newMetrics: PerformanceMetric[] = [
        {
          name: 'Page Load Time',
          value: navigationTiming ? navigationTiming.loadEventEnd - navigationTiming.fetchStart : 0,
          unit: 'ms',
          status: navigationTiming && (navigationTiming.loadEventEnd - navigationTiming.fetchStart) < 2000 ? 'good' : 'warning',
          threshold: 2000,
          icon: Clock
        },
        {
          name: 'DOM Interactive',
          value: navigationTiming ? navigationTiming.domInteractive - navigationTiming.fetchStart : 0,
          unit: 'ms',
          status: navigationTiming && (navigationTiming.domInteractive - navigationTiming.fetchStart) < 1500 ? 'good' : 'warning',
          threshold: 1500,
          icon: Activity
        },
        {
          name: 'First Contentful Paint',
          value: (() => {
            const fcpEntry = performance.getEntriesByName('first-contentful-paint')[0];
            return fcpEntry ? fcpEntry.startTime : 0;
          })(),
          unit: 'ms',
          status: (() => {
            const fcpEntry = performance.getEntriesByName('first-contentful-paint')[0];
            return fcpEntry && fcpEntry.startTime < 1800 ? 'good' : 'warning';
          })(),
          threshold: 1800,
          icon: Zap
        }
      ];

      if (memoryStats) {
        newMetrics.push({
          name: 'Memory Usage',
          value: memoryStats.usagePercentage,
          unit: '%',
          status: memoryStats.usagePercentage < 60 ? 'good' : memoryStats.usagePercentage < 80 ? 'warning' : 'critical',
          threshold: 80,
          icon: MemoryStick
        });

        // Update memory history for trend analysis
        setMemoryHistory(prev => [...prev.slice(-9), memoryStats.usedMB]);
      }

      setMetrics(newMetrics);
    });
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'good': return 'text-green-600';
      case 'warning': return 'text-yellow-600';
      case 'critical': return 'text-red-600';
      default: return 'text-gray-600';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'good': return CheckCircle;
      case 'warning': return AlertTriangle;
      case 'critical': return AlertTriangle;
      default: return Activity;
    }
  };

  const getMemoryTrend = () => {
    if (memoryHistory.length < 2) return null;
    const recent = memoryHistory.slice(-3);
    const isIncreasing = recent.every((val, i) => i === 0 || val > recent[i - 1]);
    const isDecreasing = recent.every((val, i) => i === 0 || val < recent[i - 1]);
    
    if (isIncreasing) return { trend: 'up', icon: TrendingUp, color: 'text-red-500' };
    if (isDecreasing) return { trend: 'down', icon: TrendingDown, color: 'text-green-500' };
    return null;
  };

  useEffect(() => {
    collectMetrics();
    
    if (isMonitoring) {
      const interval = setInterval(collectMetrics, 5000);
      return () => clearInterval(interval);
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isMonitoring]);

  const memoryTrend = getMemoryTrend();

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <h2 className="text-2xl font-bold">Advanced Performance Monitor</h2>
          {isSupported && (
            <Badge variant="outline" className="text-xs">
              Memory API Supported
            </Badge>
          )}
        </div>
        <div className="flex gap-2">
          <Button 
            onClick={() => setIsMonitoring(!isMonitoring)}
            variant={isMonitoring ? "destructive" : "default"}
          >
            {isMonitoring ? 'Stop' : 'Start'} Monitoring
          </Button>
          <Button onClick={collectMetrics} variant="outline">
            <Activity className="mr-2 h-4 w-4" />
            Collect Metrics
          </Button>
          {isSupported && (
            <Button onClick={forceGarbageCollection} variant="outline">
              🗑️ Force GC
            </Button>
          )}
        </div>
      </div>

      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        {metrics.map((metric) => {
          const StatusIcon = getStatusIcon(metric.status);
          const MetricIcon = metric.icon;
          
          return (
            <Card key={metric.name}>
              <CardHeader className="pb-2">
                <CardTitle className="flex items-center gap-2 text-sm">
                  <MetricIcon className="h-4 w-4" />
                  {metric.name}
                  <StatusIcon className={`h-4 w-4 ${getStatusColor(metric.status)}`} />
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">
                  {metric.value.toFixed(0)}{metric.unit}
                </div>
                <Progress 
                  value={metric.name === 'Memory Usage' ? metric.value : (metric.value / metric.threshold) * 100} 
                  className="mt-2"
                />
                <div className="text-xs text-muted-foreground mt-1">
                  Threshold: {metric.threshold}{metric.unit}
                </div>
              </CardContent>
            </Card>
          );
        })}
      </div>

      {memoryHistory.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <MemoryStick className="h-5 w-5" />
              Memory Usage Trend
              {memoryTrend && (
                <div className="flex items-center gap-1">
                  <memoryTrend.icon className={`h-4 w-4 ${memoryTrend.color}`} />
                  <span className={`text-xs ${memoryTrend.color}`}>
                    {memoryTrend.trend === 'up' ? 'Increasing' : 'Decreasing'}
                  </span>
                </div>
              )}
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex items-end gap-1 h-16">
              {memoryHistory.map((usage, index) => (
                <div
                  key={index}
                  className="bg-primary/20 min-w-[8px] rounded-t"
                  style={{ 
                    height: `${(usage / Math.max(...memoryHistory)) * 100}%` 
                  }}
                  title={`${usage}MB`}
                />
              ))}
            </div>
            <div className="text-xs text-muted-foreground mt-2">
              Current: {memoryHistory[memoryHistory.length - 1]}MB | 
              Peak: {Math.max(...memoryHistory)}MB
            </div>
          </CardContent>
        </Card>
      )}

      <Card>
        <CardHeader>
          <CardTitle>Performance Recommendations</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="text-sm">
            <h4 className="font-medium mb-2">🚀 Optimization Tips:</h4>
            <ul className="space-y-1 text-muted-foreground">
              <li>• Monitor memory usage trends for potential leaks</li>
              <li>• Keep page load times under 2 seconds</li>
              <li>• Use React.memo() for expensive components</li>
              <li>• Implement virtual scrolling for large lists</li>
              <li>• Lazy load non-critical resources</li>
            </ul>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};