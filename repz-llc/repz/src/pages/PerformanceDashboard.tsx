import React, { useState, useEffect } from 'react';
import { Card } from '@/ui/molecules/Card';

interface PerformanceMetrics {
  overallScore: number;
  grade: string;
  webVitals: {
    lcp: number;
    fid: number;
    cls: number;
    fcp: number;
    ttfb: number;
  };
  buildPerformance: {
    buildTime: number;
    bundleSize: number;
    chunkCount: number;
  };
  alerts: Array<{
    level: string;
    message: string;
    timestamp: string;
  }>;
}

export const PerformanceDashboard = () => {
  const [metrics, setMetrics] = useState<PerformanceMetrics | null>(null);
  const [loading, setLoading] = useState(true);
  
  useEffect(() => {
    fetchPerformanceMetrics();
    
    // Refresh every 30 seconds
    const interval = setInterval(fetchPerformanceMetrics, 30000);
    return () => clearInterval(interval);
  }, []);
  
  const fetchPerformanceMetrics = async () => {
    try {
      const response = await fetch('/api/performance-metrics');
      const data = await response.json();
      setMetrics(data);
    } catch (error) {
      console.error('Failed to fetch performance metrics:', error);
    } finally {
      setLoading(false);
    }
  };
  
  if (loading) {
    return <div className="p-8">Loading performance dashboard...</div>;
  }
  
  if (!metrics) {
    return <div className="p-8">Failed to load performance metrics</div>;
  }
  
  const getScoreColor = (score: number) => {
    if (score >= 90) return 'text-green-600';
    if (score >= 80) return 'text-yellow-600';
    return 'text-red-600';
  };
  
  const getVitalsStatus = (metric: string, value: number) => {
    const thresholds: Record<string, { good: number; poor: number }> = {
      lcp: { good: 2500, poor: 4000 },
      fid: { good: 100, poor: 300 },
      cls: { good: 0.1, poor: 0.25 },
      fcp: { good: 1800, poor: 3000 },
      ttfb: { good: 800, poor: 1800 }
    };
    
    const threshold = thresholds[metric];
    if (!threshold) return 'unknown';
    
    if (value <= threshold.good) return 'good';
    if (value <= threshold.poor) return 'needs-improvement';
    return 'poor';
  };
  
  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'good': return 'Good';
      case 'needs-improvement': return 'Needs Improvement';
      case 'poor': return 'Poor';
      default: return 'Unknown';
    }
  };
  
  return (
    <div className="p-6 max-w-7xl mx-auto">
      <h1 className="text-3xl font-bold mb-8">Performance Dashboard</h1>
      
      {/* Overall Score */}
      <Card className="mb-8">
        <div className="p-6">
          <h2 className="text-xl font-semibold mb-4">Overall Performance</h2>
          <div className="flex items-center gap-4">
            <div className={`text-6xl font-bold ${getScoreColor(metrics.overallScore)}`}>
              {metrics.overallScore}
            </div>
            <div>
              <div className="text-2xl font-semibold">Grade: {metrics.grade}</div>
              <div className="text-gray-600">Out of 100</div>
            </div>
          </div>
        </div>
      </Card>
      
      {/* Web Vitals */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-8">
        <Card>
          <div className="p-4">
            <h3 className="font-semibold mb-2">Largest Contentful Paint</h3>
            <div className="flex items-center gap-2">
              {getStatusIcon(getVitalsStatus('lcp', metrics.webVitals.lcp))}
              <span className="text-2xl font-bold">{metrics.webVitals.lcp}ms</span>
            </div>
          </div>
        </Card>
        
        <Card>
          <div className="p-4">
            <h3 className="font-semibold mb-2">First Input Delay</h3>
            <div className="flex items-center gap-2">
              {getStatusIcon(getVitalsStatus('fid', metrics.webVitals.fid))}
              <span className="text-2xl font-bold">{metrics.webVitals.fid}ms</span>
            </div>
          </div>
        </Card>
        
        <Card>
          <div className="p-4">
            <h3 className="font-semibold mb-2">Cumulative Layout Shift</h3>
            <div className="flex items-center gap-2">
              {getStatusIcon(getVitalsStatus('cls', metrics.webVitals.cls))}
              <span className="text-2xl font-bold">{metrics.webVitals.cls}</span>
            </div>
          </div>
        </Card>
      </div>
      
      {/* Build Performance */}
      <Card className="mb-8">
        <div className="p-6">
          <h2 className="text-xl font-semibold mb-4">Build Performance</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div>
              <div className="text-sm text-gray-600">Build Time</div>
              <div className="text-2xl font-bold">{metrics.buildPerformance.buildTime}s</div>
            </div>
            <div>
              <div className="text-sm text-gray-600">Bundle Size</div>
              <div className="text-2xl font-bold">{metrics.buildPerformance.bundleSize}MB</div>
            </div>
            <div>
              <div className="text-sm text-gray-600">Chunks</div>
              <div className="text-2xl font-bold">{metrics.buildPerformance.chunkCount}</div>
            </div>
          </div>
        </div>
      </Card>
      
      {/* Recent Alerts */}
      {metrics.alerts.length > 0 && (
        <Card>
          <div className="p-6">
            <h2 className="text-xl font-semibold mb-4">Recent Alerts</h2>
            <div className="space-y-2">
              {metrics.alerts.map((alert, index) => (
                <div key={index} className={`p-3 rounded ${
                  alert.level === 'critical' ? 'bg-red-100 text-red-800' :
                  alert.level === 'warning' ? 'bg-yellow-100 text-yellow-800' :
                  'bg-blue-100 text-blue-800'
                }`}>
                  <div className="font-medium">{alert.message}</div>
                  <div className="text-sm opacity-75">{alert.timestamp}</div>
                </div>
              ))}
            </div>
          </div>
        </Card>
      )}
    </div>
  );
};

export default PerformanceDashboard;
