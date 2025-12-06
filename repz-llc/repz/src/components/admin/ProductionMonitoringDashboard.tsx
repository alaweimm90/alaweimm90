import React, { useState, useEffect, useMemo } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/ui/molecules/Card';
import { Badge } from '@/ui/atoms/Badge';
import { Button } from '@/ui/atoms/Button';
import { Alert, AlertDescription } from '@/ui/molecules/Alert';
import { 
  Activity, 
  Zap, 
  Clock, 
  Database, 
  Users, 
  AlertTriangle,
  CheckCircle,
  XCircle,
  TrendingUp,
  TrendingDown,
  Server,
  Globe,
  Smartphone,
  Monitor,
  Gauge,
  BarChart3,
  PieChart,
  LineChart,
  ChevronUp,
  ChevronDown,
  Minimize2
} from 'lucide-react';
import { cn } from '@/lib/utils';

interface MetricData {
  timestamp: number;
  value: number;
  threshold?: {
    warning: number;
    critical: number;
  };
}

interface SystemHealth {
  status: 'healthy' | 'warning' | 'critical';
  uptime: number;
  responseTime: number;
  errorRate: number;
  activeUsers: number;
  lastUpdated: number;
}

interface PerformanceMetrics {
  coreWebVitals: {
    lcp: MetricData[];
    fid: MetricData[];
    cls: MetricData[];
  };
  userExperience: {
    pageViews: number;
    uniqueUsers: number;
    bounceRate: number;
    avgSessionDuration: number;
  };
  technical: {
    bundleSize: number;
    loadTime: number;
    errorCount: number;
    apiLatency: number;
  };
}

const ProductionMonitoringDashboard: React.FC = () => {
  const [isCollapsed, setIsCollapsed] = useState(true);
  const [isMinimized, setIsMinimized] = useState(false);
  
  const [systemHealth, setSystemHealth] = useState<SystemHealth>({
    status: 'healthy',
    uptime: 99.95,
    responseTime: 245,
    errorRate: 0.02,
    activeUsers: 1247,
    lastUpdated: Date.now()
  });

  const [performanceMetrics, setPerformanceMetrics] = useState<PerformanceMetrics>({
    coreWebVitals: {
      lcp: [
        { timestamp: Date.now() - 3600000, value: 2100 },
        { timestamp: Date.now() - 1800000, value: 1950 },
        { timestamp: Date.now(), value: 1800 }
      ],
      fid: [
        { timestamp: Date.now() - 3600000, value: 45 },
        { timestamp: Date.now() - 1800000, value: 38 },
        { timestamp: Date.now(), value: 42 }
      ],
      cls: [
        { timestamp: Date.now() - 3600000, value: 0.08 },
        { timestamp: Date.now() - 1800000, value: 0.06 },
        { timestamp: Date.now(), value: 0.05 }
      ]
    },
    userExperience: {
      pageViews: 15420,
      uniqueUsers: 3247,
      bounceRate: 0.23,
      avgSessionDuration: 485
    },
    technical: {
      bundleSize: 825, // KB
      loadTime: 1850, // ms
      errorCount: 12,
      apiLatency: 180 // ms
    }
  });

  const [selectedTimeRange, setSelectedTimeRange] = useState<'1h' | '24h' | '7d' | '30d'>('24h');
  const [isRealTime, setIsRealTime] = useState(true);

  // Simulate real-time updates
  useEffect(() => {
    if (!isRealTime) return;

    const interval = setInterval(() => {
      // Update system health
      setSystemHealth(prev => ({
        ...prev,
        responseTime: prev.responseTime + (Math.random() - 0.5) * 50,
        errorRate: Math.max(0, prev.errorRate + (Math.random() - 0.5) * 0.01),
        activeUsers: prev.activeUsers + Math.floor((Math.random() - 0.5) * 20),
        lastUpdated: Date.now()
      }));

      // Update performance metrics
      setPerformanceMetrics(prev => ({
        ...prev,
        coreWebVitals: {
          lcp: [
            ...prev.coreWebVitals.lcp.slice(-9),
            { 
              timestamp: Date.now(), 
              value: 1800 + (Math.random() - 0.5) * 400 
            }
          ],
          fid: [
            ...prev.coreWebVitals.fid.slice(-9),
            { 
              timestamp: Date.now(), 
              value: 42 + (Math.random() - 0.5) * 20 
            }
          ],
          cls: [
            ...prev.coreWebVitals.cls.slice(-9),
            { 
              timestamp: Date.now(), 
              value: Math.max(0, 0.05 + (Math.random() - 0.5) * 0.04) 
            }
          ]
        },
        userExperience: {
          ...prev.userExperience,
          pageViews: prev.userExperience.pageViews + Math.floor(Math.random() * 5),
          uniqueUsers: prev.userExperience.uniqueUsers + Math.floor(Math.random() * 2)
        }
      }));
    }, 5000);

    return () => clearInterval(interval);
  }, [isRealTime]);

  // Calculate status colors and icons
  const getStatusInfo = (status: SystemHealth['status']) => {
    switch (status) {
      case 'healthy':
        return { color: 'text-green-500', bg: 'bg-green-100', icon: CheckCircle };
      case 'warning':
        return { color: 'text-yellow-500', bg: 'bg-yellow-100', icon: AlertTriangle };
      case 'critical':
        return { color: 'text-red-500', bg: 'bg-red-100', icon: XCircle };
      default:
        return { color: 'text-gray-500', bg: 'bg-gray-100', icon: AlertTriangle };
    }
  };

  const statusInfo = getStatusInfo(systemHealth.status);
  const StatusIcon = statusInfo.icon;

  // Performance score calculation
  const performanceScore = useMemo(() => {
    const latestLCP = performanceMetrics.coreWebVitals.lcp.slice(-1)[0]?.value || 0;
    const latestFID = performanceMetrics.coreWebVitals.fid.slice(-1)[0]?.value || 0;
    const latestCLS = performanceMetrics.coreWebVitals.cls.slice(-1)[0]?.value || 0;

    let score = 100;
    
    // LCP scoring (good: <2.5s, poor: >4s)
    if (latestLCP > 4000) score -= 30;
    else if (latestLCP > 2500) score -= 15;
    
    // FID scoring (good: <100ms, poor: >300ms)
    if (latestFID > 300) score -= 25;
    else if (latestFID > 100) score -= 10;
    
    // CLS scoring (good: <0.1, poor: >0.25)
    if (latestCLS > 0.25) score -= 25;
    else if (latestCLS > 0.1) score -= 10;

    return Math.max(0, Math.round(score));
  }, [performanceMetrics.coreWebVitals]);

  const formatDuration = (ms: number) => {
    if (ms < 1000) return `${ms}ms`;
    if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`;
    return `${(ms / 60000).toFixed(1)}m`;
  };

  const formatNumber = (num: number) => {
    if (num >= 1000000) return `${(num / 1000000).toFixed(1)}M`;
    if (num >= 1000) return `${(num / 1000).toFixed(1)}K`;
    return num.toString();
  };

  // If minimized, show just a tiny floating button
  if (isMinimized) {
    return (
      <div className="fixed bottom-4 right-4 z-50">
        <Button
          onClick={() => setIsMinimized(false)}
          size="sm"
          className="bg-white/95 backdrop-blur-sm border shadow-lg text-gray-700 hover:bg-white hover:text-gray-900"
        >
          <Activity className="h-4 w-4" />
        </Button>
      </div>
    );
  }

  return (
    <div className="fixed bottom-4 right-4 z-40 max-w-sm">
      <Card className="bg-white/95 backdrop-blur-sm border shadow-lg">
        <CardHeader className="pb-2">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Activity className="h-4 w-4 text-blue-500" />
              <CardTitle className="text-sm font-medium text-gray-900">System Monitor</CardTitle>
              <Badge className={cn(
                "text-xs px-1.5 py-0.5",
                systemHealth.status === 'healthy' 
                  ? 'bg-green-100 text-green-800' 
                  : systemHealth.status === 'warning' 
                  ? 'bg-yellow-100 text-yellow-800' 
                  : 'bg-red-100 text-red-800'
              )}>
                {systemHealth.status.toUpperCase()}
              </Badge>
            </div>
            <div className="flex items-center gap-1">
              <Button
                onClick={() => setIsCollapsed(!isCollapsed)}
                size="sm"
                variant="ghost"
                className="h-6 w-6 p-0 text-gray-500 hover:text-gray-700"
              >
                {isCollapsed ? <ChevronUp className="h-3 w-3" /> : <ChevronDown className="h-3 w-3" />}
              </Button>
              <Button
                onClick={() => setIsMinimized(true)}
                size="sm"
                variant="ghost"
                className="h-6 w-6 p-0 text-gray-500 hover:text-gray-700"
              >
                <Minimize2 className="h-3 w-3" />
              </Button>
            </div>
          </div>
        </CardHeader>
        
        {!isCollapsed && (
          <CardContent className="pt-0 pb-3">
            <div className="space-y-3">
              {/* Quick Stats */}
              <div className="grid grid-cols-2 gap-2 text-xs">
                <div className="bg-gray-50 rounded p-2">
                  <div className="flex items-center gap-1 text-gray-600">
                    <Clock className="h-3 w-3" />
                    <span>Response</span>
                  </div>
                  <div className="font-semibold text-gray-900">{systemHealth.responseTime}ms</div>
                </div>
                <div className="bg-gray-50 rounded p-2">
                  <div className="flex items-center gap-1 text-gray-600">
                    <Users className="h-3 w-3" />
                    <span>Active</span>
                  </div>
                  <div className="font-semibold text-gray-900">{systemHealth.activeUsers.toLocaleString()}</div>
                </div>
                <div className="bg-gray-50 rounded p-2">
                  <div className="flex items-center gap-1 text-gray-600">
                    <TrendingUp className="h-3 w-3" />
                    <span>Uptime</span>
                  </div>
                  <div className="font-semibold text-gray-900">{systemHealth.uptime}%</div>
                </div>
                <div className="bg-gray-50 rounded p-2">
                  <div className="flex items-center gap-1 text-gray-600">
                    <AlertTriangle className="h-3 w-3" />
                    <span>Errors</span>
                  </div>
                  <div className="font-semibold text-gray-900">{(systemHealth.errorRate * 100).toFixed(2)}%</div>
                </div>
              </div>

              {/* Core Web Vitals - Simplified */}
              <div className="border-t pt-2">
                <div className="text-xs font-medium text-gray-700 mb-2">Core Web Vitals</div>
                <div className="space-y-1">
                  <div className="flex items-center justify-between text-xs">
                    <span className="text-gray-600">LCP</span>
                    <div className="flex items-center gap-1">
                      <span className={cn(
                        "font-semibold",
                        performanceMetrics.coreWebVitals.lcp[performanceMetrics.coreWebVitals.lcp.length - 1]?.value < 2500 
                          ? "text-green-600" 
                          : "text-yellow-600"
                      )}>
                        {performanceMetrics.coreWebVitals.lcp[performanceMetrics.coreWebVitals.lcp.length - 1]?.value || 0}ms
                      </span>
                      {performanceMetrics.coreWebVitals.lcp[performanceMetrics.coreWebVitals.lcp.length - 1]?.value < 2500 
                        ? <CheckCircle className="h-3 w-3 text-green-500" />
                        : <AlertTriangle className="h-3 w-3 text-yellow-500" />
                      }
                    </div>
                  </div>
                  <div className="flex items-center justify-between text-xs">
                    <span className="text-gray-600">FID</span>
                    <div className="flex items-center gap-1">
                      <span className={cn(
                        "font-semibold",
                        performanceMetrics.coreWebVitals.fid[performanceMetrics.coreWebVitals.fid.length - 1]?.value < 100 
                          ? "text-green-600" 
                          : "text-yellow-600"
                      )}>
                        {performanceMetrics.coreWebVitals.fid[performanceMetrics.coreWebVitals.fid.length - 1]?.value || 0}ms
                      </span>
                      {performanceMetrics.coreWebVitals.fid[performanceMetrics.coreWebVitals.fid.length - 1]?.value < 100 
                        ? <CheckCircle className="h-3 w-3 text-green-500" />
                        : <AlertTriangle className="h-3 w-3 text-yellow-500" />
                      }
                    </div>
                  </div>
                  <div className="flex items-center justify-between text-xs">
                    <span className="text-gray-600">CLS</span>
                    <div className="flex items-center gap-1">
                      <span className={cn(
                        "font-semibold",
                        performanceMetrics.coreWebVitals.cls[performanceMetrics.coreWebVitals.cls.length - 1]?.value < 0.1 
                          ? "text-green-600" 
                          : "text-yellow-600"
                      )}>
                        {performanceMetrics.coreWebVitals.cls[performanceMetrics.coreWebVitals.cls.length - 1]?.value?.toFixed(3) || '0.000'}
                      </span>
                      {performanceMetrics.coreWebVitals.cls[performanceMetrics.coreWebVitals.cls.length - 1]?.value < 0.1 
                        ? <CheckCircle className="h-3 w-3 text-green-500" />
                        : <AlertTriangle className="h-3 w-3 text-yellow-500" />
                      }
                    </div>
                  </div>
                </div>
              </div>

              {/* Last Updated */}
              <div className="text-xs text-gray-500 text-center pt-1 border-t">
                Updated: {new Date(systemHealth.lastUpdated).toLocaleTimeString()}
              </div>
            </div>
          </CardContent>
        )}
      </Card>
    </div>
  );
};

export default ProductionMonitoringDashboard;
