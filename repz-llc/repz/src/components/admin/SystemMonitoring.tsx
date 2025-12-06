import React, { useState, useEffect, useMemo } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/ui/molecules/Card';
import { Badge } from '@/ui/atoms/Badge';
import { Button } from '@/ui/atoms/Button';
import {
  Server,
  Database,
  Activity,
  AlertTriangle,
  CheckCircle,
  XCircle,
  Clock,
  Zap,
  TrendingUp,
  TrendingDown,
  RefreshCw,
  Terminal,
  Eye,
  Download,
  Globe,
  HardDrive,
  Cpu,
  MemoryStick
} from 'lucide-react';
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer
} from 'recharts';
import { cn } from '@/lib/utils';

interface HealthMetrics {
  status: 'healthy' | 'warning' | 'critical';
  uptime: number;
  cpu: number;
  memory: number;
  disk: number;
  responseTime: number;
  requestsPerMinute: number;
  errorRate: number;
  activeConnections: number;
}

interface ErrorLog {
  id: string;
  timestamp: string;
  level: 'error' | 'warning' | 'info';
  message: string;
  service: string;
  details?: string;
}

interface ApiMetrics {
  endpoint: string;
  avgResponseTime: number;
  requestCount: number;
  errorCount: number;
  status: 'healthy' | 'degraded' | 'down';
}

const SystemMonitoring: React.FC = () => {
  const [isAutoRefresh, setIsAutoRefresh] = useState(true);
  const [refreshInterval, setRefreshInterval] = useState(5000);
  const [selectedLogLevel, setSelectedLogLevel] = useState<'all' | 'error' | 'warning' | 'info'>('all');

  // Mock health metrics
  const [healthMetrics, setHealthMetrics] = useState<HealthMetrics>({
    status: 'healthy',
    uptime: 99.97,
    cpu: 45.2,
    memory: 68.5,
    disk: 42.8,
    responseTime: 142,
    requestsPerMinute: 2450,
    errorRate: 0.12,
    activeConnections: 1847
  });

  // Mock performance data
  const [performanceData, setPerformanceData] = useState(() => {
    const now = Date.now();
    return Array.from({ length: 20 }, (_, i) => ({
      time: new Date(now - (19 - i) * 30000).toLocaleTimeString(),
      responseTime: 120 + Math.random() * 50,
      cpu: 40 + Math.random() * 30,
      memory: 60 + Math.random() * 20,
      requests: 2000 + Math.random() * 1000
    }));
  });

  // Mock error logs
  const [errorLogs] = useState<ErrorLog[]>([
    {
      id: 'ERR-001',
      timestamp: '2025-11-12 14:23:45',
      level: 'error',
      message: 'Database connection timeout',
      service: 'API Gateway',
      details: 'Connection to primary database timed out after 30 seconds'
    },
    {
      id: 'ERR-002',
      timestamp: '2025-11-12 14:20:12',
      level: 'warning',
      message: 'High memory usage detected',
      service: 'App Server',
      details: 'Memory usage exceeded 80% threshold'
    },
    {
      id: 'ERR-003',
      timestamp: '2025-11-12 14:15:33',
      level: 'error',
      message: 'Stripe webhook verification failed',
      service: 'Payment Service',
      details: 'Webhook signature mismatch for event evt_123456'
    },
    {
      id: 'ERR-004',
      timestamp: '2025-11-12 14:10:08',
      level: 'info',
      message: 'Scheduled backup completed',
      service: 'Backup Service',
      details: 'Database backup completed successfully (2.3GB)'
    },
    {
      id: 'ERR-005',
      timestamp: '2025-11-12 14:05:22',
      level: 'warning',
      message: 'Rate limit approaching',
      service: 'API Gateway',
      details: 'Client 192.168.1.100 approaching rate limit (950/1000 requests)'
    },
    {
      id: 'ERR-006',
      timestamp: '2025-11-12 14:00:45',
      level: 'error',
      message: 'Email delivery failure',
      service: 'Email Service',
      details: 'Failed to deliver welcome email to user@example.com'
    }
  ]);

  // Mock API metrics
  const apiMetrics: ApiMetrics[] = [
    {
      endpoint: '/api/users',
      avgResponseTime: 125,
      requestCount: 45230,
      errorCount: 12,
      status: 'healthy'
    },
    {
      endpoint: '/api/subscriptions',
      avgResponseTime: 185,
      requestCount: 23890,
      errorCount: 45,
      status: 'degraded'
    },
    {
      endpoint: '/api/workouts',
      avgResponseTime: 95,
      requestCount: 78450,
      errorCount: 5,
      status: 'healthy'
    },
    {
      endpoint: '/api/analytics',
      avgResponseTime: 450,
      requestCount: 12340,
      errorCount: 2,
      status: 'healthy'
    }
  ];

  // Mock database query performance
  const dbQueries = [
    { query: 'SELECT * FROM users WHERE...', avgTime: 12, execCount: 15420, slowest: 145 },
    { query: 'UPDATE subscriptions SET...', avgTime: 45, execCount: 3240, slowest: 420 },
    { query: 'INSERT INTO workout_logs...', avgTime: 8, execCount: 28940, slowest: 78 },
    { query: 'SELECT * FROM analytics...', avgTime: 230, execCount: 1240, slowest: 1250 }
  ];

  // Auto-refresh logic
  useEffect(() => {
    if (!isAutoRefresh) return;

    const interval = setInterval(() => {
      // Update health metrics
      setHealthMetrics((prev) => ({
        ...prev,
        cpu: Math.max(0, Math.min(100, prev.cpu + (Math.random() - 0.5) * 10)),
        memory: Math.max(0, Math.min(100, prev.memory + (Math.random() - 0.5) * 5)),
        responseTime: Math.max(50, prev.responseTime + (Math.random() - 0.5) * 30),
        requestsPerMinute: Math.max(1000, prev.requestsPerMinute + (Math.random() - 0.5) * 200),
        errorRate: Math.max(0, prev.errorRate + (Math.random() - 0.5) * 0.05)
      }));

      // Update performance data
      setPerformanceData((prev) => {
        const newData = [...prev.slice(1)];
        newData.push({
          time: new Date().toLocaleTimeString(),
          responseTime: 120 + Math.random() * 50,
          cpu: 40 + Math.random() * 30,
          memory: 60 + Math.random() * 20,
          requests: 2000 + Math.random() * 1000
        });
        return newData;
      });
    }, refreshInterval);

    return () => clearInterval(interval);
  }, [isAutoRefresh, refreshInterval]);

  const getStatusInfo = (status: string) => {
    switch (status) {
      case 'healthy':
        return { color: 'text-green-600', bg: 'bg-green-100', icon: CheckCircle };
      case 'warning':
      case 'degraded':
        return { color: 'text-yellow-600', bg: 'bg-yellow-100', icon: AlertTriangle };
      case 'critical':
      case 'down':
        return { color: 'text-red-600', bg: 'bg-red-100', icon: XCircle };
      default:
        return { color: 'text-gray-600', bg: 'bg-gray-100', icon: Activity };
    }
  };

  const statusInfo = getStatusInfo(healthMetrics.status);
  const StatusIcon = statusInfo.icon;

  const getMetricStatus = (value: number, warning: number, critical: number) => {
    if (value >= critical) return 'critical';
    if (value >= warning) return 'warning';
    return 'healthy';
  };

  const filteredLogs = useMemo(() => {
    if (selectedLogLevel === 'all') return errorLogs;
    return errorLogs.filter((log) => log.level === selectedLogLevel);
  }, [errorLogs, selectedLogLevel]);

  const getLogLevelClass = (level: string) => {
    switch (level) {
      case 'error':
        return 'bg-red-100 text-red-800';
      case 'warning':
        return 'bg-yellow-100 text-yellow-800';
      case 'info':
        return 'bg-blue-100 text-blue-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };

  return (
    <div className="space-y-6 p-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">System Monitoring</h1>
          <p className="text-gray-600 mt-1">Real-time system health and performance metrics</p>
        </div>
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2">
            <span className="text-sm text-gray-600">Auto-refresh:</span>
            <Button
              variant={isAutoRefresh ? 'default' : 'outline'}
              size="sm"
              onClick={() => setIsAutoRefresh(!isAutoRefresh)}
            >
              {isAutoRefresh ? 'On' : 'Off'}
            </Button>
          </div>
          <Button variant="outline" size="sm">
            <Download className="h-4 w-4 mr-2" />
            Export Logs
          </Button>
          <Button variant="outline" size="sm">
            <RefreshCw className={cn('h-4 w-4 mr-2', isAutoRefresh && 'animate-spin')} />
            Refresh
          </Button>
        </div>
      </div>

      {/* System Status Overview */}
      <Card className={cn('border-2', statusInfo.bg)}>
        <CardContent className="pt-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <div className={cn('h-16 w-16 rounded-full flex items-center justify-center', statusInfo.bg)}>
                <StatusIcon className={cn('h-8 w-8', statusInfo.color)} />
              </div>
              <div>
                <h3 className="text-2xl font-bold text-gray-900">
                  System Status: {healthMetrics.status.charAt(0).toUpperCase() + healthMetrics.status.slice(1)}
                </h3>
                <p className="text-gray-600">Uptime: {healthMetrics.uptime}% (Last 30 days)</p>
              </div>
            </div>
            <div className="grid grid-cols-2 gap-4">
              <div className="text-right">
                <p className="text-sm text-gray-600">Active Connections</p>
                <p className="text-2xl font-bold text-gray-900">{healthMetrics.activeConnections.toLocaleString()}</p>
              </div>
              <div className="text-right">
                <p className="text-sm text-gray-600">Requests/Min</p>
                <p className="text-2xl font-bold text-gray-900">{healthMetrics.requestsPerMinute.toLocaleString()}</p>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Server Health Indicators */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2">
                <Cpu className="h-5 w-5 text-blue-600" />
                <span className="text-sm font-medium text-gray-700">CPU Usage</span>
              </div>
              <Badge className={cn('text-xs', getStatusInfo(getMetricStatus(healthMetrics.cpu, 70, 90)).bg)}>
                {getMetricStatus(healthMetrics.cpu, 70, 90)}
              </Badge>
            </div>
            <div className="space-y-2">
              <div className="flex items-baseline gap-2">
                <span className="text-3xl font-bold text-gray-900">{healthMetrics.cpu.toFixed(1)}%</span>
                <span className="text-sm text-gray-600">of capacity</span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div
                  className={cn(
                    'h-2 rounded-full transition-all',
                    getMetricStatus(healthMetrics.cpu, 70, 90) === 'critical' ? 'bg-red-500' :
                    getMetricStatus(healthMetrics.cpu, 70, 90) === 'warning' ? 'bg-yellow-500' :
                    'bg-green-500'
                  )}
                  style={{ width: `${healthMetrics.cpu}%` }}
                />
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2">
                <MemoryStick className="h-5 w-5 text-purple-600" />
                <span className="text-sm font-medium text-gray-700">Memory Usage</span>
              </div>
              <Badge className={cn('text-xs', getStatusInfo(getMetricStatus(healthMetrics.memory, 75, 90)).bg)}>
                {getMetricStatus(healthMetrics.memory, 75, 90)}
              </Badge>
            </div>
            <div className="space-y-2">
              <div className="flex items-baseline gap-2">
                <span className="text-3xl font-bold text-gray-900">{healthMetrics.memory.toFixed(1)}%</span>
                <span className="text-sm text-gray-600">of capacity</span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div
                  className={cn(
                    'h-2 rounded-full transition-all',
                    getMetricStatus(healthMetrics.memory, 75, 90) === 'critical' ? 'bg-red-500' :
                    getMetricStatus(healthMetrics.memory, 75, 90) === 'warning' ? 'bg-yellow-500' :
                    'bg-green-500'
                  )}
                  style={{ width: `${healthMetrics.memory}%` }}
                />
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2">
                <HardDrive className="h-5 w-5 text-orange-600" />
                <span className="text-sm font-medium text-gray-700">Disk Usage</span>
              </div>
              <Badge className={cn('text-xs', getStatusInfo(getMetricStatus(healthMetrics.disk, 80, 95)).bg)}>
                {getMetricStatus(healthMetrics.disk, 80, 95)}
              </Badge>
            </div>
            <div className="space-y-2">
              <div className="flex items-baseline gap-2">
                <span className="text-3xl font-bold text-gray-900">{healthMetrics.disk.toFixed(1)}%</span>
                <span className="text-sm text-gray-600">of capacity</span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div
                  className="h-2 rounded-full bg-green-500 transition-all"
                  style={{ width: `${healthMetrics.disk}%` }}
                />
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2">
                <Zap className="h-5 w-5 text-green-600" />
                <span className="text-sm font-medium text-gray-700">Response Time</span>
              </div>
              <Badge className={cn('text-xs', getStatusInfo(getMetricStatus(healthMetrics.responseTime, 200, 500)).bg)}>
                {getMetricStatus(healthMetrics.responseTime, 200, 500)}
              </Badge>
            </div>
            <div className="space-y-2">
              <div className="flex items-baseline gap-2">
                <span className="text-3xl font-bold text-gray-900">{Math.round(healthMetrics.responseTime)}</span>
                <span className="text-sm text-gray-600">ms avg</span>
              </div>
              <div className="flex items-center gap-1 text-sm text-gray-600">
                <Clock className="h-3 w-3" />
                <span>Error rate: {healthMetrics.errorRate.toFixed(2)}%</span>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Performance Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle className="text-lg font-semibold">Response Time & Requests</CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={250}>
              <AreaChart data={performanceData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                <XAxis dataKey="time" stroke="#666" fontSize={12} />
                <YAxis stroke="#666" fontSize={12} />
                <Tooltip contentStyle={{ background: '#fff', border: '1px solid #e5e7eb', borderRadius: '8px' }} />
                <Legend />
                <Area
                  type="monotone"
                  dataKey="responseTime"
                  stroke="#3b82f6"
                  fill="#3b82f6"
                  fillOpacity={0.2}
                  name="Response Time (ms)"
                />
                <Area
                  type="monotone"
                  dataKey="requests"
                  stroke="#10b981"
                  fill="#10b981"
                  fillOpacity={0.2}
                  name="Requests/Min"
                />
              </AreaChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-lg font-semibold">Server Resources</CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={250}>
              <LineChart data={performanceData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                <XAxis dataKey="time" stroke="#666" fontSize={12} />
                <YAxis stroke="#666" fontSize={12} />
                <Tooltip contentStyle={{ background: '#fff', border: '1px solid #e5e7eb', borderRadius: '8px' }} />
                <Legend />
                <Line
                  type="monotone"
                  dataKey="cpu"
                  stroke="#8b5cf6"
                  strokeWidth={2}
                  dot={false}
                  name="CPU %"
                />
                <Line
                  type="monotone"
                  dataKey="memory"
                  stroke="#f59e0b"
                  strokeWidth={2}
                  dot={false}
                  name="Memory %"
                />
              </LineChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </div>

      {/* API Response Time Metrics */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg font-semibold">API Response Time Metrics</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-gray-200">
                  <th className="text-left py-3 px-4 text-sm font-medium text-gray-700">Endpoint</th>
                  <th className="text-left py-3 px-4 text-sm font-medium text-gray-700">Avg Response Time</th>
                  <th className="text-left py-3 px-4 text-sm font-medium text-gray-700">Total Requests</th>
                  <th className="text-left py-3 px-4 text-sm font-medium text-gray-700">Errors</th>
                  <th className="text-left py-3 px-4 text-sm font-medium text-gray-700">Status</th>
                </tr>
              </thead>
              <tbody>
                {apiMetrics.map((api, index) => (
                  <tr key={index} className="border-b border-gray-100 hover:bg-gray-50">
                    <td className="py-3 px-4 text-sm font-mono text-gray-900">{api.endpoint}</td>
                    <td className="py-3 px-4 text-sm text-gray-900">{api.avgResponseTime}ms</td>
                    <td className="py-3 px-4 text-sm text-gray-900">{api.requestCount.toLocaleString()}</td>
                    <td className="py-3 px-4 text-sm text-gray-900">{api.errorCount}</td>
                    <td className="py-3 px-4">
                      <Badge className={cn('text-xs', getStatusInfo(api.status).bg, getStatusInfo(api.status).color)}>
                        {api.status}
                      </Badge>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>

      {/* Database Query Performance */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg font-semibold">Database Query Performance</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-gray-200">
                  <th className="text-left py-3 px-4 text-sm font-medium text-gray-700">Query</th>
                  <th className="text-left py-3 px-4 text-sm font-medium text-gray-700">Avg Time</th>
                  <th className="text-left py-3 px-4 text-sm font-medium text-gray-700">Executions</th>
                  <th className="text-left py-3 px-4 text-sm font-medium text-gray-700">Slowest</th>
                </tr>
              </thead>
              <tbody>
                {dbQueries.map((query, index) => (
                  <tr key={index} className="border-b border-gray-100 hover:bg-gray-50">
                    <td className="py-3 px-4 text-sm font-mono text-gray-900 max-w-md truncate">{query.query}</td>
                    <td className="py-3 px-4 text-sm text-gray-900">{query.avgTime}ms</td>
                    <td className="py-3 px-4 text-sm text-gray-900">{query.execCount.toLocaleString()}</td>
                    <td className="py-3 px-4 text-sm text-gray-900">{query.slowest}ms</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>

      {/* Error Logs */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle className="text-lg font-semibold">Error Log Viewer (Last 100)</CardTitle>
            <div className="flex items-center gap-2">
              <select
                value={selectedLogLevel}
                onChange={(e) => setSelectedLogLevel(e.target.value as any)}
                className="h-9 px-3 rounded-md border border-gray-300 bg-white text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="all">All Levels</option>
                <option value="error">Errors</option>
                <option value="warning">Warnings</option>
                <option value="info">Info</option>
              </select>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <div className="space-y-2">
            {filteredLogs.map((log) => (
              <div key={log.id} className="border border-gray-200 rounded-lg p-4 hover:bg-gray-50">
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-2">
                      <Badge className={cn('text-xs', getLogLevelClass(log.level))}>
                        {log.level.toUpperCase()}
                      </Badge>
                      <span className="text-sm text-gray-600">{log.service}</span>
                      <span className="text-xs text-gray-500 font-mono">{log.timestamp}</span>
                    </div>
                    <p className="text-sm font-medium text-gray-900 mb-1">{log.message}</p>
                    {log.details && <p className="text-sm text-gray-600">{log.details}</p>}
                  </div>
                  <Button variant="ghost" size="sm" className="h-8 w-8 p-0">
                    <Eye className="h-4 w-4" />
                  </Button>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default SystemMonitoring;
