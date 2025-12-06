import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { 
  useMonitoringDashboard, 
  useMetricStream, 
  useAlertNotifications,
  usePerformanceInsights 
} from '@/hooks/useMonitoringDashboard';
import {
  Activity,
  AlertTriangle,
  CheckCircle,
  Clock,
  Cpu,
  Database,
  Download,
  Eye,
  EyeOff,
  MemoryStick,
  RefreshCw,
  Zap,
  TrendingUp,
  TrendingDown,
  Minus,
  Bell,
  BellOff
} from 'lucide-react';

// Health Score Indicator
const HealthScoreIndicator: React.FC<{ score: number }> = ({ score }) => {
  const getHealthColor = (score: number) => {
    if (score >= 80) return 'text-green-500';
    if (score >= 60) return 'text-yellow-500';
    return 'text-red-500';
  };

  const getHealthIcon = (score: number) => {
    if (score >= 80) return <CheckCircle className="h-4 w-4" />;
    if (score >= 60) return <AlertTriangle className="h-4 w-4" />;
    return <AlertTriangle className="h-4 w-4" />;
  };

  return (
    <div className={`flex items-center gap-2 ${getHealthColor(score)}`}>
      {getHealthIcon(score)}
      <span className="text-2xl font-bold">{score.toFixed(1)}%</span>
    </div>
  );
};

// Metric Chart Component
const MetricChart: React.FC<{
  data: number[];
  title: string;
  unit: string;
  color?: string;
}> = ({ data, title, unit, color = 'blue' }) => {
  if (data.length === 0) {
    return (
      <div className="h-20 flex items-center justify-center text-gray-400">
        No data available
      </div>
    );
  }

  const max = Math.max(...data);
  const min = Math.min(...data);
  const range = max - min || 1;

  const points = data.map((value, index) => {
    const x = (index / (data.length - 1)) * 100;
    const y = 100 - ((value - min) / range) * 100;
    return `${x},${y}`;
  }).join(' ');

  const currentValue = data[data.length - 1];
  const previousValue = data[data.length - 2];
  const trend = previousValue ? currentValue - previousValue : 0;

  const getTrendIcon = () => {
    if (trend > 0) return <TrendingUp className="h-3 w-3 text-green-500" />;
    if (trend < 0) return <TrendingDown className="h-3 w-3 text-red-500" />;
    return <Minus className="h-3 w-3 text-gray-400" />;
  };

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <h4 className="text-sm font-medium">{title}</h4>
        <div className="flex items-center gap-1">
          {getTrendIcon()}
          <span className="text-sm text-gray-600">
            {currentValue.toFixed(2)} {unit}
          </span>
        </div>
      </div>
      <div className="h-16 relative">
        <svg viewBox="0 0 100 100" className="w-full h-full">
          <polyline
            fill="none"
            stroke={color === 'blue' ? '#3b82f6' : color === 'green' ? '#10b981' : '#ef4444'}
            strokeWidth="2"
            points={points}
            className="drop-shadow-sm"
          />
          <circle
            cx={data.length > 1 ? ((data.length - 1) / (data.length - 1)) * 100 : 50}
            cy={data.length > 0 ? 100 - ((currentValue - min) / range) * 100 : 50}
            r="2"
            fill={color === 'blue' ? '#3b82f6' : color === 'green' ? '#10b981' : '#ef4444'}
          />
        </svg>
      </div>
    </div>
  );
};

// Alert Item Component
const AlertItem: React.FC<{
  alert: any;
  onAcknowledge: (id: string) => void;
}> = ({ alert, onAcknowledge }) => {
  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical': return 'destructive';
      case 'warning': return 'secondary';
      default: return 'outline';
    }
  };

  const getSeverityIcon = (severity: string) => {
    switch (severity) {
      case 'critical': return <AlertTriangle className="h-4 w-4" />;
      case 'warning': return <AlertTriangle className="h-4 w-4" />;
      default: return <Activity className="h-4 w-4" />;
    }
  };

  return (
    <div className="flex items-center justify-between p-3 border rounded-lg">
      <div className="flex items-center gap-3">
        <div className="flex items-center gap-2">
          {getSeverityIcon(alert.rule.severity)}
          <Badge variant={getSeverityColor(alert.rule.severity) as any}>
            {alert.rule.severity}
          </Badge>
        </div>
        <div>
          <p className="text-sm font-medium">{alert.rule.name}</p>
          <p className="text-xs text-gray-500">{alert.message}</p>
          <p className="text-xs text-gray-400">
            {new Date(alert.triggeredAt).toLocaleString()}
          </p>
        </div>
      </div>
      <Button
        size="sm"
        variant="outline"
        onClick={() => onAcknowledge(alert.id)}
      >
        Acknowledge
      </Button>
    </div>
  );
};

// Performance Insight Component
const PerformanceInsight: React.FC<{ insight: any }> = ({ insight }) => {
  const getInsightColor = (impact: string) => {
    switch (impact) {
      case 'high': return 'border-red-200 bg-red-50';
      case 'medium': return 'border-yellow-200 bg-yellow-50';
      default: return 'border-blue-200 bg-blue-50';
    }
  };

  const getInsightIcon = (type: string) => {
    switch (type) {
      case 'optimization': return <Zap className="h-4 w-4 text-orange-500" />;
      case 'warning': return <AlertTriangle className="h-4 w-4 text-red-500" />;
      default: return <Activity className="h-4 w-4 text-blue-500" />;
    }
  };

  return (
    <div className={`p-4 rounded-lg border ${getInsightColor(insight.impact)}`}>
      <div className="flex items-start gap-3">
        {getInsightIcon(insight.type)}
        <div className="space-y-1">
          <h4 className="text-sm font-semibold">{insight.title}</h4>
          <p className="text-sm text-gray-600">{insight.description}</p>
          <p className="text-xs font-medium text-gray-800">
            💡 {insight.recommendation}
          </p>
        </div>
      </div>
    </div>
  );
};

// Main Monitoring Dashboard Component
export const MonitoringDashboard: React.FC = () => {
  const {
    metrics,
    isLoading,
    lastUpdate,
    refreshInterval,
    autoRefresh,
    refresh,
    setAutoRefresh,
    exportMetrics,
    acknowledgeAlert
  } = useMonitoringDashboard();

  const { notifications, unreadCount, markAllAsRead, requestNotificationPermission } = useAlertNotifications();
  const performanceInsights = usePerformanceInsights();
  
  const [selectedTab, setSelectedTab] = useState('overview');

  // Format uptime
  const formatUptime = (ms: number) => {
    const seconds = Math.floor(ms / 1000);
    const minutes = Math.floor(seconds / 60);
    const hours = Math.floor(minutes / 60);
    const days = Math.floor(hours / 24);

    if (days > 0) return `${days}d ${hours % 24}h`;
    if (hours > 0) return `${hours}h ${minutes % 60}m`;
    return `${minutes}m ${seconds % 60}s`;
  };

  // Export metrics
  const handleExport = (format: 'json' | 'prometheus') => {
    const data = exportMetrics(format);
    const blob = new Blob([data], { 
      type: format === 'json' ? 'application/json' : 'text/plain' 
    });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `qmlab-metrics-${Date.now()}.${format === 'json' ? 'json' : 'txt'}`;
    link.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="space-y-6 p-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">Quantum System Monitoring</h1>
          <p className="text-gray-600">
            Real-time performance metrics and system health
          </p>
        </div>
        <div className="flex items-center gap-3">
          <Button
            size="sm"
            variant="outline"
            onClick={() => requestNotificationPermission()}
          >
            {unreadCount > 0 ? <Bell className="h-4 w-4" /> : <BellOff className="h-4 w-4" />}
            {unreadCount > 0 && (
              <Badge variant="destructive" className="ml-1">
                {unreadCount}
              </Badge>
            )}
          </Button>
          <Button
            size="sm"
            variant="outline"
            onClick={() => setAutoRefresh(!autoRefresh)}
          >
            {autoRefresh ? <Eye className="h-4 w-4" /> : <EyeOff className="h-4 w-4" />}
            Auto-refresh
          </Button>
          <Button
            size="sm"
            variant="outline"
            onClick={refresh}
            disabled={isLoading}
          >
            <RefreshCw className={`h-4 w-4 ${isLoading ? 'animate-spin' : ''}`} />
            Refresh
          </Button>
        </div>
      </div>

      {/* System Health Overview */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="flex items-center gap-2 text-sm">
              <Activity className="h-4 w-4" />
              System Health
            </CardTitle>
          </CardHeader>
          <CardContent>
            <HealthScoreIndicator score={metrics.quantumHealth} />
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="flex items-center gap-2 text-sm">
              <AlertTriangle className="h-4 w-4" />
              Active Alerts
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-red-500">
              {metrics.activeAlerts.length}
            </div>
            <p className="text-xs text-gray-500">
              {metrics.activeAlerts.filter(a => a.rule.severity === 'critical').length} critical
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="flex items-center gap-2 text-sm">
              <Cpu className="h-4 w-4" />
              Active Sessions
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {metrics.systemStats.totalCircuits + metrics.systemStats.totalTraining}
            </div>
            <p className="text-xs text-gray-500">
              {metrics.systemStats.totalCircuits} circuits, {metrics.systemStats.totalTraining} training
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="flex items-center gap-2 text-sm">
              <Clock className="h-4 w-4" />
              System Uptime
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {formatUptime(metrics.systemStats.uptime)}
            </div>
            <p className="text-xs text-gray-500">
              Since {new Date(Date.now() - metrics.systemStats.uptime).toLocaleDateString()}
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Tabs for different views */}
      <Tabs value={selectedTab} onValueChange={setSelectedTab}>
        <TabsList>
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="metrics">Metrics</TabsTrigger>
          <TabsTrigger value="alerts">Alerts</TabsTrigger>
          <TabsTrigger value="insights">Insights</TabsTrigger>
        </TabsList>

        <TabsContent value="overview" className="space-y-6">
          {/* Performance Metrics */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            <Card>
              <CardHeader className="pb-3">
                <CardTitle className="text-sm">Circuit Execution Time</CardTitle>
              </CardHeader>
              <CardContent>
                <MetricChart
                  data={metrics.recentMetrics.circuitExecutionTime}
                  title=""
                  unit="ms"
                  color="blue"
                />
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-3">
                <CardTitle className="text-sm">Training Loss</CardTitle>
              </CardHeader>
              <CardContent>
                <MetricChart
                  data={metrics.recentMetrics.trainingLoss}
                  title=""
                  unit=""
                  color="green"
                />
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-3">
                <CardTitle className="text-sm">Memory Usage</CardTitle>
              </CardHeader>
              <CardContent>
                <MetricChart
                  data={metrics.recentMetrics.simulationMemory}
                  title=""
                  unit="MB"
                  color="orange"
                />
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-3">
                <CardTitle className="text-sm">Error Rate</CardTitle>
              </CardHeader>
              <CardContent>
                <MetricChart
                  data={metrics.recentMetrics.errorRates}
                  title=""
                  unit="%"
                  color="red"
                />
              </CardContent>
            </Card>
          </div>

          {/* Recent Alerts */}
          {metrics.activeAlerts.length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <AlertTriangle className="h-4 w-4" />
                  Recent Alerts
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {metrics.activeAlerts.slice(0, 3).map(alert => (
                    <AlertItem
                      key={alert.id}
                      alert={alert}
                      onAcknowledge={acknowledgeAlert}
                    />
                  ))}
                </div>
              </CardContent>
            </Card>
          )}
        </TabsContent>

        <TabsContent value="metrics" className="space-y-6">
          <div className="flex items-center justify-between">
            <h3 className="text-lg font-semibold">System Metrics</h3>
            <div className="flex gap-2">
              <Button 
                size="sm" 
                variant="outline" 
                onClick={() => handleExport('json')}
              >
                <Download className="h-4 w-4 mr-2" />
                Export JSON
              </Button>
              <Button 
                size="sm" 
                variant="outline" 
                onClick={() => handleExport('prometheus')}
              >
                <Download className="h-4 w-4 mr-2" />
                Export Prometheus
              </Button>
            </div>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle>System Statistics</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex justify-between">
                  <span>Total Circuits:</span>
                  <span className="font-mono">{metrics.systemStats.totalCircuits}</span>
                </div>
                <div className="flex justify-between">
                  <span>Training Sessions:</span>
                  <span className="font-mono">{metrics.systemStats.totalTraining}</span>
                </div>
                <div className="flex justify-between">
                  <span>Simulations:</span>
                  <span className="font-mono">{metrics.systemStats.totalSimulations}</span>
                </div>
                <div className="flex justify-between">
                  <span>Last Update:</span>
                  <span className="font-mono text-sm">
                    {new Date(lastUpdate).toLocaleString()}
                  </span>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Configuration</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex justify-between">
                  <span>Refresh Interval:</span>
                  <span className="font-mono">{refreshInterval / 1000}s</span>
                </div>
                <div className="flex justify-between">
                  <span>Auto-refresh:</span>
                  <span className="font-mono">{autoRefresh ? 'Enabled' : 'Disabled'}</span>
                </div>
                <div className="flex justify-between">
                  <span>Data Retention:</span>
                  <span className="font-mono">1 hour</span>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="alerts">
          <Card>
            <CardHeader className="flex flex-row items-center justify-between">
              <CardTitle>Alert Management</CardTitle>
              {unreadCount > 0 && (
                <Button size="sm" onClick={markAllAsRead}>
                  Mark all as read
                </Button>
              )}
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {metrics.activeAlerts.length === 0 ? (
                  <div className="text-center py-8 text-gray-500">
                    <CheckCircle className="h-12 w-12 mx-auto mb-4 text-green-500" />
                    <p>No active alerts</p>
                    <p className="text-sm">System is running smoothly</p>
                  </div>
                ) : (
                  metrics.activeAlerts.map(alert => (
                    <AlertItem
                      key={alert.id}
                      alert={alert}
                      onAcknowledge={acknowledgeAlert}
                    />
                  ))
                )}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="insights">
          <div className="space-y-4">
            <h3 className="text-lg font-semibold">Performance Insights</h3>
            {performanceInsights.length === 0 ? (
              <Card>
                <CardContent className="text-center py-8 text-gray-500">
                  <Activity className="h-12 w-12 mx-auto mb-4 text-green-500" />
                  <p>No performance issues detected</p>
                  <p className="text-sm">System is performing optimally</p>
                </CardContent>
              </Card>
            ) : (
              <div className="grid gap-4">
                {performanceInsights.map((insight, index) => (
                  <PerformanceInsight key={index} insight={insight} />
                ))}
              </div>
            )}
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default MonitoringDashboard;