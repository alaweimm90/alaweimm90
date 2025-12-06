import React, { useState, useEffect } from 'react';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import {
  AlertTriangle,
  CheckCircle,
  Clock,
  Cpu,
  Database,
  Globe,
  HardDrive,
  MemoryStick,
  RefreshCw,
  Server,
  Shield,
  TrendingUp,
  TrendingDown,
  Wifi,
  Zap,
  Activity,
  Bell,
  Settings,
  Download,
  Eye,
  AlertCircle,
  XCircle,
  Timer,
  Users,
  BarChart3,
  WifiOff,
  CloudOff,
} from 'lucide-react';
import { ChartContainer, ChartTooltip, ChartTooltipContent } from '@/components/ui/chart';
import { LineChart, Line, XAxis, YAxis, ResponsiveContainer, AreaChart, Area } from 'recharts';
import { trackEvent } from '@/lib/analytics';

interface SystemMetric {
  timestamp: string;
  cpu: number;
  memory: number;
  disk: number;
  network: number;
  responseTime: number;
  activeConnections: number;
  errorRate: number;
}

interface ServiceStatus {
  id: string;
  name: string;
  status: 'healthy' | 'degraded' | 'down' | 'maintenance';
  uptime: number;
  lastChecked: string;
  responseTime: number;
  errorRate: number;
  version: string;
  dependencies: string[];
}

interface Alert {
  id: string;
  type: 'error' | 'warning' | 'info';
  title: string;
  message: string;
  service: string;
  timestamp: string;
  acknowledged: boolean;
  resolved: boolean;
}

const mockSystemMetrics: SystemMetric[] = [
  {
    timestamp: '10:00',
    cpu: 45,
    memory: 67,
    disk: 23,
    network: 89,
    responseTime: 120,
    activeConnections: 156,
    errorRate: 0.2,
  },
  {
    timestamp: '10:05',
    cpu: 52,
    memory: 71,
    disk: 24,
    network: 92,
    responseTime: 135,
    activeConnections: 178,
    errorRate: 0.1,
  },
  {
    timestamp: '10:10',
    cpu: 38,
    memory: 64,
    disk: 23,
    network: 76,
    responseTime: 98,
    activeConnections: 134,
    errorRate: 0.3,
  },
  {
    timestamp: '10:15',
    cpu: 61,
    memory: 78,
    disk: 25,
    network: 85,
    responseTime: 156,
    activeConnections: 203,
    errorRate: 0.4,
  },
  {
    timestamp: '10:20',
    cpu: 43,
    memory: 69,
    disk: 24,
    network: 91,
    responseTime: 112,
    activeConnections: 167,
    errorRate: 0.1,
  },
  {
    timestamp: '10:25',
    cpu: 47,
    memory: 72,
    disk: 23,
    network: 88,
    responseTime: 128,
    activeConnections: 189,
    errorRate: 0.2,
  },
];

const mockServices: ServiceStatus[] = [
  {
    id: 'api-gateway',
    name: 'API Gateway',
    status: 'healthy',
    uptime: 99.9,
    lastChecked: '1 min ago',
    responseTime: 45,
    errorRate: 0.1,
    version: '2.1.4',
    dependencies: ['auth-service', 'evaluation-engine'],
  },
  {
    id: 'evaluation-engine',
    name: 'Evaluation Engine',
    status: 'healthy',
    uptime: 99.7,
    lastChecked: '2 min ago',
    responseTime: 230,
    errorRate: 0.3,
    version: '1.8.2',
    dependencies: ['model-apis', 'result-store'],
  },
  {
    id: 'auth-service',
    name: 'Authentication Service',
    status: 'healthy',
    uptime: 99.95,
    lastChecked: '30 sec ago',
    responseTime: 23,
    errorRate: 0.05,
    version: '3.2.1',
    dependencies: ['user-db', 'session-store'],
  },
  {
    id: 'model-apis',
    name: 'Model API Aggregator',
    status: 'degraded',
    uptime: 98.2,
    lastChecked: '3 min ago',
    responseTime: 1200,
    errorRate: 2.1,
    version: '1.5.7',
    dependencies: ['openai-api', 'anthropic-api', 'google-api'],
  },
  {
    id: 'result-store',
    name: 'Result Storage',
    status: 'healthy',
    uptime: 99.8,
    lastChecked: '1 min ago',
    responseTime: 67,
    errorRate: 0.2,
    version: '2.3.0',
    dependencies: ['database', 'cache-layer'],
  },
  {
    id: 'notification-service',
    name: 'Notification Service',
    status: 'maintenance',
    uptime: 95.1,
    lastChecked: '10 min ago',
    responseTime: 0,
    errorRate: 0,
    version: '1.4.3',
    dependencies: ['email-provider', 'webhook-dispatcher'],
  },
];

const mockAlerts: Alert[] = [
  {
    id: 'alert-001',
    type: 'warning',
    title: 'High Response Time',
    message: 'Model API Aggregator response time exceeded threshold (>1000ms)',
    service: 'model-apis',
    timestamp: '2 min ago',
    acknowledged: false,
    resolved: false,
  },
  {
    id: 'alert-002',
    type: 'info',
    title: 'Scheduled Maintenance',
    message: 'Notification Service maintenance window started',
    service: 'notification-service',
    timestamp: '10 min ago',
    acknowledged: true,
    resolved: false,
  },
  {
    id: 'alert-003',
    type: 'error',
    title: 'API Rate Limit',
    message: 'OpenAI API rate limit reached, fallback to Claude activated',
    service: 'evaluation-engine',
    timestamp: '15 min ago',
    acknowledged: true,
    resolved: true,
  },
  {
    id: 'alert-004',
    type: 'warning',
    title: 'Memory Usage High',
    message: 'System memory usage at 78%, consider scaling',
    service: 'api-gateway',
    timestamp: '20 min ago',
    acknowledged: false,
    resolved: false,
  },
];

const chartConfig = {
  cpu: { label: 'CPU %', color: 'hsl(var(--primary))' },
  memory: { label: 'Memory %', color: 'hsl(var(--accent))' },
  disk: { label: 'Disk %', color: 'hsl(var(--secondary))' },
  responseTime: { label: 'Response Time (ms)', color: 'hsl(var(--primary))' },
  activeConnections: { label: 'Connections', color: 'hsl(var(--accent))' },
  errorRate: { label: 'Error Rate %', color: 'hsl(214 84% 56%)' },
};

export const SystemHealthDashboard = () => {
  const [systemMetrics, setSystemMetrics] = useState<SystemMetric[]>(mockSystemMetrics);
  const [services, setServices] = useState<ServiceStatus[]>(mockServices);
  const [alerts, setAlerts] = useState<Alert[]>(mockAlerts);
  const [lastUpdated, setLastUpdated] = useState(new Date());

  // Simulate real-time updates
  useEffect(() => {
    const interval = setInterval(() => {
      // Update system metrics
      const newTime = new Date();
      const timeString = newTime.toLocaleTimeString('en-US', { 
        hour12: false, 
        hour: '2-digit', 
        minute: '2-digit' 
      });
      
      setSystemMetrics(prev => [
        ...prev.slice(-5),
        {
          timestamp: timeString,
          cpu: Math.max(20, Math.min(80, prev[prev.length - 1]?.cpu + (Math.random() - 0.5) * 20)),
          memory: Math.max(40, Math.min(85, prev[prev.length - 1]?.memory + (Math.random() - 0.5) * 10)),
          disk: Math.max(15, Math.min(40, prev[prev.length - 1]?.disk + (Math.random() - 0.5) * 5)),
          network: Math.max(60, Math.min(100, prev[prev.length - 1]?.network + (Math.random() - 0.5) * 15)),
          responseTime: Math.max(80, Math.min(200, prev[prev.length - 1]?.responseTime + (Math.random() - 0.5) * 40)),
          activeConnections: Math.max(100, Math.min(250, prev[prev.length - 1]?.activeConnections + Math.floor((Math.random() - 0.5) * 30))),
          errorRate: Math.max(0, Math.min(1, prev[prev.length - 1]?.errorRate + (Math.random() - 0.5) * 0.2)),
        },
      ]);

      setLastUpdated(new Date());
    }, 5000);

    return () => clearInterval(interval);
  }, []);

  const getStatusIcon = (status: ServiceStatus['status']) => {
    switch (status) {
      case 'healthy':
        return <CheckCircle className="h-4 w-4 text-green-500" />;
      case 'degraded':
        return <AlertTriangle className="h-4 w-4 text-yellow-500" />;
      case 'down':
        return <XCircle className="h-4 w-4 text-red-500" />;
      case 'maintenance':
        return <Settings className="h-4 w-4 text-blue-500" />;
    }
  };

  const getStatusColor = (status: ServiceStatus['status']) => {
    switch (status) {
      case 'healthy':
        return 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400';
      case 'degraded':
        return 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-400';
      case 'down':
        return 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-400';
      case 'maintenance':
        return 'bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-400';
    }
  };

  const getAlertIcon = (type: Alert['type']) => {
    switch (type) {
      case 'error':
        return <AlertCircle className="h-4 w-4 text-red-500" />;
      case 'warning':
        return <AlertTriangle className="h-4 w-4 text-yellow-500" />;
      case 'info':
        return <Bell className="h-4 w-4 text-blue-500" />;
    }
  };

  const getAlertColor = (type: Alert['type']) => {
    switch (type) {
      case 'error':
        return 'bg-red-50 border-red-200 dark:bg-red-900/20 dark:border-red-800';
      case 'warning':
        return 'bg-yellow-50 border-yellow-200 dark:bg-yellow-900/20 dark:border-yellow-800';
      case 'info':
        return 'bg-blue-50 border-blue-200 dark:bg-blue-900/20 dark:border-blue-800';
    }
  };

  const handleAcknowledgeAlert = (alertId: string) => {
    setAlerts(prev => prev.map(alert => 
      alert.id === alertId ? { ...alert, acknowledged: true } : alert
    ));
    trackEvent('alert_acknowledged', { alertId });
  };

  const handleRefresh = () => {
    trackEvent('system_health_refresh');
    setLastUpdated(new Date());
  };

  const healthyServices = services.filter(s => s.status === 'healthy').length;
  const degradedServices = services.filter(s => s.status === 'degraded').length;
  const downServices = services.filter(s => s.status === 'down').length;
  const avgUptime = services.reduce((sum, service) => sum + service.uptime, 0) / services.length;
  const avgResponseTime = services.filter(s => s.status !== 'maintenance').reduce((sum, service) => sum + service.responseTime, 0) / services.filter(s => s.status !== 'maintenance').length;

  const currentMetrics = systemMetrics[systemMetrics.length - 1];
  const unacknowledgedAlerts = alerts.filter(a => !a.acknowledged && !a.resolved).length;

  return (
    <div className="space-y-6">
      {/* System Overview */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card className="p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-muted-foreground">System Status</p>
              <p className="text-2xl font-bold text-green-600">Operational</p>
              <div className="flex items-center gap-1 mt-1">
                <CheckCircle className="h-3 w-3 text-green-500" />
                <span className="text-xs text-green-600">{avgUptime.toFixed(1)}% uptime</span>
              </div>
            </div>
            <Server className="h-8 w-8 text-green-500" />
          </div>
        </Card>

        <Card className="p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-muted-foreground">Active Alerts</p>
              <p className="text-2xl font-bold text-foreground">{unacknowledgedAlerts}</p>
              <div className="flex items-center gap-1 mt-1">
                {unacknowledgedAlerts > 0 ? (
                  <>
                    <AlertTriangle className="h-3 w-3 text-yellow-500" />
                    <span className="text-xs text-yellow-600">Needs attention</span>
                  </>
                ) : (
                  <>
                    <CheckCircle className="h-3 w-3 text-green-500" />
                    <span className="text-xs text-green-600">All clear</span>
                  </>
                )}
              </div>
            </div>
            <Bell className="h-8 w-8 text-primary" />
          </div>
        </Card>

        <Card className="p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-muted-foreground">Avg Response Time</p>
              <p className="text-2xl font-bold text-foreground">{avgResponseTime.toFixed(0)}ms</p>
              <div className="flex items-center gap-1 mt-1">
                <Timer className="h-3 w-3 text-primary" />
                <span className="text-xs text-muted-foreground">Last 5 minutes</span>
              </div>
            </div>
            <Zap className="h-8 w-8 text-primary" />
          </div>
        </Card>

        <Card className="p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-muted-foreground">Active Connections</p>
              <p className="text-2xl font-bold text-foreground">{currentMetrics?.activeConnections || 0}</p>
              <div className="flex items-center gap-1 mt-1">
                <Activity className="h-3 w-3 text-primary" />
                <span className="text-xs text-muted-foreground">Real-time</span>
              </div>
            </div>
            <Users className="h-8 w-8 text-primary" />
          </div>
        </Card>
      </div>

      {/* System Resources */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Card className="p-4">
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center gap-2">
              <Cpu className="h-5 w-5 text-primary" />
              <span className="font-medium">CPU Usage</span>
            </div>
            <span className="text-sm font-medium">{currentMetrics?.cpu || 0}%</span>
          </div>
          <Progress value={currentMetrics?.cpu || 0} className="h-2" />
          <p className="text-xs text-muted-foreground mt-2">
            {currentMetrics?.cpu > 70 ? 'High usage detected' : 'Normal operation'}
          </p>
        </Card>

        <Card className="p-4">
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center gap-2">
              <MemoryStick className="h-5 w-5 text-accent" />
              <span className="font-medium">Memory Usage</span>
            </div>
            <span className="text-sm font-medium">{currentMetrics?.memory || 0}%</span>
          </div>
          <Progress value={currentMetrics?.memory || 0} className="h-2" />
          <p className="text-xs text-muted-foreground mt-2">
            {currentMetrics?.memory > 80 ? 'Consider scaling up' : 'Optimal usage'}
          </p>
        </Card>

        <Card className="p-4">
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center gap-2">
              <HardDrive className="h-5 w-5 text-secondary" />
              <span className="font-medium">Disk Usage</span>
            </div>
            <span className="text-sm font-medium">{currentMetrics?.disk || 0}%</span>
          </div>
          <Progress value={currentMetrics?.disk || 0} className="h-2" />
          <p className="text-xs text-muted-foreground mt-2">
            Plenty of storage available
          </p>
        </Card>
      </div>

      {/* Real-time Monitoring */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card className="p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-bold text-foreground">System Performance</h3>
            <div className="flex items-center gap-2 text-sm text-muted-foreground">
              <RefreshCw className="h-4 w-4" />
              Updated: {lastUpdated.toLocaleTimeString()}
            </div>
          </div>
          <ChartContainer config={chartConfig} className="h-[200px]">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={systemMetrics}>
                <XAxis dataKey="timestamp" />
                <YAxis />
                <ChartTooltip content={<ChartTooltipContent />} />
                <Area 
                  type="monotone" 
                  dataKey="cpu" 
                  stroke="var(--color-cpu)" 
                  fill="var(--color-cpu)"
                  fillOpacity={0.1}
                />
                <Area 
                  type="monotone" 
                  dataKey="memory" 
                  stroke="var(--color-memory)" 
                  fill="var(--color-memory)"
                  fillOpacity={0.1}
                />
              </AreaChart>
            </ResponsiveContainer>
          </ChartContainer>
        </Card>

        <Card className="p-6">
          <h3 className="text-lg font-bold text-foreground mb-4">Response Time & Connections</h3>
          <ChartContainer config={chartConfig} className="h-[200px]">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={systemMetrics}>
                <XAxis dataKey="timestamp" />
                <YAxis yAxisId="left" />
                <YAxis yAxisId="right" orientation="right" />
                <ChartTooltip content={<ChartTooltipContent />} />
                <Line 
                  yAxisId="left"
                  type="monotone" 
                  dataKey="responseTime" 
                  stroke="var(--color-responseTime)" 
                  strokeWidth={2}
                />
                <Line 
                  yAxisId="right"
                  type="monotone" 
                  dataKey="activeConnections" 
                  stroke="var(--color-activeConnections)" 
                  strokeWidth={2}
                />
              </LineChart>
            </ResponsiveContainer>
          </ChartContainer>
        </Card>
      </div>

      {/* Service Status */}
      <Card className="p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-bold text-foreground">Service Status</h3>
          <Button variant="outline" size="sm" onClick={handleRefresh}>
            <RefreshCw className="h-4 w-4 mr-2" />
            Refresh
          </Button>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {services.map((service) => (
            <Card key={service.id} className="p-4">
              <div className="flex items-start justify-between mb-3">
                <div className="flex items-center gap-2">
                  {getStatusIcon(service.status)}
                  <span className="font-medium">{service.name}</span>
                </div>
                <Badge className={getStatusColor(service.status)}>
                  {service.status}
                </Badge>
              </div>

              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Uptime</span>
                  <span className="font-medium">{service.uptime}%</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Response</span>
                  <span className="font-medium">
                    {service.status === 'maintenance' ? 'N/A' : `${service.responseTime}ms`}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Error Rate</span>
                  <span className="font-medium">{service.errorRate}%</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Version</span>
                  <span className="font-medium">{service.version}</span>
                </div>
              </div>

              <div className="mt-3 pt-3 border-t border-border">
                <p className="text-xs text-muted-foreground">
                  Last checked: {service.lastChecked}
                </p>
                {service.dependencies.length > 0 && (
                  <p className="text-xs text-muted-foreground mt-1">
                    Dependencies: {service.dependencies.join(', ')}
                  </p>
                )}
              </div>
            </Card>
          ))}
        </div>
      </Card>

      {/* Active Alerts */}
      <Card className="p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-bold text-foreground">
            Active Alerts ({unacknowledgedAlerts})
          </h3>
          <div className="flex gap-2">
            <Button variant="outline" size="sm">
              <Download className="h-4 w-4 mr-2" />
              Export Logs
            </Button>
            <Button variant="outline" size="sm">
              <Settings className="h-4 w-4 mr-2" />
              Alert Settings
            </Button>
          </div>
        </div>
        
        <div className="space-y-3">
          {alerts.slice(0, 5).map((alert) => (
            <div 
              key={alert.id} 
              className={`p-4 rounded-lg border ${getAlertColor(alert.type)} ${
                alert.acknowledged ? 'opacity-60' : ''
              }`}
            >
              <div className="flex items-start justify-between">
                <div className="flex items-start gap-3">
                  {getAlertIcon(alert.type)}
                  <div>
                    <h4 className="font-medium text-foreground">{alert.title}</h4>
                    <p className="text-sm text-muted-foreground mt-1">{alert.message}</p>
                    <div className="flex items-center gap-4 mt-2 text-xs text-muted-foreground">
                      <span>Service: {alert.service}</span>
                      <span>{alert.timestamp}</span>
                    </div>
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  {alert.resolved && (
                    <Badge className="bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400">
                      Resolved
                    </Badge>
                  )}
                  {alert.acknowledged && !alert.resolved && (
                    <Badge variant="outline">Acknowledged</Badge>
                  )}
                  {!alert.acknowledged && !alert.resolved && (
                    <Button 
                      size="sm" 
                      variant="outline"
                      onClick={() => handleAcknowledgeAlert(alert.id)}
                    >
                      Acknowledge
                    </Button>
                  )}
                </div>
              </div>
            </div>
          ))}
          
          {alerts.length === 0 && (
            <div className="text-center py-8 text-muted-foreground">
              <CheckCircle className="h-12 w-12 mx-auto mb-4 text-green-500" />
              <p>No active alerts. All systems operational.</p>
            </div>
          )}
        </div>
      </Card>
    </div>
  );
};