import React, { useState, useEffect } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/ui/molecules/Card';
import { Badge } from '@/ui/atoms/Badge';
import { Button } from '@/ui/atoms/Button';
import { Progress } from '@/components/ui/progress';
import { Alert, AlertDescription } from '@/ui/molecules/Alert';
import { useToast } from '@/hooks/use-toast';
import { 
  Globe, 
  CheckCircle, 
  XCircle, 
  Clock, 
  AlertTriangle,
  Zap,
  Database,
  RefreshCw,
  TrendingUp,
  TrendingDown,
  Activity
} from 'lucide-react';

interface APIEndpoint {
  id: string;
  name: string;
  url: string;
  method: 'GET' | 'POST' | 'PUT' | 'DELETE' | 'PATCH';
  status: 'healthy' | 'degraded' | 'down' | 'maintenance';
  responseTime: number;
  uptime: number;
  lastChecked: Date;
  errorRate: number;
  requestsPerMinute: number;
  successRate: number;
  averageResponseTime: number;
  p95ResponseTime: number;
  p99ResponseTime: number;
}

interface HealthCheck {
  id: string;
  endpoint: string;
  timestamp: Date;
  status: 'success' | 'error' | 'timeout';
  responseTime: number;
  statusCode?: number;
  error?: string;
}

interface Alert {
  id: string;
  endpoint: string;
  type: 'downtime' | 'slow_response' | 'high_error_rate' | 'timeout';
  severity: 'low' | 'medium' | 'high' | 'critical';
  message: string;
  timestamp: Date;
  acknowledged: boolean;
}

export const APIMonitoringDashboard = () => {
  const [endpoints, setEndpoints] = useState<APIEndpoint[]>([]);
  const [healthChecks, setHealthChecks] = useState<HealthCheck[]>([]);
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [isMonitoring, setIsMonitoring] = useState(false);
  const { toast } = useToast();

  const mockEndpoints: APIEndpoint[] = [
    {
      id: 'auth-api',
      name: 'Authentication API',
      url: '/api/auth',
      method: 'POST',
      status: 'healthy',
      responseTime: 145,
      uptime: 99.8,
      lastChecked: new Date(),
      errorRate: 0.2,
      requestsPerMinute: 45,
      successRate: 99.8,
      averageResponseTime: 145,
      p95ResponseTime: 280,
      p99ResponseTime: 450
    },
    {
      id: 'user-api',
      name: 'User Management API',
      url: '/api/users',
      method: 'GET',
      status: 'healthy',
      responseTime: 89,
      uptime: 99.9,
      lastChecked: new Date(Date.now() - 1000 * 30),
      errorRate: 0.1,
      requestsPerMinute: 120,
      successRate: 99.9,
      averageResponseTime: 89,
      p95ResponseTime: 165,
      p99ResponseTime: 245
    },
    {
      id: 'payment-api',
      name: 'Payment Processing API',
      url: '/api/payments',
      method: 'POST',
      status: 'degraded',
      responseTime: 1250,
      uptime: 98.5,
      lastChecked: new Date(Date.now() - 1000 * 45),
      errorRate: 1.5,
      requestsPerMinute: 25,
      successRate: 98.5,
      averageResponseTime: 1250,
      p95ResponseTime: 2100,
      p99ResponseTime: 3500
    },
    {
      id: 'notification-api',
      name: 'Notification Service',
      url: '/api/notifications',
      method: 'POST',
      status: 'down',
      responseTime: 0,
      uptime: 85.2,
      lastChecked: new Date(Date.now() - 1000 * 300),
      errorRate: 15.8,
      requestsPerMinute: 0,
      successRate: 84.2,
      averageResponseTime: 0,
      p95ResponseTime: 0,
      p99ResponseTime: 0
    },
    {
      id: 'analytics-api',
      name: 'Analytics API',
      url: '/api/analytics',
      method: 'GET',
      status: 'healthy',
      responseTime: 234,
      uptime: 99.7,
      lastChecked: new Date(Date.now() - 1000 * 15),
      errorRate: 0.3,
      requestsPerMinute: 15,
      successRate: 99.7,
      averageResponseTime: 234,
      p95ResponseTime: 456,
      p99ResponseTime: 789
    },
    {
      id: 'file-api',
      name: 'File Upload API',
      url: '/api/files',
      method: 'POST',
      status: 'maintenance',
      responseTime: 0,
      uptime: 95.0,
      lastChecked: new Date(Date.now() - 1000 * 120),
      errorRate: 0,
      requestsPerMinute: 0,
      successRate: 100,
      averageResponseTime: 0,
      p95ResponseTime: 0,
      p99ResponseTime: 0
    }
  ];

  const mockAlerts: Alert[] = [
    {
      id: 'alert-1',
      endpoint: 'Payment Processing API',
      type: 'slow_response',
      severity: 'medium',
      message: 'Response time exceeded 1000ms threshold',
      timestamp: new Date(Date.now() - 1000 * 60 * 15),
      acknowledged: false
    },
    {
      id: 'alert-2',
      endpoint: 'Notification Service',
      type: 'downtime',
      severity: 'critical',
      message: 'Service is down - no responses in the last 5 minutes',
      timestamp: new Date(Date.now() - 1000 * 60 * 5),
      acknowledged: false
    },
    {
      id: 'alert-3',
      endpoint: 'Payment Processing API',
      type: 'high_error_rate',
      severity: 'high',
      message: 'Error rate of 1.5% exceeds threshold of 1%',
      timestamp: new Date(Date.now() - 1000 * 60 * 30),
      acknowledged: true
    }
  ];

  const startMonitoring = () => {
    setIsMonitoring(true);
    setEndpoints(mockEndpoints);
    setAlerts(mockAlerts);
    
    toast({
      title: "API Monitoring Started",
      description: "Real-time API monitoring is now active",
    });

    // Simulate continuous monitoring
    const interval = setInterval(() => {
      // Update response times and status
      setEndpoints(prev => prev.map(endpoint => ({
        ...endpoint,
        responseTime: endpoint.status === 'healthy' 
          ? Math.floor(Math.random() * 200) + 50
          : endpoint.status === 'degraded'
          ? Math.floor(Math.random() * 1000) + 500
          : 0,
        lastChecked: new Date(),
        requestsPerMinute: endpoint.status === 'healthy' 
          ? Math.floor(Math.random() * 50) + endpoint.requestsPerMinute
          : 0
      })));
    }, 5000);

    return () => clearInterval(interval);
  };

  const stopMonitoring = () => {
    setIsMonitoring(false);
    
    toast({
      title: "API Monitoring Stopped",
      description: "Real-time API monitoring has been disabled",
    });
  };

  const acknowledgeAlert = (alertId: string) => {
    setAlerts(prev => prev.map(alert => 
      alert.id === alertId ? { ...alert, acknowledged: true } : alert
    ));

    toast({
      title: "Alert Acknowledged",
      description: "Alert has been marked as acknowledged",
    });
  };

  const runHealthCheck = async (endpointId: string) => {
    const endpoint = endpoints.find(e => e.id === endpointId);
    if (!endpoint) return;

    toast({
      title: "Health Check Started",
      description: `Checking ${endpoint.name}...`,
    });

    // Simulate health check
    await new Promise(resolve => setTimeout(resolve, 2000));

    const healthCheck: HealthCheck = {
      id: `check-${Date.now()}`,
      endpoint: endpoint.name,
      timestamp: new Date(),
      status: endpoint.status === 'down' ? 'error' : 'success',
      responseTime: endpoint.responseTime,
      statusCode: endpoint.status === 'down' ? 503 : 200,
      error: endpoint.status === 'down' ? 'Service unavailable' : undefined
    };

    setHealthChecks(prev => [healthCheck, ...prev.slice(0, 19)]); // Keep last 20

    toast({
      title: "Health Check Complete",
      description: `${endpoint.name}: ${healthCheck.status}`,
      variant: healthCheck.status === 'error' ? "destructive" : "default"
    });
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'healthy': return 'text-green-600 bg-green-50 border-green-200';
      case 'degraded': return 'text-yellow-600 bg-yellow-50 border-yellow-200';
      case 'down': return 'text-red-600 bg-red-50 border-red-200';
      case 'maintenance': return 'text-blue-600 bg-blue-50 border-blue-200';
      default: return 'text-gray-600 bg-gray-50 border-gray-200';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'healthy': return <CheckCircle className="h-4 w-4 text-green-600" />;
      case 'degraded': return <AlertTriangle className="h-4 w-4 text-yellow-600" />;
      case 'down': return <XCircle className="h-4 w-4 text-red-600" />;
      case 'maintenance': return <Clock className="h-4 w-4 text-blue-600" />;
      default: return <Clock className="h-4 w-4 text-gray-400" />;
    }
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical': return 'text-red-700 bg-red-100 border-red-200';
      case 'high': return 'text-orange-700 bg-orange-100 border-orange-200';
      case 'medium': return 'text-yellow-700 bg-yellow-100 border-yellow-200';
      case 'low': return 'text-green-700 bg-green-100 border-green-200';
      default: return 'text-gray-700 bg-gray-100 border-gray-200';
    }
  };

  const getMethodColor = (method: string) => {
    switch (method) {
      case 'GET': return 'text-blue-600 bg-blue-50';
      case 'POST': return 'text-green-600 bg-green-50';
      case 'PUT': return 'text-orange-600 bg-orange-50';
      case 'DELETE': return 'text-red-600 bg-red-50';
      case 'PATCH': return 'text-purple-600 bg-purple-50';
      default: return 'text-gray-600 bg-gray-50';
    }
  };

  const getOverallHealth = () => {
    const healthy = endpoints.filter(e => e.status === 'healthy').length;
    const degraded = endpoints.filter(e => e.status === 'degraded').length;
    const down = endpoints.filter(e => e.status === 'down').length;
    const maintenance = endpoints.filter(e => e.status === 'maintenance').length;
    const total = endpoints.length;

    return { healthy, degraded, down, maintenance, total };
  };

  const health = getOverallHealth();

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Globe className="h-6 w-6 text-blue-600" />
          <h2 className="text-2xl font-bold">API Monitoring Dashboard</h2>
        </div>
        <div className="flex gap-2">
          {!isMonitoring ? (
            <Button onClick={startMonitoring} className="gap-2">
              <Activity className="h-4 w-4" />
              Start Monitoring
            </Button>
          ) : (
            <Button onClick={stopMonitoring} variant="outline" className="gap-2">
              <Activity className="h-4 w-4" />
              Stop Monitoring
            </Button>
          )}
        </div>
      </div>

      {/* Overall Health Stats */}
      <div className="grid gap-4 md:grid-cols-5">
        <Card>
          <CardContent className="pt-6">
            <div className="text-2xl font-bold">{health.total}</div>
            <p className="text-xs text-muted-foreground">Total APIs</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-6">
            <div className="text-2xl font-bold text-green-600">{health.healthy}</div>
            <p className="text-xs text-muted-foreground">Healthy</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-6">
            <div className="text-2xl font-bold text-yellow-600">{health.degraded}</div>
            <p className="text-xs text-muted-foreground">Degraded</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-6">
            <div className="text-2xl font-bold text-red-600">{health.down}</div>
            <p className="text-xs text-muted-foreground">Down</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-6">
            <div className="text-2xl font-bold text-blue-600">{health.maintenance}</div>
            <p className="text-xs text-muted-foreground">Maintenance</p>
          </CardContent>
        </Card>
      </div>

      {/* Active Alerts */}
      {alerts.filter(a => !a.acknowledged).length > 0 && (
        <Alert className="border-red-200 bg-red-50">
          <AlertTriangle className="h-4 w-4 text-red-600" />
          <AlertDescription className="text-red-700">
            <strong>{alerts.filter(a => !a.acknowledged).length} active alerts</strong> require attention.
            Check the alerts section below for details.
          </AlertDescription>
        </Alert>
      )}

      {/* API Endpoints */}
      <div className="grid gap-4 md:grid-cols-2">
        {endpoints.map((endpoint) => (
          <Card key={endpoint.id}>
            <CardHeader>
              <CardTitle className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  {getStatusIcon(endpoint.status)}
                  {endpoint.name}
                </div>
                <div className="flex gap-2">
                  <Badge className={getMethodColor(endpoint.method)}>
                    {endpoint.method}
                  </Badge>
                  <Badge className={getStatusColor(endpoint.status)}>
                    {endpoint.status.toUpperCase()}
                  </Badge>
                </div>
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="text-sm text-muted-foreground">
                {endpoint.url}
              </div>

              <div className="grid gap-3 md:grid-cols-2 text-sm">
                <div>
                  <div className="text-muted-foreground">Response Time</div>
                  <div className="font-medium">
                    {endpoint.responseTime > 0 ? `${endpoint.responseTime}ms` : 'N/A'}
                  </div>
                </div>
                <div>
                  <div className="text-muted-foreground">Uptime</div>
                  <div className="font-medium">{endpoint.uptime}%</div>
                </div>
                <div>
                  <div className="text-muted-foreground">Error Rate</div>
                  <div className="font-medium">{endpoint.errorRate}%</div>
                </div>
                <div>
                  <div className="text-muted-foreground">Requests/min</div>
                  <div className="font-medium">{endpoint.requestsPerMinute}</div>
                </div>
              </div>

              <div className="space-y-2">
                <div className="flex justify-between text-xs">
                  <span>Success Rate</span>
                  <span>{endpoint.successRate}%</span>
                </div>
                <Progress value={endpoint.successRate} className="h-2" />
              </div>

              <div className="flex items-center justify-between">
                <div className="text-xs text-muted-foreground">
                  Last checked: {endpoint.lastChecked.toLocaleTimeString()}
                </div>
                <Button
                  onClick={() => runHealthCheck(endpoint.id)}
                  size="sm"
                  variant="outline"
                >
                  <RefreshCw className="h-3 w-3 mr-1" />
                  Check
                </Button>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Alerts */}
      <Card>
        <CardHeader>
          <CardTitle>Recent Alerts</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {alerts.map((alert) => (
              <div key={alert.id} className={`p-3 border rounded ${alert.acknowledged ? 'opacity-50' : ''}`}>
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-2">
                    <AlertTriangle className="h-4 w-4 text-red-600" />
                    <span className="font-medium">{alert.endpoint}</span>
                    <Badge className={getSeverityColor(alert.severity)}>
                      {alert.severity.toUpperCase()}
                    </Badge>
                  </div>
                  {!alert.acknowledged && (
                    <Button
                      onClick={() => acknowledgeAlert(alert.id)}
                      size="sm"
                      variant="outline"
                    >
                      Acknowledge
                    </Button>
                  )}
                </div>
                <div className="text-sm text-muted-foreground mb-1">
                  {alert.message}
                </div>
                <div className="text-xs text-muted-foreground">
                  {alert.timestamp.toLocaleString()}
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Recent Health Checks */}
      {healthChecks.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Recent Health Checks</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              {healthChecks.slice(0, 10).map((check) => (
                <div key={check.id} className="flex items-center justify-between p-2 border rounded text-sm">
                  <div className="flex items-center gap-2">
                    {check.status === 'success' ? (
                      <CheckCircle className="h-4 w-4 text-green-600" />
                    ) : (
                      <XCircle className="h-4 w-4 text-red-600" />
                    )}
                    <span>{check.endpoint}</span>
                  </div>
                  <div className="flex items-center gap-2 text-xs text-muted-foreground">
                    <span>{check.responseTime}ms</span>
                    <span>{check.timestamp.toLocaleTimeString()}</span>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Monitoring Best Practices */}
      <Card>
        <CardHeader>
          <CardTitle>API Monitoring Best Practices</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div>
            <h4 className="font-medium mb-2">🎯 Key Metrics</h4>
            <ul className="space-y-1 text-sm text-muted-foreground">
              <li>• Response time (aim for &lt; 200ms for critical APIs)</li>
              <li>• Uptime (target 99.9% availability)</li>
              <li>• Error rate (keep below 1%)</li>
              <li>• Throughput and request patterns</li>
            </ul>
          </div>
          
          <div>
            <h4 className="font-medium mb-2">🚨 Alerting Strategy</h4>
            <ul className="space-y-1 text-sm text-muted-foreground">
              <li>• Set up alerts for downtime, slow responses, and high error rates</li>
              <li>• Use different severity levels and escalation paths</li>
              <li>• Implement alert fatigue prevention</li>
              <li>• Regular review and tuning of alert thresholds</li>
            </ul>
          </div>
          
          <div>
            <h4 className="font-medium mb-2">📊 Monitoring Coverage</h4>
            <ul className="space-y-1 text-sm text-muted-foreground">
              <li>• Monitor all critical API endpoints</li>
              <li>• Include synthetic transaction monitoring</li>
              <li>• Track downstream dependencies</li>
              <li>• Monitor from multiple geographic locations</li>
            </ul>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};