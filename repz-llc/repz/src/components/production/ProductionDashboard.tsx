import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/ui/molecules/Card';
import { Button } from '@/ui/atoms/Button';
import { Badge } from '@/ui/atoms/Badge';
import { Progress } from '@/components/ui/progress';
import { 
  Activity, 
  AlertTriangle, 
  CheckCircle, 
  Clock, 
  Server, 
  Database, 
  Zap,
  Users,
  TrendingUp,
  Shield,
  Globe,
  RefreshCw
} from 'lucide-react';

interface SystemMetric {
  id: string;
  name: string;
  value: number;
  unit: string;
  status: 'healthy' | 'warning' | 'critical';
  threshold: number;
  description: string;
}

interface SystemAlert {
  id: string;
  type: 'error' | 'warning' | 'info';
  message: string;
  timestamp: Date;
  component: string;
  resolved: boolean;
}

const ProductionDashboard: React.FC = () => {
  const [metrics, setMetrics] = useState<SystemMetric[]>([
    { id: '1', name: 'Response Time', value: 245, unit: 'ms', status: 'healthy', threshold: 500, description: 'Average API response time' },
    { id: '2', name: 'CPU Usage', value: 45, unit: '%', status: 'healthy', threshold: 80, description: 'Server CPU utilization' },
    { id: '3', name: 'Memory Usage', value: 62, unit: '%', status: 'warning', threshold: 85, description: 'Server memory utilization' },
    { id: '4', name: 'Database Connections', value: 15, unit: 'active', status: 'healthy', threshold: 100, description: 'Active database connections' },
    { id: '5', name: 'Error Rate', value: 0.2, unit: '%', status: 'healthy', threshold: 1, description: 'Application error rate' },
    { id: '6', name: 'Active Users', value: 234, unit: 'users', status: 'healthy', threshold: 1000, description: 'Currently active users' }
  ]);

  const [alerts, setAlerts] = useState<SystemAlert[]>([
    { id: '1', type: 'warning', message: 'Memory usage approaching limit', timestamp: new Date(Date.now() - 300000), component: 'Server', resolved: false },
    { id: '2', type: 'error', message: 'Payment webhook timeout', timestamp: new Date(Date.now() - 600000), component: 'Stripe', resolved: true },
    { id: '3', type: 'info', message: 'Database backup completed', timestamp: new Date(Date.now() - 1800000), component: 'Database', resolved: true }
  ]);

  const [deploymentStatus, setDeploymentStatus] = useState({
    version: '2.1.4',
    deployedAt: new Date(Date.now() - 7200000),
    environment: 'production',
    healthy: true,
    lastCheck: new Date()
  });

  const [isRefreshing, setIsRefreshing] = useState(false);

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'healthy': return 'text-green-500';
      case 'warning': return 'text-yellow-500';
      case 'critical': return 'text-red-500';
      default: return 'text-gray-500';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'healthy': return <CheckCircle className="h-4 w-4 text-green-500" />;
      case 'warning': return <AlertTriangle className="h-4 w-4 text-yellow-500" />;
      case 'critical': return <AlertTriangle className="h-4 w-4 text-red-500" />;
      default: return <Clock className="h-4 w-4 text-gray-500" />;
    }
  };

  const refreshMetrics = async () => {
    setIsRefreshing(true);
    
    // Simulate API call
    await new Promise(resolve => setTimeout(resolve, 1500));
    
    // Update metrics with random variations
    setMetrics(prev => prev.map(metric => ({
      ...metric,
      value: metric.value + (Math.random() - 0.5) * 20,
      status: metric.value > metric.threshold * 0.9 ? 'warning' : 
              metric.value > metric.threshold ? 'critical' : 'healthy'
    })));
    
    setDeploymentStatus(prev => ({ ...prev, lastCheck: new Date() }));
    setIsRefreshing(false);
  };

  const resolveAlert = (alertId: string) => {
    setAlerts(prev => prev.map(alert => 
      alert.id === alertId ? { ...alert, resolved: true } : alert
    ));
  };

  const healthyMetrics = metrics.filter(m => m.status === 'healthy').length;
  const warningMetrics = metrics.filter(m => m.status === 'warning').length;
  const criticalMetrics = metrics.filter(m => m.status === 'critical').length;
  const overallHealth = (healthyMetrics / metrics.length) * 100;

  const activeAlerts = alerts.filter(a => !a.resolved);

  return (
    <div className="space-y-6">
      {/* System Overview */}
      <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card>
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-muted-foreground">System Health</p>
                <p className="text-2xl font-bold">{Math.round(overallHealth)}%</p>
              </div>
              <Activity className="h-8 w-8 text-green-500" />
            </div>
            <Progress value={overallHealth} className="mt-2" />
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-muted-foreground">Active Alerts</p>
                <p className="text-2xl font-bold">{activeAlerts.length}</p>
              </div>
              <AlertTriangle className="h-8 w-8 text-yellow-500" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-muted-foreground">Uptime</p>
                <p className="text-2xl font-bold">99.9%</p>
              </div>
              <Server className="h-8 w-8 text-blue-500" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-muted-foreground">Version</p>
                <p className="text-2xl font-bold">{deploymentStatus.version}</p>
              </div>
              <Globe className="h-8 w-8 text-purple-500" />
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Metrics Grid */}
      <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
        {metrics.map((metric) => (
          <Card key={metric.id}>
            <CardContent className="p-6">
              <div className="flex items-center justify-between mb-2">
                <h3 className="font-medium text-sm">{metric.name}</h3>
                {getStatusIcon(metric.status)}
              </div>
              <div className="space-y-2">
                <div className="flex items-baseline gap-2">
                  <span className={`text-2xl font-bold ${getStatusColor(metric.status)}`}>
                    {typeof metric.value === 'number' ? metric.value.toFixed(metric.unit === '%' ? 1 : 0) : metric.value}
                  </span>
                  <span className="text-sm text-muted-foreground">{metric.unit}</span>
                </div>
                <p className="text-xs text-muted-foreground">{metric.description}</p>
                <Progress 
                  value={(metric.value / metric.threshold) * 100} 
                  className="h-1"
                />
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Alerts */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle>System Alerts</CardTitle>
            <Button 
              variant="outline" 
              size="sm" 
              onClick={refreshMetrics}
              disabled={isRefreshing}
            >
              <RefreshCw className={`h-4 w-4 mr-2 ${isRefreshing ? 'animate-spin' : ''}`} />
              Refresh
            </Button>
          </div>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {alerts.length === 0 ? (
              <p className="text-muted-foreground text-sm">No alerts at this time</p>
            ) : (
              alerts.map((alert) => (
                <div 
                  key={alert.id}
                  className={`p-3 border rounded-lg ${alert.resolved ? 'opacity-60' : ''}`}
                >
                  <div className="flex items-start justify-between">
                    <div className="flex items-start gap-3">
                      {alert.type === 'error' && <AlertTriangle className="h-4 w-4 text-red-500 mt-0.5" />}
                      {alert.type === 'warning' && <AlertTriangle className="h-4 w-4 text-yellow-500 mt-0.5" />}
                      {alert.type === 'info' && <CheckCircle className="h-4 w-4 text-blue-500 mt-0.5" />}
                      <div className="flex-1">
                        <p className="font-medium text-sm">{alert.message}</p>
                        <div className="flex items-center gap-2 text-xs text-muted-foreground mt-1">
                          <span>{alert.component}</span>
                          <span>•</span>
                          <span>{alert.timestamp.toLocaleTimeString()}</span>
                        </div>
                      </div>
                    </div>
                    <div className="flex items-center gap-2">
                      <Badge variant={alert.resolved ? 'secondary' : 'outline'}>
                        {alert.resolved ? 'Resolved' : 'Active'}
                      </Badge>
                      {!alert.resolved && (
                        <Button 
                          variant="outline" 
                          size="sm"
                          onClick={() => resolveAlert(alert.id)}
                        >
                          Resolve
                        </Button>
                      )}
                    </div>
                  </div>
                </div>
              ))
            )}
          </div>
        </CardContent>
      </Card>

      {/* Deployment Info */}
      <Card>
        <CardHeader>
          <CardTitle>Deployment Status</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid md:grid-cols-2 gap-6">
            <div className="space-y-3">
              <div className="flex justify-between">
                <span className="text-sm font-medium">Environment:</span>
                <Badge variant="default">{deploymentStatus.environment}</Badge>
              </div>
              <div className="flex justify-between">
                <span className="text-sm font-medium">Version:</span>
                <span className="text-sm">{deploymentStatus.version}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-sm font-medium">Deployed:</span>
                <span className="text-sm">{deploymentStatus.deployedAt.toLocaleString()}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-sm font-medium">Last Check:</span>
                <span className="text-sm">{deploymentStatus.lastCheck.toLocaleTimeString()}</span>
              </div>
            </div>
            <div className="space-y-3">
              <div className="flex items-center gap-2">
                <Shield className="h-4 w-4 text-green-500" />
                <span className="text-sm">Security headers configured</span>
              </div>
              <div className="flex items-center gap-2">
                <Database className="h-4 w-4 text-green-500" />
                <span className="text-sm">Database migrations applied</span>
              </div>
              <div className="flex items-center gap-2">
                <Zap className="h-4 w-4 text-green-500" />
                <span className="text-sm">Edge functions deployed</span>
              </div>
              <div className="flex items-center gap-2">
                <Users className="h-4 w-4 text-green-500" />
                <span className="text-sm">Authentication configured</span>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default ProductionDashboard;