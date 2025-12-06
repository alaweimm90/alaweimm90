import React, { useState, useMemo } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/ui/molecules/Card';
import { Button } from '@/ui/atoms/Button';
import { Badge } from '@/ui/atoms/Badge';
import { Progress } from '@/components/ui/progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/ui/molecules/Tabs';
import { Alert, AlertDescription } from '@/ui/molecules/Alert';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar } from 'recharts';
import { 
  Rocket, 
  CheckCircle, 
  XCircle, 
  Clock,
  AlertTriangle,
  Activity,
  GitBranch,
  Server,
  Database,
  Globe,
  ArrowUp,
  ArrowDown,
  RefreshCw,
  Shield,
  Zap
} from 'lucide-react';

interface Deployment {
  id: string;
  version: string;
  environment: 'development' | 'staging' | 'production';
  status: 'pending' | 'deploying' | 'success' | 'failed' | 'rollback' | 'cancelled';
  startTime: string;
  endTime?: string;
  duration?: number;
  commit: {
    hash: string;
    message: string;
    author: string;
  };
  healthChecks: HealthCheck[];
  rollbackAvailable: boolean;
  deployedBy: string;
}

interface HealthCheck {
  id: string;
  name: string;
  type: 'api' | 'database' | 'cache' | 'external' | 'performance';
  status: 'pending' | 'running' | 'passed' | 'failed' | 'warning';
  endpoint?: string;
  responseTime?: number;
  message?: string;
}

interface DeploymentMetric {
  date: string;
  deployments: number;
  success: number;
  failed: number;
  averageDuration: number;
}

export const DeploymentMonitor = () => {
  const [selectedEnvironment, setSelectedEnvironment] = useState<string>('all');
  const [autoRefresh, setAutoRefresh] = useState(true);

  const deployments: Deployment[] = useMemo(() => [
    {
      id: 'deploy-1',
      version: 'v2.4.1',
      environment: 'production',
      status: 'success',
      startTime: '2024-01-15T14:30:00Z',
      endTime: '2024-01-15T14:35:30Z',
      duration: 330,
      commit: {
        hash: 'a7b2c3d',
        message: 'Fix payment processing bug and optimize database queries',
        author: 'John Doe'
      },
      healthChecks: [
        { id: 'hc-1', name: 'API Health', type: 'api', status: 'passed', endpoint: '/health', responseTime: 45 },
        { id: 'hc-2', name: 'Database Connection', type: 'database', status: 'passed', responseTime: 12 },
        { id: 'hc-3', name: 'Cache Layer', type: 'cache', status: 'passed', responseTime: 8 },
        { id: 'hc-4', name: 'Payment Gateway', type: 'external', status: 'passed', responseTime: 156 },
        { id: 'hc-5', name: 'Performance Baseline', type: 'performance', status: 'passed', responseTime: 89 }
      ],
      rollbackAvailable: true,
      deployedBy: 'CI/CD Pipeline'
    },
    {
      id: 'deploy-2',
      version: 'v2.4.0',
      environment: 'staging',
      status: 'deploying',
      startTime: '2024-01-15T14:20:00Z',
      commit: {
        hash: 'f8e9a2b',
        message: 'Add new analytics dashboard features',
        author: 'Jane Smith'
      },
      healthChecks: [
        { id: 'hc-6', name: 'API Health', type: 'api', status: 'passed', endpoint: '/health', responseTime: 32 },
        { id: 'hc-7', name: 'Database Connection', type: 'database', status: 'running' },
        { id: 'hc-8', name: 'Cache Layer', type: 'cache', status: 'pending' },
        { id: 'hc-9', name: 'Payment Gateway', type: 'external', status: 'pending' }
      ],
      rollbackAvailable: false,
      deployedBy: 'Jane Smith'
    },
    {
      id: 'deploy-3',
      version: 'v2.3.8',
      environment: 'production',
      status: 'failed',
      startTime: '2024-01-15T13:45:00Z',
      endTime: '2024-01-15T13:52:15Z',
      duration: 435,
      commit: {
        hash: 'c4d5e6f',
        message: 'Update authentication system',
        author: 'Mike Johnson'
      },
      healthChecks: [
        { id: 'hc-10', name: 'API Health', type: 'api', status: 'failed', endpoint: '/health', message: 'Service unavailable' },
        { id: 'hc-11', name: 'Database Connection', type: 'database', status: 'passed', responseTime: 18 },
        { id: 'hc-12', name: 'Cache Layer', type: 'cache', status: 'failed', message: 'Connection timeout' },
        { id: 'hc-13', name: 'Authentication Service', type: 'external', status: 'failed', message: 'Invalid configuration' }
      ],
      rollbackAvailable: true,
      deployedBy: 'Mike Johnson'
    }
  ], []);

  const deploymentMetrics: DeploymentMetric[] = useMemo(() => {
    const data: DeploymentMetric[] = [];
    
    for (let i = 7; i >= 0; i--) {
      const date = new Date();
      date.setDate(date.getDate() - i);
      
      const deployments = Math.floor(Math.random() * 8) + 2;
      const failed = Math.floor(Math.random() * 2);
      const success = deployments - failed;
      
      data.push({
        date: date.toLocaleDateString(),
        deployments,
        success,
        failed,
        averageDuration: Math.floor(Math.random() * 300) + 200
      });
    }
    
    return data;
  }, []);

  const filteredDeployments = selectedEnvironment === 'all' ? 
    deployments : 
    deployments.filter(d => d.environment === selectedEnvironment);

  const getStatusIcon = (status: Deployment['status']) => {
    switch (status) {
      case 'deploying': return <Activity className="h-4 w-4 text-blue-500 animate-spin" />;
      case 'success': return <CheckCircle className="h-4 w-4 text-green-500" />;
      case 'failed': return <XCircle className="h-4 w-4 text-red-500" />;
      case 'rollback': return <ArrowDown className="h-4 w-4 text-orange-500" />;
      case 'cancelled': return <XCircle className="h-4 w-4 text-gray-500" />;
      default: return <Clock className="h-4 w-4 text-gray-400" />;
    }
  };

  const getStatusColor = (status: Deployment['status']) => {
    switch (status) {
      case 'success': return 'default';
      case 'failed': return 'destructive';
      case 'deploying': return 'secondary';
      case 'rollback': return 'secondary';
      default: return 'outline';
    }
  };

  const getHealthCheckIcon = (status: HealthCheck['status']) => {
    switch (status) {
      case 'running': return <Activity className="h-4 w-4 text-blue-500 animate-spin" />;
      case 'passed': return <CheckCircle className="h-4 w-4 text-green-500" />;
      case 'failed': return <XCircle className="h-4 w-4 text-red-500" />;
      case 'warning': return <AlertTriangle className="h-4 w-4 text-yellow-500" />;
      default: return <Clock className="h-4 w-4 text-gray-400" />;
    }
  };

  const getEnvironmentIcon = (env: Deployment['environment']) => {
    switch (env) {
      case 'production': return <Server className="h-4 w-4 text-red-500" />;
      case 'staging': return <Globe className="h-4 w-4 text-yellow-500" />;
      case 'development': return <GitBranch className="h-4 w-4 text-blue-500" />;
    }
  };

  const getTypeIcon = (type: HealthCheck['type']) => {
    switch (type) {
      case 'api': return <Zap className="h-4 w-4" />;
      case 'database': return <Database className="h-4 w-4" />;
      case 'cache': return <RefreshCw className="h-4 w-4" />;
      case 'external': return <Globe className="h-4 w-4" />;
      case 'performance': return <Activity className="h-4 w-4" />;
    }
  };

  const deploymentStats = {
    total: deployments.length,
    success: deployments.filter(d => d.status === 'success').length,
    failed: deployments.filter(d => d.status === 'failed').length,
    running: deployments.filter(d => d.status === 'deploying').length,
    avgDuration: deployments
      .filter(d => d.duration)
      .reduce((sum, d) => sum + (d.duration || 0), 0) / 
      deployments.filter(d => d.duration).length || 0
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold">Deployment Monitor</h2>
          <p className="text-muted-foreground">Track deployment health and manage rollbacks</p>
        </div>
        <div className="flex gap-2">
          <Button
            variant={autoRefresh ? 'default' : 'outline'}
            size="sm"
            onClick={() => setAutoRefresh(!autoRefresh)}
          >
            <RefreshCw className={`h-4 w-4 mr-2 ${autoRefresh ? 'animate-spin' : ''}`} />
            Auto Refresh
          </Button>
          <Button variant="outline" size="sm" onClick={() => console.log("DeploymentMonitor button clicked")}>
            <Rocket className="h-4 w-4 mr-2" />
            New Deployment
          </Button>
        </div>
      </div>

      {/* Deployment Stats */}
      <div className="grid gap-4 md:grid-cols-5">
        <Card>
          <CardContent className="pt-4">
            <div className="text-2xl font-bold">{deploymentStats.total}</div>
            <p className="text-xs text-muted-foreground">Total Deployments</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-4">
            <div className="text-2xl font-bold text-green-600">{deploymentStats.success}</div>
            <p className="text-xs text-muted-foreground">Successful</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-4">
            <div className="text-2xl font-bold text-red-600">{deploymentStats.failed}</div>
            <p className="text-xs text-muted-foreground">Failed</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-4">
            <div className="text-2xl font-bold text-blue-600">{deploymentStats.running}</div>
            <p className="text-xs text-muted-foreground">In Progress</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-4">
            <div className="text-2xl font-bold">{Math.round(deploymentStats.avgDuration)}s</div>
            <p className="text-xs text-muted-foreground">Avg Duration</p>
          </CardContent>
        </Card>
      </div>

      {/* Failed Deployment Alert */}
      {deploymentStats.failed > 0 && (
        <Alert>
          <AlertTriangle className="h-4 w-4" />
          <AlertDescription>
            <strong>Deployment Issues Detected!</strong> {deploymentStats.failed} failed deployment(s) require attention.
          </AlertDescription>
        </Alert>
      )}

      <Tabs defaultValue="deployments" className="w-full">
        <TabsList>
          <TabsTrigger value="deployments">Deployments</TabsTrigger>
          <TabsTrigger value="health">Health Checks</TabsTrigger>
          <TabsTrigger value="metrics">Deployment Metrics</TabsTrigger>
          <TabsTrigger value="rollback">Rollback Management</TabsTrigger>
        </TabsList>

        <TabsContent value="deployments">
          <div className="space-y-4">
            {/* Environment Filter */}
            <div className="flex gap-2">
              <Button
                variant={selectedEnvironment === 'all' ? 'default' : 'outline'}
                size="sm"
                onClick={() => setSelectedEnvironment('all')}
              >
                All Environments
              </Button>
              {['production', 'staging', 'development'].map((env) => (
                <Button
                  key={env}
                  variant={selectedEnvironment === env ? 'default' : 'outline'}
                  size="sm"
                  onClick={() => setSelectedEnvironment(env)}
                  className="capitalize"
                >
                  {env}
                </Button>
              ))}
            </div>

            {/* Deployment List */}
            <div className="space-y-4">
              {filteredDeployments.map((deployment) => (
                <Card key={deployment.id}>
                  <CardHeader>
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        {getStatusIcon(deployment.status)}
                        <CardTitle className="text-lg">{deployment.version}</CardTitle>
                        <Badge variant={getStatusColor(deployment.status)}>
                          {deployment.status}
                        </Badge>
                      </div>
                      <div className="flex items-center gap-2">
                        {getEnvironmentIcon(deployment.environment)}
                        <Badge variant="outline" className="capitalize">
                          {deployment.environment}
                        </Badge>
                        {deployment.rollbackAvailable && deployment.status === 'failed' && (
                          <Button size="sm" variant="destructive" onClick={() => console.log("DeploymentMonitor button clicked")}>
                            <ArrowDown className="h-4 w-4 mr-2" />
                            Rollback
                          </Button>
                        )}
                      </div>
                    </div>
                  </CardHeader>
                  <CardContent>
                    <div className="grid gap-4 md:grid-cols-2">
                      <div className="space-y-2">
                        <div>
                          <p className="text-sm font-medium">Commit</p>
                          <div className="flex items-center gap-2">
                            <code className="text-sm bg-muted px-2 py-1 rounded">
                              {deployment.commit.hash}
                            </code>
                            <span className="text-sm text-muted-foreground">
                              by {deployment.commit.author}
                            </span>
                          </div>
                          <p className="text-sm text-muted-foreground mt-1">
                            {deployment.commit.message}
                          </p>
                        </div>
                        
                        <div>
                          <p className="text-sm font-medium">Deployment Info</p>
                          <div className="text-sm text-muted-foreground space-y-1">
                            <p>Started: {new Date(deployment.startTime).toLocaleString()}</p>
                            {deployment.endTime && (
                              <p>Completed: {new Date(deployment.endTime).toLocaleString()}</p>
                            )}
                            {deployment.duration && (
                              <p>Duration: {deployment.duration}s</p>
                            )}
                            <p>Deployed by: {deployment.deployedBy}</p>
                          </div>
                        </div>
                      </div>

                      <div className="space-y-2">
                        <p className="text-sm font-medium">Health Checks</p>
                        <div className="space-y-2">
                          {deployment.healthChecks.map((check) => (
                            <div key={check.id} className="flex items-center justify-between p-2 border rounded">
                              <div className="flex items-center gap-2">
                                {getHealthCheckIcon(check.status)}
                                {getTypeIcon(check.type)}
                                <span className="text-sm">{check.name}</span>
                              </div>
                              <div className="text-right text-sm">
                                {check.responseTime && (
                                  <span className="text-muted-foreground">{check.responseTime}ms</span>
                                )}
                                {check.message && (
                                  <p className="text-red-500 text-xs">{check.message}</p>
                                )}
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    </div>

                    {deployment.status === 'deploying' && (
                      <div className="mt-4">
                        <div className="flex justify-between text-sm mb-2">
                          <span>Deployment Progress</span>
                          <span>Running health checks...</span>
                        </div>
                        <Progress value={65} />
                      </div>
                    )}
                  </CardContent>
                </Card>
              ))}
            </div>
          </div>
        </TabsContent>

        <TabsContent value="health">
          <div className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>Health Check Status</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {['api', 'database', 'cache', 'external', 'performance'].map((type) => {
                    const checks = deployments
                      .flatMap(d => d.healthChecks)
                      .filter(hc => hc.type === type);
                    const passed = checks.filter(hc => hc.status === 'passed').length;
                    const total = checks.length;
                    const successRate = total > 0 ? (passed / total) * 100 : 0;
                    
                    return (
                      <div key={type}>
                        <div className="flex justify-between items-center mb-2">
                          <div className="flex items-center gap-2">
                            {getTypeIcon(type as HealthCheck['type'])}
                            <span className="font-medium capitalize">{type} Health</span>
                          </div>
                          <span className="text-sm">{passed}/{total} passed</span>
                        </div>
                        <Progress value={successRate} />
                      </div>
                    );
                  })}
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="metrics">
          <div className="grid gap-6">
            <Card>
              <CardHeader>
                <CardTitle>Deployment Trends</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="h-80">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={deploymentMetrics}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="date" />
                      <YAxis />
                      <Tooltip />
                      <Line 
                        type="monotone" 
                        dataKey="deployments" 
                        stroke="hsl(var(--primary))" 
                        strokeWidth={2}
                        name="Total Deployments"
                      />
                      <Line 
                        type="monotone" 
                        dataKey="success" 
                        stroke="hsl(var(--primary))" 
                        strokeWidth={2}
                        name="Successful"
                      />
                      <Line 
                        type="monotone" 
                        dataKey="failed" 
                        stroke="hsl(var(--destructive))" 
                        strokeWidth={2}
                        name="Failed"
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Deployment Duration</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="h-80">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={deploymentMetrics}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="date" />
                      <YAxis />
                      <Tooltip />
                      <Bar dataKey="averageDuration" fill="hsl(var(--primary))" name="Avg Duration (s)" />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="rollback">
          <div className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <ArrowDown className="h-5 w-5" />
                  Rollback Management
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {deployments.filter(d => d.rollbackAvailable).map((deployment) => (
                    <div key={deployment.id} className="flex items-center justify-between p-4 border rounded-lg">
                      <div>
                        <div className="flex items-center gap-2">
                          {getStatusIcon(deployment.status)}
                          <span className="font-medium">{deployment.version}</span>
                          <Badge variant={getStatusColor(deployment.status)}>
                            {deployment.status}
                          </Badge>
                          <Badge variant="outline" className="capitalize">
                            {deployment.environment}
                          </Badge>
                        </div>
                        <p className="text-sm text-muted-foreground mt-1">
                          {deployment.commit.message}
                        </p>
                        <p className="text-xs text-muted-foreground">
                          Deployed: {new Date(deployment.startTime).toLocaleString()}
                        </p>
                      </div>
                      <div className="flex gap-2">
                        <Button size="sm" variant="outline" onClick={() => console.log("DeploymentMonitor button clicked")}>
                          View Logs
                        </Button>
                        {deployment.status === 'failed' && (
                          <Button size="sm" variant="destructive" onClick={() => console.log("DeploymentMonitor button clicked")}>
                            <ArrowDown className="h-4 w-4 mr-2" />
                            Rollback
                          </Button>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Rollback Strategy</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="prose prose-sm max-w-none">
                  <h4>Automated Rollback Triggers:</h4>
                  <ul>
                    <li>Health check failures exceeding threshold</li>
                    <li>Error rate spikes above 5%</li>
                    <li>Performance degradation beyond acceptable limits</li>
                    <li>External service connectivity issues</li>
                  </ul>
                  
                  <h4>Manual Rollback Process:</h4>
                  <ol>
                    <li>Identify the problematic deployment</li>
                    <li>Verify rollback target version</li>
                    <li>Execute rollback with health check validation</li>
                    <li>Monitor post-rollback system health</li>
                  </ol>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
};