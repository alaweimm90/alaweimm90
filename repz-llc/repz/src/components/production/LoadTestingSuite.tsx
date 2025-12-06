import React, { useState, useMemo } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/ui/molecules/Card';
import { Button } from '@/ui/atoms/Button';
import { Progress } from '@/components/ui/progress';
import { Badge } from '@/ui/atoms/Badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/ui/molecules/Tabs';
import { Input } from '@/ui/atoms/Input';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/ui/molecules/Select';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area } from 'recharts';
import { 
  Zap, 
  Activity, 
  Clock,
  Users,
  TrendingUp,
  TrendingDown,
  AlertTriangle,
  CheckCircle,
  XCircle,
  Play,
  Square,
  Settings,
  Target
} from 'lucide-react';

interface LoadTest {
  id: string;
  name: string;
  type: 'spike' | 'load' | 'stress' | 'volume' | 'endurance';
  status: 'idle' | 'running' | 'completed' | 'failed' | 'cancelled';
  targetUrl: string;
  virtualUsers: number;
  duration: number;
  rampUpTime: number;
  startTime?: string;
  endTime?: string;
  results?: LoadTestResults;
}

interface LoadTestResults {
  averageResponseTime: number;
  maxResponseTime: number;
  minResponseTime: number;
  requestsPerSecond: number;
  totalRequests: number;
  successRate: number;
  errorRate: number;
  throughput: number;
  bottlenecks: string[];
}

interface MetricData {
  timestamp: string;
  responseTime: number;
  throughput: number;
  errorRate: number;
  activeUsers: number;
  cpuUsage: number;
  memoryUsage: number;
}

export const LoadTestingSuite = () => {
  const [isRunning, setIsRunning] = useState(false);
  const [selectedTest, setSelectedTest] = useState<string>('');
  const [currentMetrics, setCurrentMetrics] = useState<MetricData[]>([]);

  const [loadTests, setLoadTests] = useState<LoadTest[]>([
    {
      id: 'test-1',
      name: 'Homepage Load Test',
      type: 'load',
      status: 'completed',
      targetUrl: '/dashboard',
      virtualUsers: 100,
      duration: 300,
      rampUpTime: 60,
      startTime: '2024-01-15T14:00:00Z',
      endTime: '2024-01-15T14:05:00Z',
      results: {
        averageResponseTime: 245,
        maxResponseTime: 1200,
        minResponseTime: 89,
        requestsPerSecond: 85.6,
        totalRequests: 25680,
        successRate: 98.5,
        errorRate: 1.5,
        throughput: 42.3,
        bottlenecks: ['Database queries', 'Image loading']
      }
    },
    {
      id: 'test-2',
      name: 'API Stress Test',
      type: 'stress',
      status: 'completed',
      targetUrl: '/api/users',
      virtualUsers: 500,
      duration: 600,
      rampUpTime: 120,
      startTime: '2024-01-15T13:00:00Z',
      endTime: '2024-01-15T13:10:00Z',
      results: {
        averageResponseTime: 567,
        maxResponseTime: 3200,
        minResponseTime: 156,
        requestsPerSecond: 156.8,
        totalRequests: 94080,
        successRate: 94.2,
        errorRate: 5.8,
        throughput: 89.7,
        bottlenecks: ['Database connection pool', 'Memory allocation']
      }
    },
    {
      id: 'test-3',
      name: 'Payment Flow Test',
      type: 'spike',
      status: 'failed',
      targetUrl: '/payment/checkout',
      virtualUsers: 1000,
      duration: 180,
      rampUpTime: 30,
      startTime: '2024-01-15T12:00:00Z',
      endTime: '2024-01-15T12:03:00Z',
      results: {
        averageResponseTime: 2340,
        maxResponseTime: 8900,
        minResponseTime: 234,
        requestsPerSecond: 23.4,
        totalRequests: 4212,
        successRate: 76.3,
        errorRate: 23.7,
        throughput: 15.6,
        bottlenecks: ['Payment gateway timeout', 'Database deadlocks', 'Memory overflow']
      }
    }
  ]);

  // Generate sample real-time metrics
  const generateMetrics = useMemo(() => {
    const data: MetricData[] = [];
    const now = new Date();
    
    for (let i = 30; i >= 0; i--) {
      const timestamp = new Date(now.getTime() - i * 10000);
      data.push({
        timestamp: timestamp.toLocaleTimeString(),
        responseTime: Math.floor(Math.random() * 1000) + 200,
        throughput: Math.floor(Math.random() * 100) + 50,
        errorRate: Math.random() * 10,
        activeUsers: Math.floor(Math.random() * 200) + 50,
        cpuUsage: Math.random() * 80 + 20,
        memoryUsage: Math.random() * 70 + 30
      });
    }
    
    return data;
  }, []);

  const startLoadTest = async (testId: string) => {
    setIsRunning(true);
    setSelectedTest(testId);
    
    setLoadTests(prev => prev.map(test => 
      test.id === testId ? { ...test, status: 'running', startTime: new Date().toISOString() } : test
    ));

    // Simulate test execution
    setTimeout(() => {
      const success = Math.random() > 0.3;
      setLoadTests(prev => prev.map(test => 
        test.id === testId ? { 
          ...test, 
          status: success ? 'completed' : 'failed',
          endTime: new Date().toISOString()
        } : test
      ));
      setIsRunning(false);
    }, 10000); // 10 second simulation
  };

  const stopLoadTest = () => {
    setIsRunning(false);
    if (selectedTest) {
      setLoadTests(prev => prev.map(test => 
        test.id === selectedTest ? { 
          ...test, 
          status: 'cancelled',
          endTime: new Date().toISOString()
        } : test
      ));
    }
  };

  const getStatusIcon = (status: LoadTest['status']) => {
    switch (status) {
      case 'running': return <Activity className="h-4 w-4 text-blue-500 animate-pulse" />;
      case 'completed': return <CheckCircle className="h-4 w-4 text-green-500" />;
      case 'failed': return <XCircle className="h-4 w-4 text-red-500" />;
      case 'cancelled': return <XCircle className="h-4 w-4 text-gray-500" />;
      default: return <Clock className="h-4 w-4 text-gray-400" />;
    }
  };

  const getStatusColor = (status: LoadTest['status']) => {
    switch (status) {
      case 'completed': return 'default';
      case 'failed': return 'destructive';
      case 'running': return 'secondary';
      case 'cancelled': return 'outline';
      default: return 'outline';
    }
  };

  const getTypeIcon = (type: LoadTest['type']) => {
    switch (type) {
      case 'spike': return <Zap className="h-4 w-4" />;
      case 'stress': return <AlertTriangle className="h-4 w-4" />;
      case 'load': return <Users className="h-4 w-4" />;
      case 'volume': return <Target className="h-4 w-4" />;
      case 'endurance': return <Clock className="h-4 w-4" />;
    }
  };

  const getPerformanceColor = (value: number, threshold: { good: number, warning: number }) => {
    if (value <= threshold.good) return 'text-green-600';
    if (value <= threshold.warning) return 'text-yellow-600';
    return 'text-red-600';
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold">Load Testing Suite</h2>
          <p className="text-muted-foreground">Performance and stress testing for your applications</p>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" size="sm" onClick={() => console.log("LoadTestingSuite button clicked")}>
            <Settings className="h-4 w-4 mr-2" />
            Configure
          </Button>
          {!isRunning ? (
            <Button 
              onClick={() => startLoadTest('test-1')} 
              className="gap-2"
            >
              <Play className="h-4 w-4" />
              Run Test
            </Button>
          ) : (
            <Button 
              onClick={stopLoadTest} 
              variant="destructive" 
              className="gap-2"
            >
              <Square className="h-4 w-4" />
              Stop Test
            </Button>
          )}
        </div>
      </div>

      {/* Quick Stats */}
      <div className="grid gap-4 md:grid-cols-5">
        <Card>
          <CardContent className="pt-4">
            <div className="text-2xl font-bold">{loadTests.length}</div>
            <p className="text-xs text-muted-foreground">Total Tests</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-4">
            <div className="text-2xl font-bold text-green-600">
              {loadTests.filter(t => t.status === 'completed').length}
            </div>
            <p className="text-xs text-muted-foreground">Completed</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-4">
            <div className="text-2xl font-bold text-red-600">
              {loadTests.filter(t => t.status === 'failed').length}
            </div>
            <p className="text-xs text-muted-foreground">Failed</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-4">
            <div className="text-2xl font-bold">
              {loadTests.find(t => t.results)?.results?.averageResponseTime || 0}ms
            </div>
            <p className="text-xs text-muted-foreground">Avg Response</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-4">
            <div className="text-2xl font-bold">
              {loadTests.find(t => t.results)?.results?.requestsPerSecond.toFixed(1) || 0}/s
            </div>
            <p className="text-xs text-muted-foreground">Req/Sec</p>
          </CardContent>
        </Card>
      </div>

      <Tabs defaultValue="tests" className="w-full">
        <TabsList>
          <TabsTrigger value="tests">Test Scenarios</TabsTrigger>
          <TabsTrigger value="metrics">Real-time Metrics</TabsTrigger>
          <TabsTrigger value="results">Test Results</TabsTrigger>
          <TabsTrigger value="analysis">Performance Analysis</TabsTrigger>
        </TabsList>

        <TabsContent value="tests">
          <div className="grid gap-4">
            {loadTests.map((test) => (
              <Card key={test.id}>
                <CardHeader>
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      {getStatusIcon(test.status)}
                      <CardTitle className="text-lg">{test.name}</CardTitle>
                      <Badge variant={getStatusColor(test.status)}>
                        {test.status}
                      </Badge>
                    </div>
                    <div className="flex items-center gap-2">
                      {getTypeIcon(test.type)}
                      <Badge variant="outline" className="capitalize">
                        {test.type}
                      </Badge>
                      <Button
                        size="sm"
                        onClick={() => startLoadTest(test.id)}
                        disabled={isRunning}
                      >
                        Run Test
                      </Button>
                    </div>
                  </div>
                </CardHeader>
                <CardContent>
                  <div className="grid gap-4 md:grid-cols-4">
                    <div>
                      <p className="text-sm font-medium">Target URL</p>
                      <p className="text-sm text-muted-foreground">{test.targetUrl}</p>
                    </div>
                    <div>
                      <p className="text-sm font-medium">Virtual Users</p>
                      <p className="text-sm">{test.virtualUsers}</p>
                    </div>
                    <div>
                      <p className="text-sm font-medium">Duration</p>
                      <p className="text-sm">{test.duration}s</p>
                    </div>
                    <div>
                      <p className="text-sm font-medium">Ramp Up</p>
                      <p className="text-sm">{test.rampUpTime}s</p>
                    </div>
                  </div>

                  {test.results && (
                    <div className="mt-4 grid gap-2 md:grid-cols-3">
                      <div>
                        <p className="text-sm font-medium">Success Rate</p>
                        <p className={`text-lg font-bold ${getPerformanceColor(100 - test.results.successRate, { good: 5, warning: 10 })}`}>
                          {test.results.successRate}%
                        </p>
                      </div>
                      <div>
                        <p className="text-sm font-medium">Avg Response Time</p>
                        <p className={`text-lg font-bold ${getPerformanceColor(test.results.averageResponseTime, { good: 500, warning: 1000 })}`}>
                          {test.results.averageResponseTime}ms
                        </p>
                      </div>
                      <div>
                        <p className="text-sm font-medium">Throughput</p>
                        <p className="text-lg font-bold">{test.results.throughput} MB/s</p>
                      </div>
                    </div>
                  )}

                  {test.status === 'running' && (
                    <div className="mt-4">
                      <div className="flex justify-between text-sm mb-2">
                        <span>Test Progress</span>
                        <span>Running...</span>
                      </div>
                      <Progress value={45} />
                    </div>
                  )}
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>

        <TabsContent value="metrics">
          <div className="grid gap-6 md:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle>Response Time</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={generateMetrics}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="timestamp" />
                      <YAxis />
                      <Tooltip />
                      <Line 
                        type="monotone" 
                        dataKey="responseTime" 
                        stroke="hsl(var(--primary))" 
                        strokeWidth={2}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Throughput & Error Rate</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={generateMetrics}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="timestamp" />
                      <YAxis />
                      <Tooltip />
                      <Line 
                        type="monotone" 
                        dataKey="throughput" 
                        stroke="hsl(var(--primary))" 
                        strokeWidth={2}
                        name="Throughput"
                      />
                      <Line 
                        type="monotone" 
                        dataKey="errorRate" 
                        stroke="hsl(var(--destructive))" 
                        strokeWidth={2}
                        name="Error Rate"
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>System Resources</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={generateMetrics}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="timestamp" />
                      <YAxis />
                      <Tooltip />
                      <Area 
                        type="monotone" 
                        dataKey="cpuUsage" 
                        stackId="1"
                        stroke="hsl(var(--primary))" 
                        fill="hsl(var(--primary))"
                        name="CPU Usage %"
                      />
                      <Area 
                        type="monotone" 
                        dataKey="memoryUsage" 
                        stackId="2"
                        stroke="hsl(var(--secondary))" 
                        fill="hsl(var(--secondary))"
                        name="Memory Usage %"
                      />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Active Users</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={generateMetrics}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="timestamp" />
                      <YAxis />
                      <Tooltip />
                      <Area 
                        type="monotone" 
                        dataKey="activeUsers" 
                        stroke="hsl(var(--primary))" 
                        fill="hsl(var(--primary))"
                        name="Active Users"
                      />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="results">
          <div className="space-y-6">
            {loadTests.filter(test => test.results).map((test) => (
              <Card key={test.id}>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    {getStatusIcon(test.status)}
                    {test.name} - Results
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="grid gap-6 md:grid-cols-3">
                    <div className="space-y-2">
                      <h4 className="font-medium">Response Times</h4>
                      <div className="space-y-1">
                        <div className="flex justify-between">
                          <span className="text-sm">Average:</span>
                          <span className="font-medium">{test.results?.averageResponseTime}ms</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-sm">Maximum:</span>
                          <span className="font-medium">{test.results?.maxResponseTime}ms</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-sm">Minimum:</span>
                          <span className="font-medium">{test.results?.minResponseTime}ms</span>
                        </div>
                      </div>
                    </div>

                    <div className="space-y-2">
                      <h4 className="font-medium">Request Metrics</h4>
                      <div className="space-y-1">
                        <div className="flex justify-between">
                          <span className="text-sm">Total Requests:</span>
                          <span className="font-medium">{test.results?.totalRequests.toLocaleString()}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-sm">Requests/sec:</span>
                          <span className="font-medium">{test.results?.requestsPerSecond}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-sm">Success Rate:</span>
                          <span className={`font-medium ${getPerformanceColor(100 - (test.results?.successRate || 0), { good: 5, warning: 10 })}`}>
                            {test.results?.successRate}%
                          </span>
                        </div>
                      </div>
                    </div>

                    <div className="space-y-2">
                      <h4 className="font-medium">Performance</h4>
                      <div className="space-y-1">
                        <div className="flex justify-between">
                          <span className="text-sm">Throughput:</span>
                          <span className="font-medium">{test.results?.throughput} MB/s</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-sm">Error Rate:</span>
                          <span className={`font-medium ${getPerformanceColor(test.results?.errorRate || 0, { good: 1, warning: 5 })}`}>
                            {test.results?.errorRate}%
                          </span>
                        </div>
                      </div>
                    </div>
                  </div>

                  {test.results?.bottlenecks && test.results.bottlenecks.length > 0 && (
                    <div className="mt-6">
                      <h4 className="font-medium mb-2 flex items-center gap-2">
                        <AlertTriangle className="h-4 w-4 text-yellow-500" />
                        Identified Bottlenecks
                      </h4>
                      <div className="space-y-2">
                        {test.results.bottlenecks.map((bottleneck, index) => (
                          <div key={index} className="flex items-center gap-2 p-2 bg-yellow-50 dark:bg-yellow-900/20 rounded-md">
                            <AlertTriangle className="h-4 w-4 text-yellow-500" />
                            <span className="text-sm">{bottleneck}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>

        <TabsContent value="analysis">
          <div className="grid gap-6">
            <Card>
              <CardHeader>
                <CardTitle>Performance Recommendations</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="p-4 border-l-4 border-red-500 bg-red-50 dark:bg-red-900/20">
                    <h4 className="font-medium text-red-700 dark:text-red-300">Critical Issues</h4>
                    <ul className="mt-2 text-sm text-red-600 dark:text-red-400 space-y-1">
                      <li>• Payment gateway timeouts under high load</li>
                      <li>• Database connection pool exhaustion</li>
                      <li>• Memory leaks during stress testing</li>
                    </ul>
                  </div>
                  
                  <div className="p-4 border-l-4 border-yellow-500 bg-yellow-50 dark:bg-yellow-900/20">
                    <h4 className="font-medium text-yellow-700 dark:text-yellow-300">Performance Optimizations</h4>
                    <ul className="mt-2 text-sm text-yellow-600 dark:text-yellow-400 space-y-1">
                      <li>• Implement database query optimization</li>
                      <li>• Add caching layer for frequently accessed data</li>
                      <li>• Optimize image loading and compression</li>
                    </ul>
                  </div>
                  
                  <div className="p-4 border-l-4 border-green-500 bg-green-50 dark:bg-green-900/20">
                    <h4 className="font-medium text-green-700 dark:text-green-300">Good Performance</h4>
                    <ul className="mt-2 text-sm text-green-600 dark:text-green-400 space-y-1">
                      <li>• API authentication response times are optimal</li>
                      <li>• Static asset delivery is well optimized</li>
                      <li>• Database read operations perform well under load</li>
                    </ul>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Load Testing Best Practices</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="prose prose-sm max-w-none">
                  <h4>Test Types:</h4>
                  <ul>
                    <li><strong>Load Testing:</strong> Normal expected load</li>
                    <li><strong>Stress Testing:</strong> Beyond normal capacity</li>
                    <li><strong>Spike Testing:</strong> Sudden traffic increases</li>
                    <li><strong>Volume Testing:</strong> Large amounts of data</li>
                    <li><strong>Endurance Testing:</strong> Extended periods</li>
                  </ul>
                  
                  <h4>Key Metrics to Monitor:</h4>
                  <ul>
                    <li>Response time (average, 95th percentile)</li>
                    <li>Throughput (requests per second)</li>
                    <li>Error rate and success rate</li>
                    <li>Resource utilization (CPU, memory, network)</li>
                  </ul>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
};