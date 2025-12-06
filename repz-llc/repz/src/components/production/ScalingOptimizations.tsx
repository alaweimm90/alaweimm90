import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/ui/molecules/Card';
import { Badge } from '@/ui/atoms/Badge';
import { Button } from '@/ui/atoms/Button';
import { Progress } from '@/components/ui/progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/ui/molecules/Tabs';
import { 
  Server, 
  Database, 
  Cloud, 
  Zap, 
  TrendingUp, 
  Users, 
  Activity,
  AlertTriangle,
  CheckCircle,
  Settings,
  BarChart3,
  Globe
} from 'lucide-react';

interface PerformanceMetric {
  name: string;
  value: number;
  threshold: number;
  unit: string;
  status: 'good' | 'warning' | 'critical';
  trend: 'up' | 'down' | 'stable';
}

interface ScalingRecommendation {
  id: string;
  type: 'performance' | 'infrastructure' | 'database' | 'cdn';
  title: string;
  description: string;
  impact: 'high' | 'medium' | 'low';
  effort: 'high' | 'medium' | 'low';
  priority: number;
  implemented: boolean;
}

const mockMetrics: PerformanceMetric[] = [
  {
    name: 'Response Time',
    value: 245,
    threshold: 300,
    unit: 'ms',
    status: 'good',
    trend: 'stable'
  },
  {
    name: 'Throughput',
    value: 1250,
    threshold: 1000,
    unit: 'req/min',
    status: 'good',
    trend: 'up'
  },
  {
    name: 'Error Rate',
    value: 0.5,
    threshold: 1.0,
    unit: '%',
    status: 'good',
    trend: 'down'
  },
  {
    name: 'CPU Usage',
    value: 78,
    threshold: 80,
    unit: '%',
    status: 'warning',
    trend: 'up'
  },
  {
    name: 'Memory Usage',
    value: 85,
    threshold: 85,
    unit: '%',
    status: 'critical',
    trend: 'up'
  },
  {
    name: 'Database Connections',
    value: 45,
    threshold: 50,
    unit: 'connections',
    status: 'warning',
    trend: 'up'
  }
];

const mockRecommendations: ScalingRecommendation[] = [
  {
    id: '1',
    type: 'infrastructure',
    title: 'Implement Auto-scaling',
    description: 'Configure automatic horizontal scaling based on CPU and memory thresholds',
    impact: 'high',
    effort: 'medium',
    priority: 1,
    implemented: false
  },
  {
    id: '2',
    type: 'database',
    title: 'Database Connection Pooling',
    description: 'Optimize database connections with connection pooling and read replicas',
    impact: 'high',
    effort: 'medium',
    priority: 2,
    implemented: false
  },
  {
    id: '3',
    type: 'cdn',
    title: 'Global CDN Implementation',
    description: 'Deploy global CDN for static assets and edge caching',
    impact: 'medium',
    effort: 'low',
    priority: 3,
    implemented: true
  },
  {
    id: '4',
    type: 'performance',
    title: 'Code Splitting & Lazy Loading',
    description: 'Implement advanced code splitting and lazy loading for components',
    impact: 'medium',
    effort: 'high',
    priority: 4,
    implemented: false
  },
  {
    id: '5',
    type: 'database',
    title: 'Query Optimization',
    description: 'Optimize database queries and implement proper indexing strategies',
    impact: 'high',
    effort: 'low',
    priority: 5,
    implemented: true
  }
];

export const ScalingOptimizations: React.FC = () => {
  const [metrics, setMetrics] = useState<PerformanceMetric[]>(mockMetrics);
  const [recommendations, setRecommendations] = useState<ScalingRecommendation[]>(mockRecommendations);
  const [activeTab, setActiveTab] = useState('overview');

  // Simulate real-time metric updates
  useEffect(() => {
    const interval = setInterval(() => {
      setMetrics(prevMetrics => 
        prevMetrics.map(metric => ({
          ...metric,
          value: metric.value + (Math.random() - 0.5) * 10,
          trend: Math.random() > 0.5 ? 'up' : 'down'
        }))
      );
    }, 5000);

    return () => clearInterval(interval);
  }, []);

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'good':
        return 'bg-green-100 text-green-800';
      case 'warning':
        return 'bg-yellow-100 text-yellow-800';
      case 'critical':
        return 'bg-red-100 text-red-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };

  const getImpactColor = (impact: string) => {
    switch (impact) {
      case 'high':
        return 'bg-red-100 text-red-800';
      case 'medium':
        return 'bg-yellow-100 text-yellow-800';
      case 'low':
        return 'bg-green-100 text-green-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };

  const getTypeIcon = (type: string) => {
    switch (type) {
      case 'infrastructure':
        return <Server className="w-4 h-4" />;
      case 'database':
        return <Database className="w-4 h-4" />;
      case 'cdn':
        return <Globe className="w-4 h-4" />;
      case 'performance':
        return <Zap className="w-4 h-4" />;
      default:
        return <Settings className="w-4 h-4" />;
    }
  };

  const toggleRecommendation = (id: string) => {
    setRecommendations(prev => 
      prev.map(rec => 
        rec.id === id ? { ...rec, implemented: !rec.implemented } : rec
      )
    );
  };

  const implementedCount = recommendations.filter(rec => rec.implemented).length;
  const implementationProgress = (implementedCount / recommendations.length) * 100;

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold mb-2">Scaling Optimizations</h2>
        <p className="text-muted-foreground">
          Monitor performance metrics and implement scaling recommendations
        </p>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="metrics">Performance Metrics</TabsTrigger>
          <TabsTrigger value="recommendations">Recommendations</TabsTrigger>
          <TabsTrigger value="infrastructure">Infrastructure</TabsTrigger>
        </TabsList>

        <TabsContent value="overview" className="space-y-6">
          {/* Overall Health */}
          <div className="grid md:grid-cols-4 gap-4">
            <Card>
              <CardContent className="p-4">
                <div className="flex items-center gap-2 mb-2">
                  <Activity className="w-5 h-5 text-green-600" />
                  <span className="text-sm font-medium">System Health</span>
                </div>
                <div className="text-2xl font-bold text-green-600">98.5%</div>
                <div className="text-xs text-muted-foreground">Uptime</div>
              </CardContent>
            </Card>
            
            <Card>
              <CardContent className="p-4">
                <div className="flex items-center gap-2 mb-2">
                  <Users className="w-5 h-5 text-blue-600" />
                  <span className="text-sm font-medium">Active Users</span>
                </div>
                <div className="text-2xl font-bold text-blue-600">2,847</div>
                <div className="text-xs text-muted-foreground">+12% from last week</div>
              </CardContent>
            </Card>
            
            <Card>
              <CardContent className="p-4">
                <div className="flex items-center gap-2 mb-2">
                  <TrendingUp className="w-5 h-5 text-repz-orange" />
                  <span className="text-sm font-medium">Load</span>
                </div>
                <div className="text-2xl font-bold text-repz-orange">76%</div>
                <div className="text-xs text-muted-foreground">Average CPU</div>
              </CardContent>
            </Card>
            
            <Card>
              <CardContent className="p-4">
                <div className="flex items-center gap-2 mb-2">
                  <CheckCircle className="w-5 h-5 text-purple-600" />
                  <span className="text-sm font-medium">Optimizations</span>
                </div>
                <div className="text-2xl font-bold text-purple-600">
                  {implementedCount}/{recommendations.length}
                </div>
                <div className="text-xs text-muted-foreground">Implemented</div>
              </CardContent>
            </Card>
          </div>

          {/* Implementation Progress */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <BarChart3 className="w-5 h-5" />
                Optimization Progress
              </CardTitle>
              <CardDescription>
                Track implementation of scaling recommendations
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <span>Overall Progress</span>
                  <span className="font-medium">{Math.round(implementationProgress)}%</span>
                </div>
                <Progress value={implementationProgress} className="h-2" />
                <div className="text-sm text-muted-foreground">
                  {implementedCount} of {recommendations.length} recommendations implemented
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Critical Alerts */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <AlertTriangle className="w-5 h-5 text-red-600" />
                Critical Alerts
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {metrics.filter(m => m.status === 'critical').map((metric) => (
                  <div key={metric.name} className="flex items-center justify-between p-3 border border-red-200 rounded-lg bg-red-50">
                    <div>
                      <div className="font-medium text-red-800">{metric.name}</div>
                      <div className="text-sm text-red-600">
                        {metric.value.toFixed(1)}{metric.unit} (threshold: {metric.threshold}{metric.unit})
                      </div>
                    </div>
                    <Badge className="bg-red-100 text-red-800">Critical</Badge>
                  </div>
                ))}
                {metrics.filter(m => m.status === 'critical').length === 0 && (
                  <div className="text-center py-8 text-muted-foreground">
                    No critical alerts at this time
                  </div>
                )}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="metrics" className="space-y-6">
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
            {metrics.map((metric) => (
              <Card key={metric.name}>
                <CardHeader className="pb-3">
                  <div className="flex items-center justify-between">
                    <CardTitle className="text-base">{metric.name}</CardTitle>
                    <Badge className={getStatusColor(metric.status)}>
                      {metric.status}
                    </Badge>
                  </div>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3">
                    <div className="text-2xl font-bold">
                      {metric.value.toFixed(1)}{metric.unit}
                    </div>
                    <div className="text-sm text-muted-foreground">
                      Threshold: {metric.threshold}{metric.unit}
                    </div>
                    <Progress 
                      value={(metric.value / metric.threshold) * 100} 
                      className="h-2"
                    />
                    <div className="flex items-center justify-between text-xs">
                      <span className="text-muted-foreground">
                        Trend: {metric.trend}
                      </span>
                      <span className={
                        metric.value > metric.threshold 
                          ? 'text-red-600' 
                          : 'text-green-600'
                      }>
                        {((metric.value / metric.threshold) * 100).toFixed(1)}%
                      </span>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>

        <TabsContent value="recommendations" className="space-y-6">
          <div className="space-y-4">
            {recommendations
              .sort((a, b) => a.priority - b.priority)
              .map((recommendation) => (
                <Card key={recommendation.id}>
                  <CardHeader>
                    <div className="flex items-start justify-between">
                      <div className="flex items-center gap-3">
                        {getTypeIcon(recommendation.type)}
                        <div>
                          <CardTitle className="text-lg">{recommendation.title}</CardTitle>
                          <CardDescription>{recommendation.description}</CardDescription>
                        </div>
                      </div>
                      <div className="flex items-center gap-2">
                        {recommendation.implemented ? (
                          <CheckCircle className="w-5 h-5 text-green-600" />
                        ) : (
                          <Button
                            size="sm"
                            onClick={() => toggleRecommendation(recommendation.id)}
                          >
                            Implement
                          </Button>
                        )}
                      </div>
                    </div>
                  </CardHeader>
                  <CardContent>
                    <div className="flex items-center gap-4">
                      <div className="flex items-center gap-2">
                        <span className="text-sm text-muted-foreground">Impact:</span>
                        <Badge className={getImpactColor(recommendation.impact)}>
                          {recommendation.impact}
                        </Badge>
                      </div>
                      <div className="flex items-center gap-2">
                        <span className="text-sm text-muted-foreground">Effort:</span>
                        <Badge variant="outline">
                          {recommendation.effort}
                        </Badge>
                      </div>
                      <div className="flex items-center gap-2">
                        <span className="text-sm text-muted-foreground">Priority:</span>
                        <Badge variant="secondary">
                          #{recommendation.priority}
                        </Badge>
                      </div>
                      <div className="flex items-center gap-2">
                        <span className="text-sm text-muted-foreground">Type:</span>
                        <Badge variant="outline">
                          {recommendation.type}
                        </Badge>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              ))}
          </div>
        </TabsContent>

        <TabsContent value="infrastructure" className="space-y-6">
          {/* Infrastructure Overview */}
          <div className="grid md:grid-cols-3 gap-4">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Server className="w-5 h-5" />
                  Application Servers
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span>Active Instances</span>
                    <span className="font-medium">3</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Load Balancer</span>
                    <Badge className="bg-green-100 text-green-800">Healthy</Badge>
                  </div>
                  <div className="flex justify-between">
                    <span>Auto-scaling</span>
                    <Badge className="bg-yellow-100 text-yellow-800">Configured</Badge>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Database className="w-5 h-5" />
                  Database
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span>Primary DB</span>
                    <Badge className="bg-green-100 text-green-800">Online</Badge>
                  </div>
                  <div className="flex justify-between">
                    <span>Read Replicas</span>
                    <span className="font-medium">2</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Backup Status</span>
                    <Badge className="bg-green-100 text-green-800">Current</Badge>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Cloud className="w-5 h-5" />
                  Edge & CDN
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span>Edge Locations</span>
                    <span className="font-medium">12</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Cache Hit Rate</span>
                    <span className="font-medium">94%</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Global Latency</span>
                    <span className="font-medium">45ms</span>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
};