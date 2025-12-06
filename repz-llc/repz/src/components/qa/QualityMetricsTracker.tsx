import React, { useState, useMemo } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/ui/molecules/Card';
import { Button } from '@/ui/atoms/Button';
import { Progress } from '@/components/ui/progress';
import { Badge } from '@/ui/atoms/Badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/ui/molecules/Tabs';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar } from 'recharts';
import { 
  TrendingUp, 
  TrendingDown, 
  Target,
  Clock,
  CheckCircle,
  XCircle,
  AlertTriangle,
  BarChart3,
  Activity
} from 'lucide-react';

interface QualityMetric {
  id: string;
  name: string;
  current: number;
  target: number;
  trend: 'up' | 'down' | 'stable';
  category: 'testing' | 'performance' | 'security' | 'deployment';
  unit: string;
  description: string;
}

interface MetricHistory {
  timestamp: string;
  value: number;
  date: string;
}

export const QualityMetricsTracker = () => {
  const [selectedMetric, setSelectedMetric] = useState<string>('test-coverage');
  const [timeRange, setTimeRange] = useState<'7d' | '30d' | '90d'>('30d');

  const qualityMetrics: QualityMetric[] = useMemo(() => [
    {
      id: 'test-coverage',
      name: 'Test Coverage',
      current: 87,
      target: 85,
      trend: 'up',
      category: 'testing',
      unit: '%',
      description: 'Code coverage across all test suites'
    },
    {
      id: 'test-success',
      name: 'Test Success Rate',
      current: 94,
      target: 95,
      trend: 'down',
      category: 'testing',
      unit: '%',
      description: 'Percentage of passing tests'
    },
    {
      id: 'performance-score',
      name: 'Performance Score',
      current: 91,
      target: 90,
      trend: 'up',
      category: 'performance',
      unit: 'pts',
      description: 'Overall performance metrics score'
    },
    {
      id: 'security-vulnerabilities',
      name: 'Security Issues',
      current: 2,
      target: 0,
      trend: 'stable',
      category: 'security',
      unit: 'issues',
      description: 'Number of security vulnerabilities'
    },
    {
      id: 'deployment-success',
      name: 'Deployment Success',
      current: 98,
      target: 95,
      trend: 'up',
      category: 'deployment',
      unit: '%',
      description: 'Successful deployment rate'
    },
    {
      id: 'build-time',
      name: 'Build Time',
      current: 8.5,
      target: 10,
      trend: 'down',
      category: 'performance',
      unit: 'min',
      description: 'Average build duration'
    }
  ], []);

  const metricHistory: Record<string, MetricHistory[]> = useMemo(() => {
    const generateHistory = (baseValue: number, volatility: number = 5) => {
      const data: MetricHistory[] = [];
      const days = timeRange === '7d' ? 7 : timeRange === '30d' ? 30 : 90;
      
      for (let i = days; i >= 0; i--) {
        const date = new Date();
        date.setDate(date.getDate() - i);
        const variation = (Math.random() - 0.5) * volatility;
        const value = Math.max(0, baseValue + variation);
        
        data.push({
          timestamp: date.toISOString(),
          value: Math.round(value * 100) / 100,
          date: date.toLocaleDateString()
        });
      }
      return data;
    };

    return {
      'test-coverage': generateHistory(87, 3),
      'test-success': generateHistory(94, 4),
      'performance-score': generateHistory(91, 5),
      'security-vulnerabilities': generateHistory(2, 1),
      'deployment-success': generateHistory(98, 2),
      'build-time': generateHistory(8.5, 1.5)
    };
  }, [timeRange]);

  const getTrendIcon = (trend: string) => {
    switch (trend) {
      case 'up': return <TrendingUp className="h-4 w-4 text-green-500" />;
      case 'down': return <TrendingDown className="h-4 w-4 text-red-500" />;
      default: return <Activity className="h-4 w-4 text-gray-500" />;
    }
  };

  const getStatusColor = (current: number, target: number, unit: string) => {
    if (unit === 'issues') {
      return current <= target ? 'green' : current <= target + 2 ? 'yellow' : 'red';
    }
    if (unit === 'min') {
      return current <= target ? 'green' : current <= target + 2 ? 'yellow' : 'red';
    }
    return current >= target ? 'green' : current >= target - 5 ? 'yellow' : 'red';
  };

  const getCategoryIcon = (category: string) => {
    switch (category) {
      case 'testing': return <CheckCircle className="h-4 w-4" />;
      case 'performance': return <BarChart3 className="h-4 w-4" />;
      case 'security': return <AlertTriangle className="h-4 w-4" />;
      case 'deployment': return <Target className="h-4 w-4" />;
      default: return <Activity className="h-4 w-4" />;
    }
  };

  const selectedMetricData = qualityMetrics.find(m => m.id === selectedMetric);
  const chartData = metricHistory[selectedMetric] || [];

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold">Quality Metrics Tracker</h2>
          <p className="text-muted-foreground">Monitor and track quality metrics over time</p>
        </div>
        <div className="flex gap-2">
          {(['7d', '30d', '90d'] as const).map((range) => (
            <Button
              key={range}
              variant={timeRange === range ? 'default' : 'outline'}
              size="sm"
              onClick={() => setTimeRange(range)}
            >
              {range}
            </Button>
          ))}
        </div>
      </div>

      {/* Metrics Overview */}
      <div className="grid gap-4 md:grid-cols-3 lg:grid-cols-6">
        {qualityMetrics.map((metric) => {
          const statusColor = getStatusColor(metric.current, metric.target, metric.unit);
          return (
            <Card 
              key={metric.id} 
              className={`cursor-pointer transition-colors ${
                selectedMetric === metric.id ? 'ring-2 ring-primary' : ''
              }`}
              onClick={() => setSelectedMetric(metric.id)}
            >
              <CardContent className="pt-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    {getCategoryIcon(metric.category)}
                    <Badge variant="outline" className="text-xs">
                      {metric.category}
                    </Badge>
                  </div>
                  {getTrendIcon(metric.trend)}
                </div>
                
                <div className="mt-2">
                  <div className="text-2xl font-bold">
                    {metric.current}{metric.unit}
                  </div>
                  <p className="text-xs text-muted-foreground truncate">
                    {metric.name}
                  </p>
                </div>

                <div className="mt-2 flex items-center gap-2">
                  <Progress 
                    value={
                      metric.unit === 'issues' 
                        ? Math.max(0, 100 - (metric.current / Math.max(metric.target, 1)) * 100)
                        : metric.unit === 'min'
                        ? Math.max(0, 100 - (metric.current / metric.target) * 100)
                        : (metric.current / metric.target) * 100
                    } 
                    className="flex-1"
                  />
                  <Badge 
                    variant={statusColor === 'green' ? 'default' : statusColor === 'yellow' ? 'secondary' : 'destructive'}
                    className="text-xs"
                  >
                    {statusColor === 'green' ? 'Good' : statusColor === 'yellow' ? 'Warning' : 'Critical'}
                  </Badge>
                </div>

                <div className="mt-2 text-xs text-muted-foreground">
                  Target: {metric.target}{metric.unit}
                </div>
              </CardContent>
            </Card>
          );
        })}
      </div>

      {/* Detailed Chart View */}
      <Tabs defaultValue="trend" className="w-full">
        <TabsList>
          <TabsTrigger value="trend">Trend Analysis</TabsTrigger>
          <TabsTrigger value="comparison">Target Comparison</TabsTrigger>
        </TabsList>

        <TabsContent value="trend">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Clock className="h-5 w-5" />
                {selectedMetricData?.name} - Trend Over Time
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="h-80">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={chartData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="date" />
                    <YAxis />
                    <Tooltip />
                    <Line 
                      type="monotone" 
                      dataKey="value" 
                      stroke="hsl(var(--primary))" 
                      strokeWidth={2}
                      dot={{ fill: "hsl(var(--primary))" }}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="comparison">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Target className="h-5 w-5" />
                Current vs Target Comparison
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="h-80">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={qualityMetrics}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="name" />
                    <YAxis />
                    <Tooltip />
                    <Bar dataKey="current" fill="hsl(var(--primary))" name="Current" />
                    <Bar dataKey="target" fill="hsl(var(--muted))" name="Target" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      {/* Metric Details */}
      {selectedMetricData && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              {getCategoryIcon(selectedMetricData.category)}
              {selectedMetricData.name} Details
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid gap-4 md:grid-cols-4">
              <div className="space-y-2">
                <p className="text-sm font-medium">Current Value</p>
                <p className="text-2xl font-bold">
                  {selectedMetricData.current}{selectedMetricData.unit}
                </p>
              </div>
              <div className="space-y-2">
                <p className="text-sm font-medium">Target</p>
                <p className="text-2xl font-bold text-muted-foreground">
                  {selectedMetricData.target}{selectedMetricData.unit}
                </p>
              </div>
              <div className="space-y-2">
                <p className="text-sm font-medium">Trend</p>
                <div className="flex items-center gap-2">
                  {getTrendIcon(selectedMetricData.trend)}
                  <span className="capitalize">{selectedMetricData.trend}</span>
                </div>
              </div>
              <div className="space-y-2">
                <p className="text-sm font-medium">Status</p>
                <Badge 
                  variant={
                    getStatusColor(selectedMetricData.current, selectedMetricData.target, selectedMetricData.unit) === 'green' 
                      ? 'default' 
                      : getStatusColor(selectedMetricData.current, selectedMetricData.target, selectedMetricData.unit) === 'yellow' 
                      ? 'secondary' 
                      : 'destructive'
                  }
                >
                  {getStatusColor(selectedMetricData.current, selectedMetricData.target, selectedMetricData.unit) === 'green' ? 'On Track' : 
                   getStatusColor(selectedMetricData.current, selectedMetricData.target, selectedMetricData.unit) === 'yellow' ? 'Needs Attention' : 'Critical'}
                </Badge>
              </div>
            </div>
            <div className="mt-4">
              <p className="text-sm text-muted-foreground">
                {selectedMetricData.description}
              </p>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
};