/**
 * Performance Monitor Dashboard
 * Real-time performance monitoring and analytics
 */

import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Progress } from '@/components/ui/progress';
import { 
  Activity, 
  Clock, 
  Cpu, 
  MemoryStick,
  TrendingUp,
  RefreshCw,
  Download,
  Zap
} from 'lucide-react';
import { performanceMonitor, type PerformanceMetrics, type UserInteraction } from '@/lib/performance-monitoring';
import Plot from 'react-plotly.js';

interface PerformanceMonitorDashboardProps {
  className?: string;
}

export function PerformanceMonitorDashboard({ className }: PerformanceMonitorDashboardProps) {
  const [metrics, setMetrics] = useState<PerformanceMetrics[]>([]);
  const [interactions, setInteractions] = useState<UserInteraction[]>([]);
  const [stats, setStats] = useState<any>(null);
  const [refreshing, setRefreshing] = useState(false);

  useEffect(() => {
    loadData();
    const interval = setInterval(loadData, 2000); // Update every 2 seconds
    return () => clearInterval(interval);
  }, []);

  const loadData = () => {
    const recentMetrics = performanceMonitor.getMetrics({ limit: 100 });
    const recentInteractions = performanceMonitor.getInteractions({ limit: 50 });
    const performanceStats = performanceMonitor.getPerformanceStats();
    
    setMetrics(recentMetrics);
    setInteractions(recentInteractions);
    setStats(performanceStats);
  };

  const handleRefresh = async () => {
    setRefreshing(true);
    await loadData();
    setTimeout(() => setRefreshing(false), 500);
  };

  const handleExport = () => {
    const data = { metrics, interactions, stats };
    const dataStr = JSON.stringify(data, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `performance-data-${new Date().toISOString().split('T')[0]}.json`;
    link.click();
    URL.revokeObjectURL(url);
  };

  if (!stats) {
    return (
      <Card className={className}>
        <CardHeader>
          <CardTitle>Performance Monitor</CardTitle>
          <CardDescription>Loading performance data...</CardDescription>
        </CardHeader>
        <CardContent>
          <Progress value={50} className="w-full" />
        </CardContent>
      </Card>
    );
  }

  // Prepare chart data
  const calculationTimes = metrics
    .filter(m => m.metric.includes('physics_calculation'))
    .slice(-20)
    .map((m, index) => ({
      x: index,
      y: m.value,
      text: m.context?.name || m.metric,
      type: 'scatter'
    }));

  const interactionData = interactions.slice(-10).map((interaction, index) => ({
    x: index,
    y: interaction.type,
    text: interaction.target,
    type: 'scatter'
  }));

  return (
    <div className={`space-y-6 ${className}`}>
      {/* Summary Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-spacing-md">
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium flex items-center gap-2">
              <Activity className="h-4 w-4" />
              Total Metrics
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{stats.totalMetrics}</div>
            <div className="text-xs text-muted-foreground">
              {stats.recentMetrics} recent
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium flex items-center gap-2">
              <Clock className="h-4 w-4" />
              Avg Load Time
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {stats.averageLoadTime?.toFixed(1) || 0}ms
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium flex items-center gap-2">
              <Zap className="h-4 w-4" />
              Avg Calc Time
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {stats.averageCalculationTime?.toFixed(1) || 0}ms
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium flex items-center gap-2">
              <MemoryStick className="h-4 w-4" />
              Memory Usage
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {stats.memoryUsage?.percentage?.toFixed(1) || 0}%
            </div>
            <div className="text-xs text-muted-foreground">
              {stats.memoryUsage ? `${(stats.memoryUsage.used / 1024 / 1024).toFixed(1)}MB` : 'N/A'}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Controls */}
      <div className="flex items-center gap-spacing-xs">
        <Button
          onClick={handleRefresh}
          disabled={refreshing}
          size="sm"
          variant="outline"
          className="flex items-center gap-2 min-h-touch"
        >
          <RefreshCw className={`h-4 w-4 ${refreshing ? 'animate-spin' : ''}`} />
          Refresh
        </Button>

        <Button
          onClick={handleExport}
          size="sm"
          variant="outline"
          className="flex items-center gap-2 min-h-touch"
        >
          <Download className="h-4 w-4" />
          Export
        </Button>

        <Button
          onClick={() => performanceMonitor.clearData()}
          size="sm"
          variant="outline"
          className="text-destructive min-h-touch"
        >
          Clear Data
        </Button>
      </div>

      <Tabs defaultValue="overview" className="w-full">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="metrics">Metrics</TabsTrigger>
          <TabsTrigger value="interactions">Interactions</TabsTrigger>
          <TabsTrigger value="charts">Charts</TabsTrigger>
        </TabsList>

        <TabsContent value="overview" className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <Card>
              <CardHeader>
                <CardTitle>Performance Summary</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex items-center justify-between">
                  <span>Load Performance</span>
                  <Badge variant={stats.averageLoadTime < 100 ? "default" : "destructive"}>
                    {stats.averageLoadTime < 100 ? "Good" : "Needs Improvement"}
                  </Badge>
                </div>
                
                <div className="flex items-center justify-between">
                  <span>Calculation Speed</span>
                  <Badge variant={stats.averageCalculationTime < 50 ? "default" : "secondary"}>
                    {stats.averageCalculationTime < 50 ? "Fast" : "Moderate"}
                  </Badge>
                </div>

                {stats.memoryUsage && (
                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <span>Memory Efficiency</span>
                      <Badge variant={stats.memoryUsage.percentage < 70 ? "default" : "destructive"}>
                        {stats.memoryUsage.percentage < 70 ? "Efficient" : "High Usage"}
                      </Badge>
                    </div>
                    <Progress value={stats.memoryUsage.percentage} className="w-full" />
                  </div>
                )}
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Recent Activity</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <span>Interactions (last min)</span>
                    <span className="font-bold">{stats.recentInteractions}</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span>Metrics (last min)</span>
                    <span className="font-bold">{stats.recentMetrics}</span>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="metrics" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Recent Performance Metrics</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-2 max-h-96 overflow-y-auto">
                {metrics.slice(0, 20).map((metric, index) => (
                  <div key={index} className="flex items-center justify-between p-2 border rounded">
                    <div>
                      <div className="font-medium text-sm">{metric.metric}</div>
                      <div className="text-xs text-muted-foreground">
                        {new Date(metric.timestamp).toLocaleTimeString()}
                      </div>
                    </div>
                    <div className="text-right">
                      <div className="font-bold">{metric.value.toFixed(2)} {metric.unit}</div>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="interactions" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>User Interactions</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-2 max-h-96 overflow-y-auto">
                {interactions.map((interaction, index) => (
                  <div key={index} className="flex items-center justify-between p-2 border rounded">
                    <div>
                      <div className="flex items-center gap-2">
                        <Badge variant="outline">{interaction.type}</Badge>
                        <span className="text-sm">{interaction.target}</span>
                      </div>
                      <div className="text-xs text-muted-foreground">
                        {new Date(interaction.timestamp).toLocaleTimeString()}
                      </div>
                    </div>
                    {interaction.duration && (
                      <div className="text-sm font-medium">
                        {interaction.duration.toFixed(0)}ms
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="charts" className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <Card>
              <CardHeader>
                <CardTitle>Calculation Performance</CardTitle>
              </CardHeader>
              <CardContent>
                {calculationTimes.length > 0 && (
                  <Plot
                    data={[{
                      x: calculationTimes.map(d => d.x),
                      y: calculationTimes.map(d => d.y),
                      type: 'scatter',
                      mode: 'lines+markers',
                      marker: { color: '#3b82f6' },
                      name: 'Calculation Time (ms)'
                    }]}
                    layout={{
                      width: 400,
                      height: 250,
                      title: 'Physics Calculation Times',
                      xaxis: { title: 'Sequence' },
                      yaxis: { title: 'Time (ms)' }
                    }}
                    config={{ responsive: true }}
                  />
                )}
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Interaction Types</CardTitle>
              </CardHeader>
              <CardContent>
                {interactions.length > 0 && (
                  <Plot
                    data={[{
                      values: Object.values(
                        interactions.reduce((acc, i) => {
                          acc[i.type] = (acc[i.type] || 0) + 1;
                          return acc;
                        }, {} as Record<string, number>)
                      ),
                      labels: Object.keys(
                        interactions.reduce((acc, i) => {
                          acc[i.type] = (acc[i.type] || 0) + 1;
                          return acc;
                        }, {} as Record<string, number>)
                      ),
                      type: 'pie'
                    }]}
                    layout={{
                      width: 400,
                      height: 250,
                      title: 'User Interactions'
                    }}
                    config={{ responsive: true }}
                  />
                )}
              </CardContent>
            </Card>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
}

export default PerformanceMonitorDashboard;