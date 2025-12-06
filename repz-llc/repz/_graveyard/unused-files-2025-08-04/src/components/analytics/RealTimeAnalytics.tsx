import React, { useState, useEffect, useCallback } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Activity, Users, TrendingUp, Zap } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { usePerformanceMonitor } from '@/hooks/usePerformanceMonitor';

interface AnalyticsData {
  timestamp: string;
  activeUsers: number;
  pageViews: number;
  performanceScore: number;
  errors: number;
}

export const RealTimeAnalytics = () => {
  const [data, setData] = useState<AnalyticsData[]>([]);
  const [isConnected, setIsConnected] = useState(false);
  const { trackInteraction } = usePerformanceMonitor('RealTimeAnalytics');

  // Simulate real-time data
  const generateDataPoint = useCallback((): AnalyticsData => {
    const now = new Date();
    return {
      timestamp: now.toLocaleTimeString(),
      activeUsers: Math.floor(Math.random() * 100) + 50,
      pageViews: Math.floor(Math.random() * 1000) + 200,
      performanceScore: Math.floor(Math.random() * 30) + 70,
      errors: Math.floor(Math.random() * 5)
    };
  }, []);

  const startRealTimeMonitoring = useCallback(() => {
    trackInteraction('start_monitoring', () => {
      setIsConnected(true);
      console.log('🔴 Real-time analytics started');
    });
  }, [trackInteraction]);

  const stopRealTimeMonitoring = useCallback(() => {
    trackInteraction('stop_monitoring', () => {
      setIsConnected(false);
      console.log('⏹️ Real-time analytics stopped');
    });
  }, [trackInteraction]);

  useEffect(() => {
    if (!isConnected) return;

    const interval = setInterval(() => {
      const newDataPoint = generateDataPoint();
      setData(prev => {
        const updated = [...prev, newDataPoint];
        // Keep only last 20 data points
        return updated.slice(-20);
      });
    }, 2000);

    return () => clearInterval(interval);
  }, [isConnected, generateDataPoint]);

  const latestData = data[data.length - 1];

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold">Real-Time Analytics</h2>
        <div className="flex gap-2">
          <Badge variant={isConnected ? 'default' : 'secondary'}>
            {isConnected ? 'Live' : 'Stopped'}
          </Badge>
          {isConnected ? (
            <Button onClick={stopRealTimeMonitoring} variant="outline" size="sm">
              Stop
            </Button>
          ) : (
            <Button onClick={startRealTimeMonitoring} size="sm">
              <Activity className="mr-2 h-4 w-4" />
              Start Monitoring
            </Button>
          )}
        </div>
      </div>

      {latestData && (
        <div className="grid gap-4 md:grid-cols-4">
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Active Users</CardTitle>
              <Users className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{latestData.activeUsers}</div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Page Views</CardTitle>
              <TrendingUp className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{latestData.pageViews}</div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Performance</CardTitle>
              <Zap className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{latestData.performanceScore}</div>
              <p className="text-xs text-muted-foreground">Score out of 100</p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Errors</CardTitle>
              <Activity className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className={`text-2xl font-bold ${
                latestData.errors > 3 ? 'text-red-600' : 'text-green-600'
              }`}>
                {latestData.errors}
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {data.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Performance Trends</CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={data}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="timestamp" />
                <YAxis />
                <Tooltip />
                <Line 
                  type="monotone" 
                  dataKey="activeUsers" 
                  stroke="hsl(var(--primary))" 
                  strokeWidth={2}
                  name="Active Users"
                />
                <Line 
                  type="monotone" 
                  dataKey="performanceScore" 
                  stroke="hsl(var(--secondary))" 
                  strokeWidth={2}
                  name="Performance"
                />
              </LineChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      )}

      {!isConnected && data.length === 0 && (
        <Card>
          <CardContent className="flex flex-col items-center justify-center py-12">
            <Activity className="h-12 w-12 text-muted-foreground mb-4" />
            <h3 className="text-lg font-semibold mb-2">Start Real-Time Monitoring</h3>
            <p className="text-muted-foreground text-center mb-4">
              Click the button above to begin tracking live analytics data
            </p>
          </CardContent>
        </Card>
      )}
    </div>
  );
};
