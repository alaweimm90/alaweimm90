import React, { useState, useEffect } from 'react';
import { BarChart3, TrendingUp, Clock } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Progress } from '@/components/ui/progress';
import { Badge } from '@/components/ui/badge';
import { cn } from '@/lib/utils';

interface AnalyticsData {
  pageViews: number;
  uniqueUsers: number;
  avgSessionTime: string;
  topSimulations: Array<{ name: string; count: number; trend: number }>;
  performanceMetrics: {
    loadTime: number;
    renderTime: number;
    interactionScore: number;
  };
}

interface AdvancedAnalyticsProps {
  className?: string;
  variant?: 'full' | 'compact';
}

export const AdvancedAnalytics: React.FC<AdvancedAnalyticsProps> = ({ className, variant = 'full' }) => {
  const [analyticsData, setAnalyticsData] = useState<AnalyticsData>({
    pageViews: 12847,
    uniqueUsers: 3421,
    avgSessionTime: '4m 32s',
    topSimulations: [
      { name: 'Graphene Band Structure', count: 2847, trend: 12.3 },
      { name: 'Ising Model', count: 1923, trend: 8.7 },
      { name: 'LLG Dynamics', count: 1456, trend: -2.1 },
      { name: 'MoS2 Valley Physics', count: 1203, trend: 15.4 },
      { name: 'Quantum Tunneling', count: 987, trend: 5.8 }
    ],
    performanceMetrics: {
      loadTime: 1.2,
      renderTime: 0.8,
      interactionScore: 94
    }
  });

  const [realTimeMetrics, setRealTimeMetrics] = useState({
    activeUsers: 43,
    currentLoad: 67,
    responseTime: 245
  });

  useEffect(() => {
    const interval = setInterval(() => {
      setRealTimeMetrics(prev => ({
        activeUsers: prev.activeUsers + Math.floor(Math.random() * 6) - 3,
        currentLoad: Math.max(20, Math.min(100, prev.currentLoad + Math.floor(Math.random() * 20) - 10)),
        responseTime: prev.responseTime + Math.floor(Math.random() * 100) - 50
      }));
    }, 3000);

    return () => clearInterval(interval);
  }, []);

  const getPerformanceColor = (score: number) => {
    if (score >= 90) return 'text-accent-foreground';
    if (score >= 70) return 'text-secondary-foreground';
    return 'text-destructive';
  };

  const getTrendColor = (trend: number) => {
    if (trend > 0) return 'text-accent-foreground';
    if (trend < 0) return 'text-destructive';
    return 'text-muted-foreground';
  };

  // Compact variant early return
  if (variant === 'compact') {
    return (
      <Card className={cn("w-full", className)}>
        <CardHeader className="py-3">
          <CardTitle className="flex items-center gap-2 text-sm">
            <BarChart3 className="w-4 h-4" />
            Analytics
          </CardTitle>
        </CardHeader>
        <CardContent className="pt-0">
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            <div className="p-3 bg-muted/50 rounded-lg">
              <p className="text-[10px] text-muted-foreground">Views</p>
              <p className="text-base font-bold">{analyticsData.pageViews.toLocaleString()}</p>
            </div>
            <div className="p-3 bg-muted/50 rounded-lg">
              <p className="text-[10px] text-muted-foreground">Perf</p>
              <p className="text-base font-bold">{analyticsData.performanceMetrics.interactionScore}%</p>
            </div>
          </div>
        </CardContent>
      </Card>
    );
  }
  return (
    <Card className={cn("w-full", className)}>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <BarChart3 className="w-5 h-5" />
          Analytics
        </CardTitle>
      </CardHeader>
      <CardContent>
        <Tabs defaultValue="overview" className="w-full">
          <TabsList className="grid w-full grid-cols-2">
            <TabsTrigger value="overview">Overview</TabsTrigger>
            <TabsTrigger value="realtime">Real-time</TabsTrigger>
          </TabsList>

          <TabsContent value="overview" className="space-y-4">
            <div className="grid grid-cols-1 gap-3">
              <div className="p-3 bg-muted/50 rounded-lg">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-xs text-muted-foreground">Views</p>
                    <p className="text-lg font-bold">{analyticsData.pageViews.toLocaleString()}</p>
                  </div>
                  <TrendingUp className="w-6 h-6 text-primary" />
                </div>
              </div>


              <div className="p-3 bg-muted/50 rounded-lg">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-xs text-muted-foreground">Avg Session</p>
                    <p className="text-lg font-bold">{analyticsData.avgSessionTime}</p>
                  </div>
                  <Clock className="w-6 h-6 text-primary" />
                </div>
              </div>
            </div>

            <div className="space-y-2">
              <h4 className="text-sm font-medium">Top Simulations</h4>
              {analyticsData.topSimulations.slice(0, 3).map((sim, index) => (
                <div key={sim.name} className="flex items-center justify-between p-2 rounded bg-muted/30">
                  <div className="flex items-center gap-2">
                    <Badge variant="secondary" className="text-xs">#{index + 1}</Badge>
                    <span className="text-xs font-medium truncate">{sim.name}</span>
                  </div>
                  <span className={cn("text-xs font-medium", getTrendColor(sim.trend))}>
                    {sim.trend > 0 ? '+' : ''}{sim.trend}%
                  </span>
                </div>
              ))}
            </div>
          </TabsContent>

          <TabsContent value="realtime" className="space-y-4">

              <div className="p-spacing-sm bg-muted/50 rounded-lg border border-border">
                <div className="space-y-spacing-xs">
                  <div className="flex justify-between">
                    <span className="text-xs text-muted-foreground">Server Load</span>
                    <span className="text-xs font-medium text-foreground">{realTimeMetrics.currentLoad}%</span>
                  </div>
                  <Progress value={realTimeMetrics.currentLoad} className="h-2" />
                </div>
              </div>

              <div className="p-spacing-sm bg-muted/50 rounded-lg border border-border">
                <div className="text-center">
                  <p className="text-xs text-muted-foreground">Response Time</p>
                  <p className="text-lg font-bold text-foreground">{realTimeMetrics.responseTime}ms</p>
                </div>
              </div>
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
};