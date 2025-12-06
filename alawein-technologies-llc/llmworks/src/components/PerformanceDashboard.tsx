import { memo, useEffect, useState } from "react";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { 
  Activity, 
  Zap, 
  Clock, 
  Eye, 
  AlertTriangle,
  CheckCircle,
  TrendingUp,
  Wifi,
  WifiOff,
  RefreshCw,
} from "lucide-react";
import { 
  getPerformanceMetrics, 
  getPerformanceScore, 
  analyzeResourceTiming 
} from "@/lib/performance";
import { getSwStatus, isOffline } from "@/lib/service-worker";

interface PerformanceMetrics {
  lcp?: number;
  fid?: number;
  cls?: number;
  ttfb?: number;
  [key: string]: number | undefined;
}

interface PerformanceData {
  score: number;
  metrics: PerformanceMetrics;
  swStatus: {
    supported: boolean;
    registered: boolean;
    active: boolean;
    waiting: boolean;
  };
  offline: boolean;
}

export const PerformanceDashboard = memo(() => {
  const [data, setData] = useState<PerformanceData>({
    score: 0,
    metrics: {},
    swStatus: { supported: false, registered: false, active: false, waiting: false },
    offline: false,
  });
  
  const [isLoading, setIsLoading] = useState(true);

  const loadData = async () => {
    try {
      const [score, metrics, swStatus] = await Promise.all([
        getPerformanceScore(),
        getPerformanceMetrics(),
        getSwStatus(),
      ]);

      setData({
        score,
        metrics,
        swStatus,
        offline: isOffline(),
      });
    } catch (error) {
      console.error('[PerformanceDashboard] Failed to load data:', error);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    loadData();
    
    // Refresh data every 30 seconds
    const interval = setInterval(loadData, 30000);
    return () => clearInterval(interval);
  }, []);

  const getScoreColor = (score: number) => {
    if (score >= 90) return "text-green-600";
    if (score >= 70) return "text-yellow-600";
    return "text-red-600";
  };

  const getScoreVariant = (score: number) => {
    if (score >= 90) return "default";
    if (score >= 70) return "secondary";
    return "destructive";
  };

  const formatTime = (ms?: number) => {
    if (ms === undefined) return "N/A";
    if (ms < 1000) return `${Math.round(ms)}ms`;
    return `${(ms / 1000).toFixed(2)}s`;
  };

  if (isLoading) {
    return (
      <Card className="p-6">
        <div className="flex items-center justify-center">
          <RefreshCw className="h-6 w-6 animate-spin text-primary mr-2" />
          <span>Loading performance data...</span>
        </div>
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-foreground">Performance Dashboard</h2>
          <p className="text-muted-foreground">Real-time performance monitoring and Core Web Vitals</p>
        </div>
        <div className="flex items-center gap-2">
          {data.offline ? (
            <Badge variant="destructive" className="flex items-center gap-1">
              <WifiOff className="h-3 w-3" />
              Offline
            </Badge>
          ) : (
            <Badge variant="default" className="flex items-center gap-1">
              <Wifi className="h-3 w-3" />
              Online
            </Badge>
          )}
          <Button variant="outline" size="sm" onClick={loadData}>
            <RefreshCw className="h-4 w-4 mr-2" />
            Refresh
          </Button>
        </div>
      </div>

      {/* Performance Score Overview */}
      <Card className="p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold">Overall Performance Score</h3>
          <Badge variant={getScoreVariant(data.score)} className="text-lg px-3 py-1">
            {data.score}/100
          </Badge>
        </div>
        
        <div className="w-full bg-gray-200 rounded-full h-3 mb-4">
          <div
            className={`h-3 rounded-full transition-all duration-300 ${
              data.score >= 90 ? "bg-green-600" : 
              data.score >= 70 ? "bg-yellow-600" : "bg-red-600"
            }`}
            style={{ width: `${data.score}%` }}
          />
        </div>
        
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
          <div className="text-center">
            <div className="font-medium text-green-600">Good</div>
            <div className="text-muted-foreground">90-100</div>
          </div>
          <div className="text-center">
            <div className="font-medium text-yellow-600">Needs Improvement</div>
            <div className="text-muted-foreground">50-89</div>
          </div>
          <div className="text-center">
            <div className="font-medium text-red-600">Poor</div>
            <div className="text-muted-foreground">0-49</div>
          </div>
          <div className="text-center">
            <div className="font-medium">Current</div>
            <div className={`font-bold ${getScoreColor(data.score)}`}>{data.score}</div>
          </div>
        </div>
      </Card>

      <Tabs defaultValue="vitals" className="w-full">
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="vitals">Core Web Vitals</TabsTrigger>
          <TabsTrigger value="caching">Caching & SW</TabsTrigger>
          <TabsTrigger value="resources">Resource Analysis</TabsTrigger>
        </TabsList>

        <TabsContent value="vitals" className="space-y-4">
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-4">
            {/* Largest Contentful Paint */}
            <Card className="p-4">
              <div className="flex items-center gap-2 mb-2">
                <Eye className="h-4 w-4 text-primary" />
                <span className="font-medium">LCP</span>
              </div>
              <div className="text-2xl font-bold">
                {formatTime(data.metrics.lcp)}
              </div>
              <p className="text-xs text-muted-foreground">Largest Contentful Paint</p>
              <div className="mt-2">
                <Badge 
                  variant={
                    !data.metrics.lcp ? "secondary" :
                    data.metrics.lcp <= 2500 ? "default" :
                    data.metrics.lcp <= 4000 ? "secondary" : "destructive"
                  }
                  className="text-xs"
                >
                  {!data.metrics.lcp ? "Loading..." :
                   data.metrics.lcp <= 2500 ? "Good" :
                   data.metrics.lcp <= 4000 ? "Needs Improvement" : "Poor"}
                </Badge>
              </div>
            </Card>

            {/* First Input Delay */}
            <Card className="p-4">
              <div className="flex items-center gap-2 mb-2">
                <Zap className="h-4 w-4 text-primary" />
                <span className="font-medium">FID</span>
              </div>
              <div className="text-2xl font-bold">
                {formatTime(data.metrics.fid)}
              </div>
              <p className="text-xs text-muted-foreground">First Input Delay</p>
              <div className="mt-2">
                <Badge 
                  variant={
                    !data.metrics.fid ? "secondary" :
                    data.metrics.fid <= 100 ? "default" :
                    data.metrics.fid <= 300 ? "secondary" : "destructive"
                  }
                  className="text-xs"
                >
                  {!data.metrics.fid ? "Loading..." :
                   data.metrics.fid <= 100 ? "Good" :
                   data.metrics.fid <= 300 ? "Needs Improvement" : "Poor"}
                </Badge>
              </div>
            </Card>

            {/* Cumulative Layout Shift */}
            <Card className="p-4">
              <div className="flex items-center gap-2 mb-2">
                <TrendingUp className="h-4 w-4 text-primary" />
                <span className="font-medium">CLS</span>
              </div>
              <div className="text-2xl font-bold">
                {data.metrics.cls ? data.metrics.cls.toFixed(3) : "N/A"}
              </div>
              <p className="text-xs text-muted-foreground">Cumulative Layout Shift</p>
              <div className="mt-2">
                <Badge 
                  variant={
                    !data.metrics.cls ? "secondary" :
                    data.metrics.cls <= 0.1 ? "default" :
                    data.metrics.cls <= 0.25 ? "secondary" : "destructive"
                  }
                  className="text-xs"
                >
                  {!data.metrics.cls ? "Loading..." :
                   data.metrics.cls <= 0.1 ? "Good" :
                   data.metrics.cls <= 0.25 ? "Needs Improvement" : "Poor"}
                </Badge>
              </div>
            </Card>

            {/* Time to First Byte */}
            <Card className="p-4">
              <div className="flex items-center gap-2 mb-2">
                <Clock className="h-4 w-4 text-primary" />
                <span className="font-medium">TTFB</span>
              </div>
              <div className="text-2xl font-bold">
                {formatTime(data.metrics.ttfb)}
              </div>
              <p className="text-xs text-muted-foreground">Time to First Byte</p>
              <div className="mt-2">
                <Badge 
                  variant={
                    !data.metrics.ttfb ? "secondary" :
                    data.metrics.ttfb <= 200 ? "default" :
                    data.metrics.ttfb <= 500 ? "secondary" : "destructive"
                  }
                  className="text-xs"
                >
                  {!data.metrics.ttfb ? "Loading..." :
                   data.metrics.ttfb <= 200 ? "Good" :
                   data.metrics.ttfb <= 500 ? "Needs Improvement" : "Poor"}
                </Badge>
              </div>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="caching" className="space-y-4">
          <div className="grid md:grid-cols-2 gap-4">
            {/* Service Worker Status */}
            <Card className="p-6">
              <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                <Activity className="h-5 w-5" />
                Service Worker Status
              </h3>
              
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <span>Supported</span>
                  {data.swStatus.supported ? (
                    <CheckCircle className="h-4 w-4 text-green-600" />
                  ) : (
                    <AlertTriangle className="h-4 w-4 text-red-600" />
                  )}
                </div>
                
                <div className="flex items-center justify-between">
                  <span>Registered</span>
                  {data.swStatus.registered ? (
                    <CheckCircle className="h-4 w-4 text-green-600" />
                  ) : (
                    <AlertTriangle className="h-4 w-4 text-red-600" />
                  )}
                </div>
                
                <div className="flex items-center justify-between">
                  <span>Active</span>
                  {data.swStatus.active ? (
                    <CheckCircle className="h-4 w-4 text-green-600" />
                  ) : (
                    <AlertTriangle className="h-4 w-4 text-red-600" />
                  )}
                </div>
                
                <div className="flex items-center justify-between">
                  <span>Update Available</span>
                  {data.swStatus.waiting ? (
                    <AlertTriangle className="h-4 w-4 text-yellow-600" />
                  ) : (
                    <CheckCircle className="h-4 w-4 text-green-600" />
                  )}
                </div>
              </div>
            </Card>

            {/* Connection Status */}
            <Card className="p-6">
              <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                {data.offline ? (
                  <WifiOff className="h-5 w-5 text-red-600" />
                ) : (
                  <Wifi className="h-5 w-5 text-green-600" />
                )}
                Connection Status
              </h3>
              
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <span>Status</span>
                  <Badge variant={data.offline ? "destructive" : "default"}>
                    {data.offline ? "Offline" : "Online"}
                  </Badge>
                </div>
                
                <div className="flex items-center justify-between">
                  <span>Caching</span>
                  <Badge variant={data.swStatus.active ? "default" : "secondary"}>
                    {data.swStatus.active ? "Active" : "Inactive"}
                  </Badge>
                </div>
              </div>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="resources">
          <Card className="p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold">Resource Analysis</h3>
              <Button variant="outline" size="sm" onClick={analyzeResourceTiming}>
                <RefreshCw className="h-4 w-4 mr-2" />
                Analyze Resources
              </Button>
            </div>
            <p className="text-muted-foreground">
              Click "Analyze Resources" to see detailed resource timing analysis in the console.
              This will show the slowest loading resources and resource type breakdown.
            </p>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
});

PerformanceDashboard.displayName = 'PerformanceDashboard';