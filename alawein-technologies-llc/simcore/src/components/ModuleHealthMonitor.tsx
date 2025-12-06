/**
 * Module Health Monitor & Status System
 * 
 * Real-time monitoring of physics module health, performance,
 * and user experience metrics with automated issue detection.
 */

import { useEffect, useState, useMemo, useCallback } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { 
  CheckCircle, 
  AlertTriangle, 
  XCircle, 
  Clock, 
  Zap, 
  Users, 
  TrendingUp,
  RefreshCw,
  Download,
  Bug
} from 'lucide-react';
import { physicsModules } from '@/data/modules';

interface ModuleHealthData {
  id: string;
  name: string;
  status: 'healthy' | 'warning' | 'error' | 'maintenance';
  performance: number; // 0-100
  errorRate: number; // percentage
  loadTime: number; // milliseconds
  userSatisfaction: number; // 0-5 stars
  lastChecked: Date;
  issues: string[];
  metrics: {
    totalLoads: number;
    successfulLoads: number;
    averageLoadTime: number;
    crashCount: number;
    userRating: number;
    cacheHitRate: number;
  };
}

interface SystemHealth {
  overall: 'healthy' | 'warning' | 'error';
  score: number;
  modules: ModuleHealthData[];
  lastUpdate: Date;
  trends: {
    performance: 'improving' | 'stable' | 'declining';
    errors: 'increasing' | 'stable' | 'decreasing';
    satisfaction: 'improving' | 'stable' | 'declining';
  };
}

// Mock health data - in production this would come from analytics
const generateMockHealthData = (): SystemHealth => {
  const modules: ModuleHealthData[] = physicsModules.map(module => {
    const performance = Math.random() * 40 + 60; // 60-100
    const errorRate = Math.random() * 5; // 0-5%
    const loadTime = Math.random() * 2000 + 500; // 500-2500ms
    
    let status: ModuleHealthData['status'] = 'healthy';
    const issues: string[] = [];
    
    if (errorRate > 3) {
      status = 'error';
      issues.push('High error rate detected');
    } else if (errorRate > 1.5) {
      status = 'warning';
      issues.push('Elevated error rate');
    }
    
    if (loadTime > 2000) {
      status = status === 'error' ? 'error' : 'warning';
      issues.push('Slow loading performance');
    }
    
    if (performance < 70) {
      status = status === 'error' ? 'error' : 'warning';
      issues.push('Below target performance');
    }

    return {
      id: module.id,
      name: module.title,
      status,
      performance,
      errorRate,
      loadTime,
      userSatisfaction: Math.random() * 1.5 + 3.5, // 3.5-5
      lastChecked: new Date(),
      issues,
      metrics: {
        totalLoads: Math.floor(Math.random() * 10000 + 1000),
        successfulLoads: Math.floor(Math.random() * 9800 + 950),
        averageLoadTime: loadTime,
        crashCount: Math.floor(Math.random() * 10),
        userRating: Math.random() * 1.5 + 3.5,
        cacheHitRate: Math.random() * 20 + 80 // 80-100%
      }
    };
  });

  const healthyCount = modules.filter(m => m.status === 'healthy').length;
  const warningCount = modules.filter(m => m.status === 'warning').length;
  const errorCount = modules.filter(m => m.status === 'error').length;

  const overallScore = (healthyCount * 100 + warningCount * 60 + errorCount * 20) / modules.length;
  
  let overall: SystemHealth['overall'] = 'healthy';
  if (overallScore < 60) overall = 'error';
  else if (overallScore < 80) overall = 'warning';

  return {
    overall,
    score: overallScore,
    modules,
    lastUpdate: new Date(),
    trends: {
      performance: Math.random() > 0.5 ? 'improving' : 'stable',
      errors: Math.random() > 0.7 ? 'decreasing' : 'stable',
      satisfaction: Math.random() > 0.3 ? 'improving' : 'stable'
    }
  };
};

export function ModuleHealthMonitor() {
  const [healthData, setHealthData] = useState<SystemHealth | null>(null);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [selectedModule, setSelectedModule] = useState<string | null>(null);

  const refreshHealthData = useCallback(async () => {
    setIsRefreshing(true);
    // Simulate API call delay
    await new Promise(resolve => setTimeout(resolve, 1000));
    setHealthData(generateMockHealthData());
    setIsRefreshing(false);
  }, []);

  useEffect(() => {
    refreshHealthData();
    // Set up periodic refresh
    const interval = setInterval(refreshHealthData, 30000); // Every 30 seconds
    return () => clearInterval(interval);
  }, [refreshHealthData]);

  const statusColors = {
    healthy: 'text-green-600 bg-green-50 border-green-200',
    warning: 'text-yellow-600 bg-yellow-50 border-yellow-200',
    error: 'text-red-600 bg-red-50 border-red-200',
    maintenance: 'text-blue-600 bg-blue-50 border-blue-200'
  };

  const statusIcons = {
    healthy: <CheckCircle className="h-4 w-4" />,
    warning: <AlertTriangle className="h-4 w-4" />,
    error: <XCircle className="h-4 w-4" />,
    maintenance: <Clock className="h-4 w-4" />
  };

  if (!healthData) {
    return (
      <Card className="bg-surface/90 backdrop-blur-sm border-muted/30">
        <CardContent className="p-6">
          <div className="flex items-center gap-3">
            <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-interactive" />
            <span className="text-textSecondary">Loading health data...</span>
          </div>
        </CardContent>
      </Card>
    );
  }

  const criticalIssues = healthData.modules.filter(m => m.status === 'error');
  const warningIssues = healthData.modules.filter(m => m.status === 'warning');

  return (
    <div className="space-y-6">
      {/* System Overview */}
      <Card className={`border-2 ${statusColors[healthData.overall]}`}>
        <CardHeader className="pb-4">
          <div className="flex items-center justify-between">
            <CardTitle className="flex items-center gap-2">
              {statusIcons[healthData.overall]}
              System Health: {healthData.overall.charAt(0).toUpperCase() + healthData.overall.slice(1)}
            </CardTitle>
            <Button
              variant="outline"
              size="sm"
              onClick={refreshHealthData}
              disabled={isRefreshing}
              className="bg-surface/50"
            >
              <RefreshCw className={`h-4 w-4 mr-2 ${isRefreshing ? 'animate-spin' : ''}`} />
              Refresh
            </Button>
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex items-center gap-4">
            <div className="flex-1">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm font-medium">Overall Score</span>
                <span className="text-sm font-mono">{healthData.score.toFixed(1)}/100</span>
              </div>
              <Progress value={healthData.score} className="h-2" />
            </div>
          </div>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
            <div className="text-center">
              <div className="text-2xl font-bold text-green-600">
                {healthData.modules.filter(m => m.status === 'healthy').length}
              </div>
              <div className="text-textSecondary">Healthy</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-yellow-600">
                {warningIssues.length}
              </div>
              <div className="text-textSecondary">Warnings</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-red-600">
                {criticalIssues.length}
              </div>
              <div className="text-textSecondary">Errors</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-blue-600">
                {healthData.modules.length}
              </div>
              <div className="text-textSecondary">Total</div>
            </div>
          </div>

          <div className="text-xs text-textMuted">
            Last updated: {healthData.lastUpdate.toLocaleTimeString()}
          </div>
        </CardContent>
      </Card>

      {/* Critical Issues Alert */}
      {criticalIssues.length > 0 && (
        <Alert className="bg-red-50/50 border-red-200">
          <Bug className="h-4 w-4 text-red-600" />
          <AlertDescription className="text-red-800">
            <strong>Critical Issues Detected:</strong> {criticalIssues.length} module(s) require immediate attention.
            <div className="mt-2 space-y-1">
              {criticalIssues.slice(0, 3).map(module => (
                <div key={module.id} className="text-sm">
                  • {module.name}: {module.issues.join(', ')}
                </div>
              ))}
              {criticalIssues.length > 3 && (
                <div className="text-sm">+ {criticalIssues.length - 3} more issues</div>
              )}
            </div>
          </AlertDescription>
        </Alert>
      )}

      {/* Module Details */}
      <Tabs defaultValue="overview" className="w-full">
        <TabsList className="grid w-full grid-cols-3 bg-surface/80 backdrop-blur-sm border border-muted/30">
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="performance">Performance</TabsTrigger>
          <TabsTrigger value="issues">Issues</TabsTrigger>
        </TabsList>

        <TabsContent value="overview" className="space-y-4">
          <div className="grid gap-4">
            {healthData.modules.map(module => (
              <Card 
                key={module.id} 
                className={`bg-surface/90 backdrop-blur-sm border cursor-pointer transition-all duration-300 hover:shadow-elevation2 ${
                  selectedModule === module.id ? 'ring-2 ring-interactive' : 'border-muted/30'
                }`}
                onClick={() => setSelectedModule(selectedModule === module.id ? null : module.id)}
              >
                <CardContent className="p-4">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      {statusIcons[module.status]}
                      <div>
                        <h3 className="font-medium text-textPrimary">{module.name}</h3>
                        <p className="text-sm text-textSecondary">
                          {module.loadTime.toFixed(0)}ms load • {module.errorRate.toFixed(1)}% errors
                        </p>
                      </div>
                    </div>
                    <div className="flex items-center gap-2">
                      <Badge variant="outline" className="text-xs">
                        {module.performance.toFixed(0)}% perf
                      </Badge>
                      <Badge 
                        variant={module.status === 'healthy' ? 'default' : 
                                module.status === 'warning' ? 'secondary' : 'destructive'}
                        className="text-xs"
                      >
                        {module.status}
                      </Badge>
                    </div>
                  </div>

                  {selectedModule === module.id && (
                    <div className="mt-4 pt-4 border-t border-muted/30 space-y-3">
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                        <div>
                          <div className="text-textMuted">Total Loads</div>
                          <div className="font-mono">{module.metrics.totalLoads.toLocaleString()}</div>
                        </div>
                        <div>
                          <div className="text-textMuted">Success Rate</div>
                          <div className="font-mono">
                            {((module.metrics.successfulLoads / module.metrics.totalLoads) * 100).toFixed(1)}%
                          </div>
                        </div>
                        <div>
                          <div className="text-textMuted">User Rating</div>
                          <div className="font-mono">{module.metrics.userRating.toFixed(1)}/5</div>
                        </div>
                        <div>
                          <div className="text-textMuted">Cache Hit</div>
                          <div className="font-mono">{module.metrics.cacheHitRate.toFixed(1)}%</div>
                        </div>
                      </div>

                      {module.issues.length > 0 && (
                        <div>
                          <div className="text-sm font-medium text-textPrimary mb-2">Active Issues:</div>
                          <ul className="space-y-1">
                            {module.issues.map((issue, index) => (
                              <li key={index} className="text-sm text-red-600 flex items-center gap-2">
                                <div className="w-1 h-1 bg-red-600 rounded-full" />
                                {issue}
                              </li>
                            ))}
                          </ul>
                        </div>
                      )}
                    </div>
                  )}
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>

        <TabsContent value="performance" className="space-y-4">
          <Card className="bg-surface/90 backdrop-blur-sm border-muted/30">
            <CardHeader>
              <CardTitle className="text-textPrimary">Performance Metrics</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-6">
                {healthData.modules
                  .sort((a, b) => b.performance - a.performance)
                  .map(module => (
                    <div key={module.id} className="space-y-2">
                      <div className="flex items-center justify-between">
                        <span className="text-sm font-medium text-textPrimary">{module.name}</span>
                        <span className="text-sm font-mono text-textSecondary">
                          {module.performance.toFixed(1)}%
                        </span>
                      </div>
                      <Progress value={module.performance} className="h-2" />
                    </div>
                  ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="issues" className="space-y-4">
          {criticalIssues.length === 0 && warningIssues.length === 0 ? (
            <Card className="bg-green-50/50 border-green-200">
              <CardContent className="p-6 text-center">
                <CheckCircle className="h-12 w-12 text-green-600 mx-auto mb-4" />
                <h3 className="text-lg font-medium text-green-800 mb-2">All Systems Operational</h3>
                <p className="text-green-700">No issues detected across all physics modules.</p>
              </CardContent>
            </Card>
          ) : (
            <div className="space-y-4">
              {criticalIssues.map(module => (
                <Alert key={module.id} className="bg-red-50/50 border-red-200">
                  <XCircle className="h-4 w-4 text-red-600" />
                  <AlertDescription className="text-red-800">
                    <strong>{module.name}:</strong> {module.issues.join(', ')}
                  </AlertDescription>
                </Alert>
              ))}
              
              {warningIssues.map(module => (
                <Alert key={module.id} className="bg-yellow-50/50 border-yellow-200">
                  <AlertTriangle className="h-4 w-4 text-yellow-600" />
                  <AlertDescription className="text-yellow-800">
                    <strong>{module.name}:</strong> {module.issues.join(', ')}
                  </AlertDescription>
                </Alert>
              ))}
            </div>
          )}
        </TabsContent>
      </Tabs>
    </div>
  );
}