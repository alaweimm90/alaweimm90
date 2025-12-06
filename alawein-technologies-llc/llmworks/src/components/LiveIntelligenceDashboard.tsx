import { memo, useState, useEffect, useMemo } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Button } from "@/components/ui/button";
import { 
  Activity, 
  TrendingUp, 
  TrendingDown, 
  Zap, 
  Shield, 
  AlertTriangle, 
  CheckCircle, 
  Clock,
  Users,
  Cpu,
  Database,
  Network,
  RefreshCw,
  Maximize2
} from "lucide-react";

interface SystemMetric {
  id: string;
  label: string;
  value: number;
  unit: string;
  status: 'optimal' | 'warning' | 'critical';
  trend: 'up' | 'down' | 'stable';
  change: number;
}

interface ActiveEvaluation {
  id: string;
  modelA: string;
  modelB: string;
  task: string;
  progress: number;
  estimatedCompletion: string;
  priority: 'high' | 'medium' | 'low';
}

interface ThreatAlert {
  id: string;
  type: 'security' | 'performance' | 'reliability';
  severity: 'low' | 'medium' | 'high' | 'critical';
  message: string;
  timestamp: string;
  resolved: boolean;
}

const LiveIntelligenceDashboardComponent = () => {
  const [isExpanded, setIsExpanded] = useState(false);
  const [lastUpdate, setLastUpdate] = useState(new Date());
  const [autoRefresh, setAutoRefresh] = useState(true);

  // Mock real-time data
  const systemMetrics = useMemo<SystemMetric[]>(() => [
    { id: 'cpu', label: 'Processing Power', value: 73, unit: '%', status: 'optimal', trend: 'up', change: 5.2 },
    { id: 'memory', label: 'Memory Usage', value: 45, unit: '%', status: 'optimal', trend: 'stable', change: 0.1 },
    { id: 'throughput', label: 'Evaluation Throughput', value: 1247, unit: 'req/min', status: 'optimal', trend: 'up', change: 12.3 },
    { id: 'latency', label: 'Response Latency', value: 89, unit: 'ms', status: 'warning', trend: 'down', change: -3.1 },
  ], []);

  const activeEvaluations = useMemo<ActiveEvaluation[]>(() => [
    { 
      id: '1', 
      modelA: 'GPT-4 Turbo', 
      modelB: 'Claude-3 Opus', 
      task: 'Strategic Reasoning Challenge',
      progress: 78,
      estimatedCompletion: '2m 34s',
      priority: 'high'
    },
    { 
      id: '2', 
      modelA: 'Gemini Ultra', 
      modelB: 'GPT-3.5 Turbo', 
      task: 'Creative Problem Solving',
      progress: 34,
      estimatedCompletion: '5m 12s',
      priority: 'medium'
    },
    { 
      id: '3', 
      modelA: 'Claude-3 Sonnet', 
      modelB: 'LLaMA-2 70B', 
      task: 'Analytical Benchmarking',
      progress: 91,
      estimatedCompletion: '45s',
      priority: 'low'
    },
  ], []);

  const threatAlerts = useMemo<ThreatAlert[]>(() => [
    {
      id: '1',
      type: 'performance',
      severity: 'medium',
      message: 'Elevated response times detected in EU-West region',
      timestamp: '2 minutes ago',
      resolved: false
    },
    {
      id: '2',
      type: 'security',
      severity: 'low',
      message: 'Unusual traffic pattern from 192.168.1.x range',
      timestamp: '7 minutes ago',
      resolved: true
    },
  ], []);

  // Auto-refresh simulation
  useEffect(() => {
    if (!autoRefresh) return;
    
    const interval = setInterval(() => {
      setLastUpdate(new Date());
    }, 5000); // Update every 5 seconds

    return () => clearInterval(interval);
  }, [autoRefresh]);

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'optimal': return 'text-green-500';
      case 'warning': return 'text-yellow-500';
      case 'critical': return 'text-red-500';
      default: return 'text-muted-foreground';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'optimal': return <CheckCircle className="h-4 w-4 text-green-500" />;
      case 'warning': return <AlertTriangle className="h-4 w-4 text-yellow-500" />;
      case 'critical': return <AlertTriangle className="h-4 w-4 text-red-500" />;
      default: return <Clock className="h-4 w-4 text-muted-foreground" />;
    }
  };

  const getTrendIcon = (trend: string, change: number) => {
    const color = change > 0 ? 'text-green-500' : change < 0 ? 'text-red-500' : 'text-muted-foreground';
    switch (trend) {
      case 'up': return <TrendingUp className={`h-3 w-3 ${color}`} />;
      case 'down': return <TrendingDown className={`h-3 w-3 ${color}`} />;
      default: return <div className="w-3 h-3 rounded-full bg-muted-foreground/50" />;
    }
  };

  const getPriorityColor = (priority: string) => {
    switch (priority) {
      case 'high': return 'rank-gold strategic-rank';
      case 'medium': return 'rank-silver strategic-rank';
      case 'low': return 'rank-bronze strategic-rank';
      default: return 'performance-standard strategic-rank';
    }
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical': return 'text-red-500 bg-red-500/10 border-red-500/30';
      case 'high': return 'text-orange-500 bg-orange-500/10 border-orange-500/30';
      case 'medium': return 'text-yellow-500 bg-yellow-500/10 border-yellow-500/30';
      case 'low': return 'text-blue-500 bg-blue-500/10 border-blue-500/30';
      default: return 'text-muted-foreground bg-muted/10 border-muted/30';
    }
  };

  return (
    <div className={`glass-panel transition-all duration-500 ${isExpanded ? 'fixed inset-4 z-50' : 'w-full max-w-4xl'}`}>
      <CardHeader className="pb-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="glass-subtle p-2 rounded-xl">
              <Activity className="h-5 w-5 text-primary animate-pulse" />
            </div>
            <div>
              <CardTitle className="heading-refined text-lg">Live Intelligence Dashboard</CardTitle>
              <p className="text-xs text-muted-foreground">
                Real-time strategic command center • Last updated: {lastUpdate.toLocaleTimeString()}
              </p>
            </div>
          </div>
          
          <div className="flex items-center gap-2">
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setAutoRefresh(!autoRefresh)}
              className="glass-minimal px-3 py-1"
            >
              <RefreshCw className={`h-4 w-4 mr-2 ${autoRefresh ? 'animate-spin' : ''}`} />
              <span className="text-xs">{autoRefresh ? 'LIVE' : 'PAUSED'}</span>
            </Button>
            
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setIsExpanded(!isExpanded)}
              className="glass-minimal p-2"
            >
              <Maximize2 className="h-4 w-4" />
            </Button>
          </div>
        </div>
      </CardHeader>

      <CardContent className="space-y-6">
        {/* System Metrics Grid */}
        <div>
          <h3 className="heading-refined text-sm mb-4 flex items-center gap-2">
            <div className="w-1 h-4 bg-gradient-to-b from-primary to-secondary rounded-full"></div>
            System Performance Matrix
          </h3>
          
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
            {systemMetrics.map((metric) => (
              <Card key={metric.id} className="glass-subtle border-border/20 hover:shadow-md transition-all duration-300">
                <CardContent className="p-4">
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center gap-2">
                      {getStatusIcon(metric.status)}
                      <span className="text-xs font-medium">{metric.label}</span>
                    </div>
                    {getTrendIcon(metric.trend, metric.change)}
                  </div>
                  
                  <div className="flex items-end gap-2">
                    <span className="heading-display text-xl font-bold">{metric.value.toLocaleString()}</span>
                    <span className="text-xs text-muted-foreground mb-1">{metric.unit}</span>
                  </div>
                  
                  <div className="flex items-center gap-2 mt-2">
                    <Progress value={Math.min(metric.value, 100)} className="h-1 flex-1" />
                    <span className={`text-xs font-mono ${metric.change > 0 ? 'text-green-500' : metric.change < 0 ? 'text-red-500' : 'text-muted-foreground'}`}>
                      {metric.change > 0 ? '+' : ''}{metric.change}%
                    </span>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>

        {/* Active Evaluations */}
        <div>
          <h3 className="heading-refined text-sm mb-4 flex items-center gap-2">
            <div className="w-1 h-4 bg-gradient-to-b from-secondary to-accent rounded-full"></div>
            Active Strategic Evaluations
          </h3>
          
          <div className="space-y-3">
            {activeEvaluations.map((evaluation) => (
              <Card key={evaluation.id} className="glass-subtle border-border/20 hover:shadow-md transition-all duration-300">
                <CardContent className="p-4">
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center gap-3">
                      <Badge className={getPriorityColor(evaluation.priority)}>
                        {evaluation.priority}
                      </Badge>
                      <div>
                        <div className="text-sm font-medium">{evaluation.task}</div>
                        <div className="text-xs text-muted-foreground">
                          {evaluation.modelA} vs {evaluation.modelB}
                        </div>
                      </div>
                    </div>
                    
                    <div className="text-right">
                      <div className="text-sm font-bold text-primary">{evaluation.progress}%</div>
                      <div className="text-xs text-muted-foreground">ETA: {evaluation.estimatedCompletion}</div>
                    </div>
                  </div>
                  
                  <Progress value={evaluation.progress} className="h-2" />
                </CardContent>
              </Card>
            ))}
          </div>
        </div>

        {/* Threat Intelligence */}
        <div>
          <h3 className="heading-refined text-sm mb-4 flex items-center gap-2">
            <div className="w-1 h-4 bg-gradient-to-b from-accent to-primary rounded-full"></div>
            Threat Intelligence Alerts
          </h3>
          
          <div className="space-y-2">
            {threatAlerts.map((alert) => (
              <div key={alert.id} className={`glass-minimal p-3 rounded-lg border ${getSeverityColor(alert.severity)} ${alert.resolved ? 'opacity-50' : ''}`}>
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <div className="flex items-center gap-2">
                      {alert.type === 'security' && <Shield className="h-4 w-4" />}
                      {alert.type === 'performance' && <Zap className="h-4 w-4" />}
                      {alert.type === 'reliability' && <Database className="h-4 w-4" />}
                      <Badge variant="outline" className={`text-xs ${getSeverityColor(alert.severity)}`}>
                        {alert.severity.toUpperCase()}
                      </Badge>
                    </div>
                    
                    <div>
                      <div className="text-sm">{alert.message}</div>
                      <div className="text-xs text-muted-foreground">{alert.timestamp}</div>
                    </div>
                  </div>
                  
                  {alert.resolved && (
                    <CheckCircle className="h-4 w-4 text-green-500" />
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Command Status */}
        <div className="flex items-center justify-center pt-4 border-t border-border/10">
          <div className="flex items-center gap-6">
            <div className="flex items-center gap-2 glass-minimal px-3 py-2 rounded-full">
              <Users className="h-4 w-4 text-primary" />
              <span className="text-xs font-medium">47 Active Models</span>
            </div>
            
            <div className="flex items-center gap-2 glass-minimal px-3 py-2 rounded-full">
              <Cpu className="h-4 w-4 text-secondary" />
              <span className="text-xs font-medium">12.4K Evaluations Today</span>
            </div>
            
            <div className="flex items-center gap-2 glass-minimal px-3 py-2 rounded-full">
              <Network className="h-4 w-4 text-accent" />
              <span className="text-xs font-medium">99.7% Uptime</span>
            </div>
          </div>
        </div>
      </CardContent>
    </div>
  );
};

export const LiveIntelligenceDashboard = memo(LiveIntelligenceDashboardComponent);