import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { 
  Play, 
  Pause, 
  Square, 
  Download, 
  Share2, 
  Settings, 
  BarChart3,
  Cpu,
  HardDrive,
  Clock,
  Zap,
  TrendingUp,
  Activity,
  Database,
  Cloud
} from 'lucide-react';
import { useToast } from '@/hooks/use-toast';

interface SimulationMetrics {
  fps: number;
  memoryUsage: number;
  cpuUsage: number;
  computeTime: number;
  iterations: number;
  convergence: number;
}

interface SimulationState {
  isRunning: boolean;
  isPaused: boolean;
  progress: number;
  currentStep: number;
  totalSteps: number;
  timeElapsed: number;
}

interface SimulationDashboardProps {
  title: string;
  description?: string;
  children: React.ReactNode;
  onStart?: () => void;
  onPause?: () => void;
  onStop?: () => void;
  onExport?: () => void;
  onShare?: () => void;
  onSettings?: () => void;
  metrics?: SimulationMetrics;
  state?: SimulationState;
  enablePerformanceMonitoring?: boolean;
  enableCloudSync?: boolean;
}

export const SimulationDashboard: React.FC<SimulationDashboardProps> = ({
  title,
  description,
  children,
  onStart,
  onPause,
  onStop,
  onExport,
  onShare,
  onSettings,
  metrics,
  state,
  enablePerformanceMonitoring = true,
  enableCloudSync = false
}) => {
  const [activeTab, setActiveTab] = useState('simulation');
  const [localMetrics, setLocalMetrics] = useState<SimulationMetrics>({
    fps: 60,
    memoryUsage: 45,
    cpuUsage: 30,
    computeTime: 125,
    iterations: 1000,
    convergence: 95.5
  });
  const [localState, setLocalState] = useState<SimulationState>({
    isRunning: false,
    isPaused: false,
    progress: 0,
    currentStep: 0,
    totalSteps: 1000,
    timeElapsed: 0
  });
  const { toast } = useToast();

  const currentMetrics = metrics || localMetrics;
  const currentState = state || localState;

  // Simulate real-time metrics updates
  useEffect(() => {
    if (!metrics && currentState.isRunning) {
      const interval = setInterval(() => {
        setLocalMetrics(prev => ({
          ...prev,
          fps: Math.max(30, prev.fps + (Math.random() - 0.5) * 10),
          memoryUsage: Math.min(90, Math.max(20, prev.memoryUsage + (Math.random() - 0.5) * 5)),
          cpuUsage: Math.min(80, Math.max(10, prev.cpuUsage + (Math.random() - 0.5) * 8)),
          computeTime: prev.computeTime + Math.random() * 10,
          iterations: prev.iterations + Math.floor(Math.random() * 50),
          convergence: Math.min(100, prev.convergence + Math.random() * 0.1)
        }));
        
        setLocalState(prev => ({
          ...prev,
          progress: Math.min(100, prev.progress + Math.random() * 2),
          currentStep: prev.currentStep + Math.floor(Math.random() * 10),
          timeElapsed: prev.timeElapsed + 1
        }));
      }, 1000);

      return () => clearInterval(interval);
    }
  }, [metrics, currentState.isRunning]);

  const handleStart = () => {
    setLocalState(prev => ({ ...prev, isRunning: true, isPaused: false }));
    onStart?.();
    toast({
      title: "Simulation Started",
      description: "Physics simulation is now running.",
    });
  };

  const handlePause = () => {
    setLocalState(prev => ({ ...prev, isPaused: !prev.isPaused }));
    onPause?.();
    toast({
      title: currentState.isPaused ? "Simulation Resumed" : "Simulation Paused",
      description: currentState.isPaused ? "Simulation continues." : "Simulation is paused.",
    });
  };

  const handleStop = () => {
    setLocalState({
      isRunning: false,
      isPaused: false,
      progress: 0,
      currentStep: 0,
      totalSteps: 1000,
      timeElapsed: 0
    });
    onStop?.();
    toast({
      title: "Simulation Stopped",
      description: "All simulation data has been reset.",
    });
  };

  const formatTime = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const getPerformanceStatus = (value: number, thresholds: { good: number; warning: number }) => {
    if (value <= thresholds.good) return 'default';
    if (value <= thresholds.warning) return 'secondary';
    return 'destructive';
  };

  return (
    <div className="w-full space-y-6">
      {/* Header */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="flex items-center gap-2">
                <Activity className="w-5 h-5 text-primary" />
                {title}
              </CardTitle>
              {description && <CardDescription>{description}</CardDescription>}
            </div>
            <div className="flex items-center gap-2">
              {enableCloudSync && (
                <Badge variant="outline" className="flex items-center gap-1">
                  <Cloud className="w-3 h-3" />
                  Cloud Sync
                </Badge>
              )}
              <Badge 
                variant={currentState.isRunning ? 'default' : 'secondary'}
                className="flex items-center gap-1"
              >
                {currentState.isRunning ? (
                  <>
                    <Play className="w-3 h-3" />
                    Running
                  </>
                ) : (
                  <>
                    <Square className="w-3 h-3" />
                    Stopped
                  </>
                )}
              </Badge>
            </div>
          </div>
        </CardHeader>
      </Card>

      {/* Control Panel */}
      <Card>
        <CardContent className="pt-6">
          <div className="flex flex-wrap items-center justify-between gap-4">
            {/* Control Buttons */}
            <div className="flex items-center gap-2">
              <Button 
                onClick={handleStart} 
                disabled={currentState.isRunning}
                className="flex items-center gap-2"
              >
                <Play className="w-4 h-4" />
                Start
              </Button>
              <Button 
                onClick={handlePause} 
                disabled={!currentState.isRunning}
                variant="outline"
                className="flex items-center gap-2"
              >
                <Pause className="w-4 h-4" />
                {currentState.isPaused ? 'Resume' : 'Pause'}
              </Button>
              <Button 
                onClick={handleStop} 
                disabled={!currentState.isRunning && currentState.progress === 0}
                variant="destructive"
                className="flex items-center gap-2"
              >
                <Square className="w-4 h-4" />
                Stop
              </Button>
            </div>

            {/* Progress Info */}
            <div className="flex items-center gap-4 text-sm text-muted-foreground">
              <div className="flex items-center gap-1">
                <Clock className="w-4 h-4" />
                {formatTime(currentState.timeElapsed)}
              </div>
              <div className="flex items-center gap-1">
                <Database className="w-4 h-4" />
                {currentState.currentStep.toLocaleString()} / {currentState.totalSteps.toLocaleString()}
              </div>
            </div>

            {/* Action Buttons */}
            <div className="flex items-center gap-2">
              {onSettings && (
                <Button variant="outline" size="sm" onClick={onSettings}>
                  <Settings className="w-4 h-4" />
                </Button>
              )}
              {onExport && (
                <Button variant="outline" size="sm" onClick={onExport}>
                  <Download className="w-4 h-4" />
                </Button>
              )}
              {onShare && (
                <Button variant="outline" size="sm" onClick={onShare}>
                  <Share2 className="w-4 h-4" />
                </Button>
              )}
            </div>
          </div>

          {/* Progress Bar */}
          {currentState.isRunning && (
            <div className="mt-4 space-y-2">
              <div className="flex justify-between text-sm">
                <span>Progress</span>
                <span>{currentState.progress.toFixed(1)}%</span>
              </div>
              <Progress value={currentState.progress} className="h-2" />
            </div>
          )}
        </CardContent>
      </Card>

      {/* Main Content */}
      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="simulation">Simulation</TabsTrigger>
          <TabsTrigger value="performance">Performance</TabsTrigger>
          <TabsTrigger value="analytics">Analytics</TabsTrigger>
        </TabsList>

        <TabsContent value="simulation" className="space-y-4">
          <Card>
            <CardContent className="pt-6">
              {children}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="performance" className="space-y-4">
          {enablePerformanceMonitoring && (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {/* FPS */}
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm flex items-center gap-2">
                    <Zap className="w-4 h-4" />
                    Frame Rate
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">{currentMetrics.fps.toFixed(1)} FPS</div>
                  <Badge 
                    variant={getPerformanceStatus(60 - currentMetrics.fps, { good: 10, warning: 20 })}
                    className="mt-2"
                  >
                    {currentMetrics.fps >= 50 ? 'Excellent' : currentMetrics.fps >= 30 ? 'Good' : 'Poor'}
                  </Badge>
                </CardContent>
              </Card>

              {/* Memory Usage */}
              <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm flex items-center gap-2">
                  <HardDrive className="w-4 h-4" />
                  Memory Usage
                </CardTitle>
              </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">{currentMetrics.memoryUsage.toFixed(1)}%</div>
                  <Progress value={currentMetrics.memoryUsage} className="mt-2" />
                  <Badge 
                    variant={getPerformanceStatus(currentMetrics.memoryUsage, { good: 50, warning: 75 })}
                    className="mt-2"
                  >
                    {currentMetrics.memoryUsage < 50 ? 'Optimal' : currentMetrics.memoryUsage < 75 ? 'Moderate' : 'High'}
                  </Badge>
                </CardContent>
              </Card>

              {/* CPU Usage */}
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm flex items-center gap-2">
                    <Cpu className="w-4 h-4" />
                    CPU Usage
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">{currentMetrics.cpuUsage.toFixed(1)}%</div>
                  <Progress value={currentMetrics.cpuUsage} className="mt-2" />
                  <Badge 
                    variant={getPerformanceStatus(currentMetrics.cpuUsage, { good: 40, warning: 70 })}
                    className="mt-2"
                  >
                    {currentMetrics.cpuUsage < 40 ? 'Light' : currentMetrics.cpuUsage < 70 ? 'Moderate' : 'Heavy'}
                  </Badge>
                </CardContent>
              </Card>
            </div>
          )}
        </TabsContent>

        <TabsContent value="analytics" className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* Computation Stats */}
            <Card>
              <CardHeader>
                <CardTitle className="text-sm flex items-center gap-2">
                  <BarChart3 className="w-4 h-4" />
                  Computation Statistics
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex justify-between">
                  <span>Total Iterations</span>
                  <span className="font-mono">{currentMetrics.iterations.toLocaleString()}</span>
                </div>
                <div className="flex justify-between">
                  <span>Compute Time</span>
                  <span className="font-mono">{currentMetrics.computeTime.toFixed(2)}ms</span>
                </div>
                <div className="flex justify-between">
                  <span>Convergence</span>
                  <span className="font-mono">{currentMetrics.convergence.toFixed(2)}%</span>
                </div>
              </CardContent>
            </Card>

            {/* Performance Trends */}
            <Card>
              <CardHeader>
                <CardTitle className="text-sm flex items-center gap-2">
                  <TrendingUp className="w-4 h-4" />
                  Performance Trends
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-center py-8 text-muted-foreground">
                  Performance charts and trends visualization would be displayed here in a production environment.
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
};