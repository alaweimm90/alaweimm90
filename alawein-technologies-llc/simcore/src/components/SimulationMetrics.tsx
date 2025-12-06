import React, { useState, useEffect, useMemo } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { Separator } from '@/components/ui/separator';
import { 
  Activity, 
  Clock, 
  Cpu, 
  MemoryStick, 
  Zap,
  TrendingUp,
  TrendingDown,
  Minus,
  Download,
  RefreshCw
} from 'lucide-react';
import { useResponsive } from '@/hooks/use-responsive';
import { cn } from '@/lib/utils';

interface MetricItem {
  label: string;
  value: number | string;
  unit?: string;
  format?: 'number' | 'percentage' | 'time' | 'memory' | 'scientific';
  trend?: 'up' | 'down' | 'stable';
  trendValue?: number;
  status?: 'good' | 'warning' | 'error';
  description?: string;
}

interface MetricGroupProps {
  title: string;
  metrics: MetricItem[];
  variant?: 'quantum' | 'statistical' | 'fields' | 'performance';
  showTrends?: boolean;
  collapsible?: boolean;
}

interface PerformanceMetricsProps {
  fps?: number;
  memoryUsage?: number;
  computeTime?: number;
  simulationStep?: number;
  totalSteps?: number;
  convergence?: number;
  accuracy?: number;
  className?: string;
}

interface SimulationProgressProps {
  currentStep: number;
  totalSteps: number;
  isRunning: boolean;
  timeElapsed: number;
  estimatedTimeRemaining?: number;
  onPause?: () => void;
  onResume?: () => void;
  onStop?: () => void;
}

const formatValue = (value: number | string, format?: string, unit?: string) => {
  if (typeof value === 'string') return value;
  
  switch (format) {
    case 'percentage':
      return `${(value * 100).toFixed(1)}%`;
    case 'time':
      if (value < 60) return `${value.toFixed(1)}s`;
      if (value < 3600) return `${Math.floor(value / 60)}m ${(value % 60).toFixed(0)}s`;
      return `${Math.floor(value / 3600)}h ${Math.floor((value % 3600) / 60)}m`;
    case 'memory':
      if (value < 1024) return `${value.toFixed(1)} B`;
      if (value < 1024 * 1024) return `${(value / 1024).toFixed(1)} KB`;
      if (value < 1024 * 1024 * 1024) return `${(value / (1024 * 1024)).toFixed(1)} MB`;
      return `${(value / (1024 * 1024 * 1024)).toFixed(1)} GB`;
    case 'scientific':
      return value.toExponential(2);
    case 'number':
    default:
      const formatted = value.toFixed(2);
      return unit ? `${formatted} ${unit}` : formatted;
  }
};

const getTrendIcon = (trend?: 'up' | 'down' | 'stable') => {
  switch (trend) {
    case 'up': return <TrendingUp className="w-3 h-3 text-green-500" />;
    case 'down': return <TrendingDown className="w-3 h-3 text-red-500" />;
    case 'stable': return <Minus className="w-3 h-3 text-yellow-500" />;
    default: return null;
  }
};

const getStatusColor = (status?: 'good' | 'warning' | 'error') => {
  switch (status) {
    case 'good': return 'text-green-600 dark:text-green-400';
    case 'warning': return 'text-yellow-600 dark:text-yellow-400';
    case 'error': return 'text-red-600 dark:text-red-400';
    default: return 'text-textPrimary';
  }
};

export const MetricGroup: React.FC<MetricGroupProps> = ({
  title,
  metrics,
  variant = 'performance',
  showTrends = true,
  collapsible = false
}) => {
  const [isExpanded, setIsExpanded] = useState(true);
  const { isMobile } = useResponsive();
  
  const variantStyles = {
    quantum: 'border-accentQuantum/20 bg-surfaceGlass',
    statistical: 'border-accentStatistical/20 bg-surfaceGlass',
    fields: 'border-accentField/20 bg-surfaceGlass',
    performance: 'border-muted/20 bg-card'
  };

  return (
    <Card className={cn(
      'transition-all duration-300',
      variantStyles[variant]
    )}>
      <CardHeader 
        className={cn(
          'pb-3',
          collapsible && 'cursor-pointer hover:bg-surfaceMuted/50'
        )}
        onClick={collapsible ? () => setIsExpanded(!isExpanded) : undefined}
      >
        <div className="flex items-center justify-between">
          <CardTitle className={cn(
            'text-sm font-medium text-textPrimary',
            isMobile && 'text-xs'
          )}>
            {title}
          </CardTitle>
          {collapsible && (
            <Button variant="ghost" size="sm">
              <RefreshCw className={cn(
                'w-4 h-4 transition-transform',
                isExpanded && 'rotate-180'
              )} />
            </Button>
          )}
        </div>
      </CardHeader>
      
      {isExpanded && (
        <CardContent className="space-y-3">
          {metrics.map((metric, index) => (
            <div key={index} className="flex items-center justify-between">
              <div className="space-y-1">
                <div className="flex items-center gap-2">
                  <span className={cn(
                    'text-xs font-medium',
                    getStatusColor(metric.status)
                  )}>
                    {metric.label}
                  </span>
                  {showTrends && getTrendIcon(metric.trend)}
                </div>
                {metric.description && (
                  <p className="text-xs text-textSecondary">
                    {metric.description}
                  </p>
                )}
              </div>
              
              <div className="text-right">
                <div className={cn(
                  'text-sm font-mono',
                  getStatusColor(metric.status)
                )}>
                  {formatValue(metric.value, metric.format, metric.unit)}
                </div>
                {showTrends && metric.trendValue !== undefined && (
                  <div className="text-xs text-textSecondary">
                    {metric.trend === 'up' ? '+' : metric.trend === 'down' ? '-' : ''}
                    {Math.abs(metric.trendValue).toFixed(1)}
                  </div>
                )}
              </div>
            </div>
          ))}
        </CardContent>
      )}
    </Card>
  );
};

export const PerformanceMetrics: React.FC<PerformanceMetricsProps> = ({
  fps = 0,
  memoryUsage = 0,
  computeTime = 0,
  simulationStep = 0,
  totalSteps = 0,
  convergence = 0,
  accuracy = 0,
  className
}) => {
  const { isMobile } = useResponsive();
  
  const performanceMetrics: MetricItem[] = [
    {
      label: 'Frame Rate',
      value: fps,
      unit: 'fps',
      format: 'number',
      status: fps > 30 ? 'good' : fps > 15 ? 'warning' : 'error',
      description: 'Rendering performance'
    },
    {
      label: 'Memory Usage',
      value: memoryUsage,
      format: 'memory',
      status: memoryUsage < 100 * 1024 * 1024 ? 'good' : memoryUsage < 200 * 1024 * 1024 ? 'warning' : 'error',
      description: 'RAM consumption'
    },
    {
      label: 'Compute Time',
      value: computeTime,
      format: 'time',
      status: computeTime < 0.1 ? 'good' : computeTime < 0.5 ? 'warning' : 'error',
      description: 'Time per step'
    }
  ];
  
  const simulationMetrics: MetricItem[] = [
    {
      label: 'Progress',
      value: totalSteps > 0 ? simulationStep / totalSteps : 0,
      format: 'percentage',
      description: `${simulationStep} / ${totalSteps} steps`
    },
    {
      label: 'Convergence',
      value: convergence,
      format: 'scientific',
      status: convergence < 1e-6 ? 'good' : convergence < 1e-3 ? 'warning' : 'error',
      description: 'Numerical convergence'
    },
    {
      label: 'Accuracy',
      value: accuracy,
      format: 'percentage', 
      status: accuracy > 0.99 ? 'good' : accuracy > 0.95 ? 'warning' : 'error',
      description: 'Solution accuracy'
    }
  ];
  
  return (
    <div className={cn('space-y-4', className)}>
      <MetricGroup
        title="Performance"
        metrics={performanceMetrics}
        variant="performance"
        showTrends={true}
      />
      
      <MetricGroup
        title="Simulation"
        metrics={simulationMetrics}
        variant="quantum"
        showTrends={true}
      />
    </div>
  );
};

export const SimulationProgress: React.FC<SimulationProgressProps> = ({
  currentStep,
  totalSteps,
  isRunning,
  timeElapsed,
  estimatedTimeRemaining,
  onPause,
  onResume,
  onStop
}) => {
  const { isMobile } = useResponsive();
  const progress = totalSteps > 0 ? (currentStep / totalSteps) * 100 : 0;
  
  return (
    <Card className="border-muted/20 bg-card">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-sm font-medium text-textPrimary">
            Simulation Progress
          </CardTitle>
          <Badge variant={isRunning ? 'default' : 'secondary'}>
            {isRunning ? 'Running' : 'Paused'}
          </Badge>
        </div>
      </CardHeader>
      
      <CardContent className="space-y-4">
        <div className="space-y-2">
          <div className="flex justify-between text-xs text-textSecondary">
            <span>Step {currentStep.toLocaleString()} of {totalSteps.toLocaleString()}</span>
            <span>{progress.toFixed(1)}%</span>
          </div>
          <Progress value={progress} className="h-2" />
        </div>
        
        <div className="grid grid-cols-2 gap-4 text-xs">
          <div>
            <div className="text-textSecondary mb-1">Time Elapsed</div>
            <div className="font-mono text-textPrimary">
              {formatValue(timeElapsed, 'time')}
            </div>
          </div>
          {estimatedTimeRemaining !== undefined && (
            <div>
              <div className="text-textSecondary mb-1">Time Remaining</div>
              <div className="font-mono text-textPrimary">
                {formatValue(estimatedTimeRemaining, 'time')}
              </div>
            </div>
          )}
        </div>
        
        <Separator />
        
        <div className="flex items-center gap-2">
          {isRunning ? (
            <Button
              variant="outline"
              size="sm"
              onClick={onPause}
              className="gap-2"
            >
              <Clock className="w-4 h-4" />
              Pause
            </Button>
          ) : (
            <Button
              variant="default"
              size="sm"
              onClick={onResume}
              className="gap-2"
            >
              <Activity className="w-4 h-4" />
              Resume
            </Button>
          )}
          
          <Button
            variant="outline"
            size="sm"
            onClick={onStop}
            className="gap-2"
          >
            <RefreshCw className="w-4 h-4" />
            Stop
          </Button>
          
          <Button
            variant="ghost"
            size="sm"
            className="ml-auto gap-2"
          >
            <Download className="w-4 h-4" />
            {!isMobile && 'Export'}
          </Button>
        </div>
      </CardContent>
    </Card>
  );
};

// Real-time metrics hook
export const useSimulationMetrics = () => {
  const [metrics, setMetrics] = useState({
    fps: 0,
    memoryUsage: 0,
    computeTime: 0,
    lastUpdateTime: Date.now()
  });
  
  const updateMetrics = (newMetrics: Partial<typeof metrics>) => {
    setMetrics(prev => ({
      ...prev,
      ...newMetrics,
      lastUpdateTime: Date.now()
    }));
  };
  
  // Monitor performance
  useEffect(() => {
    let frameCount = 0;
    let lastTime = performance.now();
    
    const measureFPS = () => {
      frameCount++;
      const currentTime = performance.now();
      
      if (currentTime - lastTime >= 1000) {
        const fps = frameCount * 1000 / (currentTime - lastTime);
        updateMetrics({ fps });
        frameCount = 0;
        lastTime = currentTime;
      }
      
      requestAnimationFrame(measureFPS);
    };
    
    measureFPS();
  }, []);
  
  // Monitor memory usage (if available)
  useEffect(() => {
    const interval = setInterval(() => {
      if ('memory' in performance) {
        const memory = (performance as any).memory;
        updateMetrics({ 
          memoryUsage: memory.usedJSHeapSize 
        });
      }
    }, 2000);
    
    return () => clearInterval(interval);
  }, []);
  
  return { metrics, updateMetrics };
};