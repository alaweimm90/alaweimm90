import React, { useEffect, useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Slider } from '@/components/ui/slider';
import { Badge } from '@/components/ui/badge';
import { Play, Pause, Square, Settings } from 'lucide-react';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Label } from '@/components/ui/label';
import { AdvancedSimulation, SimulationConfig, SimulationState } from '@/lib/advanced-simulation';
import { ShareDialog } from './ShareDialog';
import { collaborationManager, SimulationState as CollabSimulationState } from '@/lib/collaboration-manager';

import { MLAnalysis } from './MLAnalysis';

interface SimulationControlsProps {
  simulation: AdvancedSimulation;
  config: SimulationConfig;
  onStateChange?: (state: SimulationState) => void;
  moduleName?: string;
  onLoadSimulation?: (state: CollabSimulationState) => void;
}

export const SimulationControls: React.FC<SimulationControlsProps> = ({
  simulation,
  config,
  onStateChange,
  moduleName = 'Quantum Simulation',
  onLoadSimulation
}) => {
  const [state, setState] = useState<SimulationState>({
    isRunning: false,
    time: 0,
    deltaTime: config.timestep,
    parameters: {},
    data: {}
  });

  useEffect(() => {
    const unsubscribe = simulation.subscribe((newState) => {
      setState(newState);
      onStateChange?.(newState);
    });
    
    return unsubscribe;
  }, [simulation, onStateChange]);

  const handleParameterChange = (key: string, value: number[]) => {
    simulation.setParameter(key, value[0]);
  };

  const formatTime = (time: number): string => {
    return `${time.toFixed(3)}s`;
  };

  const formatParameter = (key: string, value: number): string => {
    const param = config.parameters[key];
    if (param.step && param.step < 0.1) {
      return value.toFixed(3);
    }
    return value.toFixed(1);
  };

  // Create simulation state for sharing
  const createCollabSimulationState = (): CollabSimulationState => ({
    id: `sim-${Date.now()}`,
    name: `${moduleName} - ${new Date().toLocaleDateString()}`,
    module: moduleName,
    parameters: state.parameters,
    timestamp: Date.now(),
    version: '1.0.0',
  });

  // Handle loading simulation from share
  const handleLoadSimulation = (collabState: CollabSimulationState) => {
    if (onLoadSimulation) {
      onLoadSimulation(collabState);
    } else {
      // Fallback: apply parameters directly
      Object.entries(collabState.parameters).forEach(([key, value]) => {
        if (typeof value === 'number' && config.parameters[key]) {
          simulation.setParameter(key, value);
        }
      });
    }
  };

  return (
    <Card className="bg-card/50 backdrop-blur-sm border-border/50">
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          <span className="flex items-center gap-2">
            <Settings className="w-5 h-5" />
            Simulation Controls
          </span>
          <Badge variant={state.isRunning ? "default" : "secondary"}>
            {state.isRunning ? "Running" : "Stopped"}
          </Badge>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Playback Controls */}
        <div className="flex items-center gap-[--spacing-xs] flex-wrap">
          <Button
            variant={state.isRunning ? "secondary" : "default"}
            size="sm"
            onClick={() => state.isRunning ? simulation.stop() : simulation.start()}
            className="min-h-[--touch-target-min]"
          >
            {state.isRunning ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
            {state.isRunning ? "Pause" : "Start"}
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={() => simulation.reset()}
            className="min-h-[--touch-target-min]"
          >
            <Square className="w-4 h-4" />
            Reset
          </Button>
          <ShareDialog
            simulationState={createCollabSimulationState()}
            onLoadSimulation={handleLoadSimulation}
          />
          <div className="ml-auto text-sm text-muted-foreground">
            Time: {formatTime(state.time)} / {formatTime(config.maxTime)}
          </div>
        </div>

        {/* Progress Bar */}
        <div className="w-full bg-muted rounded-full h-[--spacing-xs]">
          <div 
            className="bg-primary h-[--spacing-xs] rounded-full transition-all duration-200"
            style={{ width: `${(state.time / config.maxTime) * 100}%` }}
          />
        </div>

        {/* Parameter Controls */}
        <Tabs defaultValue="parameters" className="w-full">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="parameters">Parameters</TabsTrigger>
            <TabsTrigger value="advanced">Advanced</TabsTrigger>
            <TabsTrigger value="ml">ML Analysis</TabsTrigger>
          </TabsList>
          
          <TabsContent value="parameters" className="space-y-[--spacing-md] mt-[--spacing-md]">
            {Object.entries(config.parameters).map(([key, param]) => (
              <div key={key} className="space-y-[--spacing-xs]">
                <div className="flex items-center justify-between">
                  <Label className="text-sm font-medium capitalize">
                    {key.replace(/([A-Z])/g, ' $1').trim()}
                  </Label>
                  <span className="text-sm text-muted-foreground">
                    {formatParameter(key, state.parameters[key] || param.default)}
                  </span>
                </div>
                <Slider
                  value={[state.parameters[key] || param.default]}
                  min={param.min}
                  max={param.max}
                  step={param.step || 0.1}
                  onValueChange={(value) => handleParameterChange(key, value)}
                  className="w-full min-h-[--touch-target-min]"
                  aria-label={`${key.replace(/([A-Z])/g, ' $1').trim()} parameter control`}
                />
                <div className="flex justify-between text-xs text-muted-foreground">
                  <span>{param.min}</span>
                  <span>{param.max}</span>
                </div>
              </div>
            ))}
          </TabsContent>
          
          <TabsContent value="advanced" className="space-y-4 mt-4">
            <MLAnalysis
              parameters={state.parameters}
              data={Object.values(state.data).map(array => Array.from(array))}
            />
            
            <div className="space-y-2">
              <Label className="text-sm font-medium">Time Step</Label>
              <div className="text-sm text-muted-foreground">
                Current: {state.deltaTime.toFixed(6)}s
              </div>
            </div>
            
            <div className="space-y-2">
              <Label className="text-sm font-medium">WebGPU Status</Label>
              <Badge variant={navigator.gpu ? "default" : "secondary"}>
                {navigator.gpu ? "Available" : "Not Available"}
              </Badge>
            </div>
            
            <div className="space-y-2">
              <Label className="text-sm font-medium">Data Arrays</Label>
              <div className="text-xs space-y-1">
                {Object.entries(state.data).map(([key, array]) => (
                  <div key={key} className="flex justify-between">
                    <span>{key}:</span>
                    <span>{array.length} elements</span>
                  </div>
                ))}
              </div>
            </div>
          </TabsContent>
          
          <TabsContent value="ml" className="space-y-4 mt-4">
            <MLAnalysis
              parameters={state.parameters}
              data={Object.values(state.data).map(array => Array.from(array))}
            />
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
};

interface RealTimeGraphProps {
  data: Float32Array | number[];
  title: string;
  xLabel?: string;
  yLabel?: string;
  color?: string;
  animate?: boolean;
}

export const RealTimeGraph: React.FC<RealTimeGraphProps> = ({
  data,
  title,
  xLabel = "Index",
  yLabel = "Value",
  color = "#4f46e5",
  animate = true
}) => {
  const canvasRef = React.useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;
    
    // Clear canvas
    ctx.clearRect(0, 0, width, height);
    
    // Set up scaling
    const padding = 40;
    const graphWidth = width - 2 * padding;
    const graphHeight = height - 2 * padding;
    
    const maxValue = Math.max(...Array.from(data));
    const minValue = Math.min(...Array.from(data));
    const range = maxValue - minValue || 1;
    
    // Draw axes
    ctx.strokeStyle = '#666';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(padding, padding);
    ctx.lineTo(padding, height - padding);
    ctx.lineTo(width - padding, height - padding);
    ctx.stroke();
    
    // Draw labels
    ctx.fillStyle = '#888';
    ctx.font = '12px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(xLabel, width / 2, height - 10);
    
    ctx.save();
    ctx.translate(15, height / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText(yLabel, 0, 0);
    ctx.restore();
    
    // Draw data
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    
    for (let i = 0; i < data.length; i++) {
      const x = padding + (i / (data.length - 1)) * graphWidth;
      const y = height - padding - ((data[i] - minValue) / range) * graphHeight;
      
      if (i === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    }
    
    ctx.stroke();
    
    // Add glow effect
    if (animate) {
      ctx.shadowColor = color;
      ctx.shadowBlur = 10;
      ctx.stroke();
      ctx.shadowBlur = 0;
    }
    
  }, [data, color, xLabel, yLabel, animate]);

  return (
    <Card className="bg-card/50 backdrop-blur-sm border-border/50">
      <CardHeader>
        <CardTitle className="text-sm">{title}</CardTitle>
      </CardHeader>
      <CardContent>
        <canvas
          ref={canvasRef}
          width={400}
          height={200}
          className="w-full h-32 border border-border/30 rounded"
        />
      </CardContent>
    </Card>
  );
};