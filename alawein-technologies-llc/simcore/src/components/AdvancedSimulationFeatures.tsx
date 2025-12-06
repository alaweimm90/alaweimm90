/**
 * Advanced Simulation Features
 * Provides enhanced simulation capabilities and controls
 */

import React, { useState, useEffect, useCallback } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Slider } from '@/components/ui/slider';
import { Switch } from '@/components/ui/switch';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Progress } from '@/components/ui/progress';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { 
  Play, 
  Pause, 
  Square, 
  RotateCcw, 
  Settings, 
  Zap,
  TrendingUp,
  BarChart3,
  Clock,
  Target,
  Cpu
} from 'lucide-react';
import { MLAnalysis } from './MLAnalysis';
import { SimulationExportSystem } from './SimulationExportSystem';

interface SimulationState {
  isRunning: boolean;
  isPaused: boolean;
  currentStep: number;
  totalSteps: number;
  timeElapsed: number;
  convergenceStatus: 'converging' | 'converged' | 'diverging' | 'unknown';
}

interface SimulationParameters {
  timeStep: number;
  maxSteps: number;
  tolerance: number;
  adaptiveStep: boolean;
  recordingInterval: number;
  autoStop: boolean;
  parallelProcessing: boolean;
  precision: 'single' | 'double' | 'extended';
}

interface AdvancedSimulationFeaturesProps {
  onParameterChange?: (params: SimulationParameters) => void;
  onStateChange?: (state: SimulationState) => void;
  simulationType?: string;
  initialParameters?: Partial<SimulationParameters>;
}

export function AdvancedSimulationFeatures({
  onParameterChange,
  onStateChange,
  simulationType = 'physics',
  initialParameters = {}
}: AdvancedSimulationFeaturesProps) {
  const [simulationState, setSimulationState] = useState<SimulationState>({
    isRunning: false,
    isPaused: false,
    currentStep: 0,
    totalSteps: 1000,
    timeElapsed: 0,
    convergenceStatus: 'unknown'
  });

  const [parameters, setParameters] = useState<SimulationParameters>({
    timeStep: 0.01,
    maxSteps: 1000,
    tolerance: 1e-6,
    adaptiveStep: true,
    recordingInterval: 10,
    autoStop: true,
    parallelProcessing: false,
    precision: 'double',
    ...initialParameters
  });

  const [simulationData, setSimulationData] = useState({
    parameters: {},
    results: [],
    metadata: { created: new Date().toISOString() },
    analysis: []
  });

  const [performanceMetrics, setPerformanceMetrics] = useState({
    stepsPerSecond: 0,
    memoryUsage: 0,
    cpuUsage: 0,
    convergenceRate: 0
  });

  // Simulation control functions
  const startSimulation = useCallback(() => {
    setSimulationState(prev => ({
      ...prev,
      isRunning: true,
      isPaused: false,
      currentStep: 0,
      timeElapsed: 0
    }));
  }, []);

  const pauseSimulation = useCallback(() => {
    setSimulationState(prev => ({ ...prev, isPaused: !prev.isPaused }));
  }, []);

  const stopSimulation = useCallback(() => {
    setSimulationState(prev => ({
      ...prev,
      isRunning: false,
      isPaused: false
    }));
  }, []);

  const resetSimulation = useCallback(() => {
    setSimulationState(prev => ({
      ...prev,
      isRunning: false,
      isPaused: false,
      currentStep: 0,
      timeElapsed: 0,
      convergenceStatus: 'unknown'
    }));
    setSimulationData({
      parameters: {},
      results: [],
      metadata: { created: new Date().toISOString() },
      analysis: []
    });
  }, []);

  // Simulate running simulation
  useEffect(() => {
    let interval: NodeJS.Timeout;
    
    if (simulationState.isRunning && !simulationState.isPaused) {
      interval = setInterval(() => {
        setSimulationState(prev => {
          const newStep = prev.currentStep + 1;
          const newTime = prev.timeElapsed + parameters.timeStep;
          
          // Check convergence
          let convergenceStatus = prev.convergenceStatus;
          if (newStep > 100) {
            const rand = Math.random();
            if (rand < 0.7) convergenceStatus = 'converging';
            else if (rand < 0.9) convergenceStatus = 'converged';
            else convergenceStatus = 'diverging';
          }
          
          // Auto-stop if converged
          const shouldStop = parameters.autoStop && 
            (convergenceStatus === 'converged' || newStep >= parameters.maxSteps);
          
          return {
            ...prev,
            currentStep: newStep,
            timeElapsed: newTime,
            convergenceStatus,
            isRunning: !shouldStop,
            isPaused: shouldStop ? false : prev.isPaused
          };
        });
        
        // Update performance metrics
        setPerformanceMetrics(prev => ({
          stepsPerSecond: Math.random() * 100 + 50,
          memoryUsage: Math.random() * 30 + 20,
          cpuUsage: Math.random() * 40 + 30,
          convergenceRate: Math.random() * 0.1 + 0.85
        }));
      }, 50);
    }
    
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [simulationState.isRunning, simulationState.isPaused, parameters]);

  // Notify parent components of changes
  useEffect(() => {
    onParameterChange?.(parameters);
  }, [parameters, onParameterChange]);

  useEffect(() => {
    onStateChange?.(simulationState);
  }, [simulationState, onStateChange]);

  const updateParameter = <K extends keyof SimulationParameters>(
    key: K,
    value: SimulationParameters[K]
  ) => {
    setParameters(prev => ({ ...prev, [key]: value }));
  };

  const getConvergenceColor = (status: SimulationState['convergenceStatus']) => {
    switch (status) {
      case 'converged': return 'text-green-600';
      case 'converging': return 'text-blue-600';
      case 'diverging': return 'text-red-600';
      default: return 'text-muted-foreground';
    }
  };

  const progress = (simulationState.currentStep / simulationState.totalSteps) * 100;

  return (
    <div className="space-y-6">
      {/* Simulation Controls */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Zap className="h-5 w-5" />
            Advanced Simulation Controls
          </CardTitle>
          <CardDescription>
            Enhanced simulation management with adaptive algorithms
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* Control Buttons */}
          <div className="flex items-center gap-4">
            <Button
              onClick={startSimulation}
              disabled={simulationState.isRunning}
              className="flex items-center gap-2"
            >
              <Play className="h-4 w-4" />
              Start
            </Button>
            
            <Button
              onClick={pauseSimulation}
              disabled={!simulationState.isRunning}
              variant="outline"
              className="flex items-center gap-2"
            >
              <Pause className="h-4 w-4" />
              {simulationState.isPaused ? 'Resume' : 'Pause'}
            </Button>
            
            <Button
              onClick={stopSimulation}
              disabled={!simulationState.isRunning}
              variant="outline"
              className="flex items-center gap-2"
            >
              <Square className="h-4 w-4" />
              Stop
            </Button>
            
            <Button
              onClick={resetSimulation}
              variant="outline"
              className="flex items-center gap-2"
            >
              <RotateCcw className="h-4 w-4" />
              Reset
            </Button>
          </div>

          {/* Status Display */}
          <div className="grid gap-4 md:grid-cols-3">
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">Progress</span>
                <Badge variant="outline">
                  {simulationState.currentStep}/{parameters.maxSteps}
                </Badge>
              </div>
              <Progress value={progress} className="w-full" />
            </div>
            
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">Time Elapsed</span>
                <span className="text-sm text-muted-foreground">
                  {simulationState.timeElapsed.toFixed(3)}s
                </span>
              </div>
              <div className="flex items-center gap-2">
                <Clock className="h-4 w-4" />
                <span className="text-sm">
                  {performanceMetrics.stepsPerSecond.toFixed(0)} steps/s
                </span>
              </div>
            </div>
            
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">Convergence</span>
                <Badge 
                  variant="outline"
                  className={getConvergenceColor(simulationState.convergenceStatus)}
                >
                  {simulationState.convergenceStatus}
                </Badge>
              </div>
              <div className="flex items-center gap-2">
                <Target className="h-4 w-4" />
                <span className="text-sm">
                  {(performanceMetrics.convergenceRate * 100).toFixed(1)}% rate
                </span>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Configuration Tabs */}
      <Tabs defaultValue="parameters" className="w-full">
        <TabsList className="grid w-full grid-cols-5">
          <TabsTrigger value="parameters">Parameters</TabsTrigger>
          <TabsTrigger value="performance">Performance</TabsTrigger>
          <TabsTrigger value="analysis">ML Analysis</TabsTrigger>
          <TabsTrigger value="export">Export</TabsTrigger>
          <TabsTrigger value="advanced">Advanced</TabsTrigger>
        </TabsList>

        <TabsContent value="parameters" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Settings className="h-4 w-4" />
                Simulation Parameters
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="grid gap-6 md:grid-cols-2">
                <div className="space-y-2">
                  <Label>Time Step: {parameters.timeStep}</Label>
                  <Slider
                    value={[parameters.timeStep]}
                    onValueChange={([value]) => updateParameter('timeStep', value)}
                    min={0.001}
                    max={0.1}
                    step={0.001}
                    className="w-full"
                  />
                </div>
                
                <div className="space-y-2">
                  <Label>Max Steps</Label>
                  <Input
                    type="number"
                    value={parameters.maxSteps}
                    onChange={(e) => updateParameter('maxSteps', parseInt(e.target.value))}
                    min={100}
                    max={10000}
                  />
                </div>
                
                <div className="space-y-2">
                  <Label>Tolerance: {parameters.tolerance.toExponential(2)}</Label>
                  <Slider
                    value={[Math.log10(parameters.tolerance)]}
                    onValueChange={([value]) => updateParameter('tolerance', Math.pow(10, value))}
                    min={-12}
                    max={-3}
                    step={0.1}
                    className="w-full"
                  />
                </div>
                
                <div className="space-y-2">
                  <Label>Recording Interval</Label>
                  <Input
                    type="number"
                    value={parameters.recordingInterval}
                    onChange={(e) => updateParameter('recordingInterval', parseInt(e.target.value))}
                    min={1}
                    max={100}
                  />
                </div>
              </div>
              
              <div className="grid gap-4 md:grid-cols-2">
                <div className="flex items-center justify-between">
                  <Label>Adaptive Time Step</Label>
                  <Switch
                    checked={parameters.adaptiveStep}
                    onCheckedChange={(checked) => updateParameter('adaptiveStep', checked)}
                  />
                </div>
                
                <div className="flex items-center justify-between">
                  <Label>Auto Stop on Convergence</Label>
                  <Switch
                    checked={parameters.autoStop}
                    onCheckedChange={(checked) => updateParameter('autoStop', checked)}
                  />
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="performance" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Cpu className="h-4 w-4" />
                Performance Metrics
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid gap-4 md:grid-cols-2">
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="font-medium">Steps per Second</span>
                    <Badge variant="outline">
                      {performanceMetrics.stepsPerSecond.toFixed(0)}
                    </Badge>
                  </div>
                  <Progress value={Math.min(performanceMetrics.stepsPerSecond, 100)} />
                </div>
                
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="font-medium">Memory Usage</span>
                    <Badge variant="outline">
                      {performanceMetrics.memoryUsage.toFixed(1)}%
                    </Badge>
                  </div>
                  <Progress value={performanceMetrics.memoryUsage} />
                </div>
                
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="font-medium">CPU Usage</span>
                    <Badge variant="outline">
                      {performanceMetrics.cpuUsage.toFixed(1)}%
                    </Badge>
                  </div>
                  <Progress value={performanceMetrics.cpuUsage} />
                </div>
                
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="font-medium">Convergence Rate</span>
                    <Badge variant="outline">
                      {(performanceMetrics.convergenceRate * 100).toFixed(1)}%
                    </Badge>
                  </div>
                  <Progress value={performanceMetrics.convergenceRate * 100} />
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="analysis">
          <MLAnalysis
            data={simulationData.results}
            parameters={{
              timeStep: parameters.timeStep,
              maxSteps: parameters.maxSteps,
              tolerance: parameters.tolerance,
              recordingInterval: parameters.recordingInterval
            }}
            simulationType={simulationType}
            onAnalysisComplete={(results) => {
              setSimulationData(prev => ({ ...prev, analysis: results }));
            }}
          />
        </TabsContent>

        <TabsContent value="export">
          <SimulationExportSystem
            simulationData={simulationData}
            simulationType={simulationType}
            onExport={(format, data) => {
              console.log(`Exported as ${format}:`, data);
            }}
          />
        </TabsContent>

        <TabsContent value="advanced" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Advanced Settings</CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="flex items-center justify-between">
                <div>
                  <Label className="font-medium">Parallel Processing</Label>
                  <p className="text-sm text-muted-foreground">
                    Enable multi-threaded computation
                  </p>
                </div>
                <Switch
                  checked={parameters.parallelProcessing}
                  onCheckedChange={(checked) => updateParameter('parallelProcessing', checked)}
                />
              </div>
              
              <div className="space-y-2">
                <Label>Numerical Precision</Label>
                <div className="grid grid-cols-3 gap-2">
                  {['single', 'double', 'extended'].map((precision) => (
                    <Button
                      key={precision}
                      variant={parameters.precision === precision ? 'default' : 'outline'}
                      onClick={() => updateParameter('precision', precision as any)}
                      className="capitalize"
                    >
                      {precision}
                    </Button>
                  ))}
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}

export default AdvancedSimulationFeatures;