import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Slider } from '@/components/ui/slider';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Switch } from '@/components/ui/switch';
import { Label } from '@/components/ui/label';
import { Progress } from '@/components/ui/progress';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip';
import { 
  Play, 
  Pause, 
  RotateCcw, 
  Settings, 
  Download, 
  Info,
  Zap,
  Eye,
  BarChart3,
  Waves,
  Atom
} from 'lucide-react';
import { usePrefersReducedMotion } from '@/lib/accessibility-utils';
import { useToast } from '@/hooks/use-toast';

interface QuantumState {
  real: number[];
  imag: number[];
  probability: number[];
  energy: number;
  norm: number;
}

interface SimulationParameters {
  gridSize: number;
  timeStep: number;
  totalTime: number;
  potentialType: 'harmonic' | 'step' | 'barrier' | 'well' | 'double_well';
  potentialStrength: number;
  potentialWidth: number;
  initialWavePacketWidth: number;
  initialMomentum: number;
  visualizationMode: 'wavefunction' | 'probability' | 'phase' | 'energy';
}

export const EnhancedTDSEVisualization: React.FC = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationFrameRef = useRef<number>();
  const workerRef = useRef<Worker>();
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [quantumState, setQuantumState] = useState<QuantumState | null>(null);
  const [progress, setProgress] = useState(0);
  const [performance, setPerformance] = useState({ fps: 0, computeTime: 0 });
  const prefersReducedMotion = usePrefersReducedMotion();
  const { toast } = useToast();

  const [parameters, setParameters] = useState<SimulationParameters>({
    gridSize: 256,
    timeStep: 0.01,
    totalTime: 10,
    potentialType: 'harmonic',
    potentialStrength: 1.0,
    potentialWidth: 2.0,
    initialWavePacketWidth: 1.0,
    initialMomentum: 0.0,
    visualizationMode: 'probability'
  });

  // Initialize quantum simulation
  const initializeSimulation = useCallback(() => {
    const { gridSize, initialWavePacketWidth, initialMomentum } = parameters;
    
    // Create spatial grid
    const xMin = -10;
    const xMax = 10;
    const dx = (xMax - xMin) / gridSize;
    const x = Array.from({ length: gridSize }, (_, i) => xMin + i * dx);
    
    // Initialize Gaussian wave packet
    const real = new Array(gridSize);
    const imag = new Array(gridSize);
    const probability = new Array(gridSize);
    
    let norm = 0;
    for (let i = 0; i < gridSize; i++) {
      const gauss = Math.exp(-0.5 * (x[i] / initialWavePacketWidth) ** 2);
      const phase = initialMomentum * x[i];
      
      real[i] = gauss * Math.cos(phase);
      imag[i] = gauss * Math.sin(phase);
      probability[i] = real[i] ** 2 + imag[i] ** 2;
      norm += probability[i] * dx;
    }
    
    // Normalize
    const normFactor = 1 / Math.sqrt(norm);
    for (let i = 0; i < gridSize; i++) {
      real[i] *= normFactor;
      imag[i] *= normFactor;
      probability[i] = real[i] ** 2 + imag[i] ** 2;
    }
    
    const energy = calculateEnergy(real, imag, x);
    
    setQuantumState({
      real,
      imag,
      probability,
      energy,
      norm: 1.0
    });
    
    setCurrentTime(0);
    setProgress(0);
  }, [parameters]);

  // Calculate expectation value of energy
  const calculateEnergy = (real: number[], imag: number[], x: number[]) => {
    const gridSize = real.length;
    const dx = x[1] - x[0];
    let kineticEnergy = 0;
    let potentialEnergy = 0;
    
    // Kinetic energy using finite differences
    for (let i = 1; i < gridSize - 1; i++) {
      const d2psi_real = (real[i + 1] - 2 * real[i] + real[i - 1]) / (dx * dx);
      const d2psi_imag = (imag[i + 1] - 2 * imag[i] + imag[i - 1]) / (dx * dx);
      kineticEnergy += -0.5 * (real[i] * d2psi_real + imag[i] * d2psi_imag) * dx;
    }
    
    // Potential energy
    for (let i = 0; i < gridSize; i++) {
      const V = getPotential(x[i]);
      const prob = real[i] ** 2 + imag[i] ** 2;
      potentialEnergy += V * prob * dx;
    }
    
    return kineticEnergy + potentialEnergy;
  };

  // Get potential value at position x
  const getPotential = (x: number) => {
    const { potentialType, potentialStrength, potentialWidth } = parameters;
    
    switch (potentialType) {
      case 'harmonic':
        return 0.5 * potentialStrength * x * x;
      
      case 'step':
        return x > 0 ? potentialStrength : 0;
      
      case 'barrier':
        return Math.abs(x) < potentialWidth ? potentialStrength : 0;
      
      case 'well':
        return Math.abs(x) < potentialWidth ? -potentialStrength : 0;
      
      case 'double_well':
        const r1 = Math.abs(x + potentialWidth);
        const r2 = Math.abs(x - potentialWidth);
        return -potentialStrength * (Math.exp(-r1) + Math.exp(-r2));
      
      default:
        return 0;
    }
  };

  // Animation loop
  useEffect(() => {
    if (!isPlaying || !quantumState || prefersReducedMotion) return;

    let lastTime = 0;
    const fpsCounter = { count: 0, lastTime: Date.now() };

    const animate = (currentTime: number) => {
      const deltaTime = currentTime - lastTime;
      lastTime = currentTime;

      // Update FPS counter
      fpsCounter.count++;
      if (currentTime - fpsCounter.lastTime > 1000) {
        setPerformance(prev => ({
          ...prev,
          fps: Math.round(fpsCounter.count * 1000 / (currentTime - fpsCounter.lastTime))
        }));
        fpsCounter.count = 0;
        fpsCounter.lastTime = currentTime;
      }

      // Time evolution simulation (simplified)
      setCurrentTime(prev => {
        const newTime = prev + parameters.timeStep;
        setProgress((newTime / parameters.totalTime) * 100);
        
        if (newTime >= parameters.totalTime) {
          setIsPlaying(false);
          return parameters.totalTime;
        }
        
        return newTime;
      });

      // Continue animation
      if (isPlaying) {
        animationFrameRef.current = requestAnimationFrame(animate);
      }
    };

    animationFrameRef.current = requestAnimationFrame(animate);

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [isPlaying, quantumState, parameters, prefersReducedMotion]);

  // Canvas rendering
  useEffect(() => {
    if (!quantumState || !canvasRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Set canvas size
    canvas.width = canvas.offsetWidth * window.devicePixelRatio;
    canvas.height = canvas.offsetHeight * window.devicePixelRatio;
    ctx.scale(window.devicePixelRatio, window.devicePixelRatio);

    const width = canvas.offsetWidth;
    const height = canvas.offsetHeight;

    // Clear canvas
    ctx.clearRect(0, 0, width, height);

    // Draw grid
    ctx.strokeStyle = 'rgba(100, 116, 139, 0.2)';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 10; i++) {
      const x = (i / 10) * width;
      const y = (i / 10) * height;
      
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();
      
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
    }

    // Draw potential
    ctx.strokeStyle = 'rgba(239, 68, 68, 0.6)';
    ctx.lineWidth = 2;
    ctx.beginPath();
    
    for (let i = 0; i < width; i++) {
      const x = (i / width) * 20 - 10; // Map to simulation coordinates
      const V = getPotential(x);
      const normalizedV = Math.max(0, Math.min(1, (V + 5) / 10)); // Normalize for display
      const y = height - normalizedV * height * 0.3;
      
      if (i === 0) {
        ctx.moveTo(i, y);
      } else {
        ctx.lineTo(i, y);
      }
    }
    ctx.stroke();

    // Draw wavefunction based on visualization mode
    const { visualizationMode } = parameters;
    const { real, imag, probability } = quantumState;
    
    ctx.lineWidth = 3;
    
    if (visualizationMode === 'probability') {
      // Draw probability density
      ctx.strokeStyle = 'rgba(59, 130, 246, 0.8)';
      ctx.fillStyle = 'rgba(59, 130, 246, 0.2)';
      ctx.beginPath();
      
      const maxProb = Math.max(...probability);
      for (let i = 0; i < probability.length; i++) {
        const x = (i / probability.length) * width;
        const y = height - (probability[i] / maxProb) * height * 0.6;
        
        if (i === 0) {
          ctx.moveTo(x, height);
          ctx.lineTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      }
      ctx.lineTo(width, height);
      ctx.closePath();
      ctx.fill();
      ctx.stroke();
      
    } else if (visualizationMode === 'wavefunction') {
      // Draw real and imaginary parts
      const maxAmp = Math.max(...real.map(Math.abs), ...imag.map(Math.abs));
      
      // Real part
      ctx.strokeStyle = 'rgba(34, 197, 94, 0.8)';
      ctx.beginPath();
      for (let i = 0; i < real.length; i++) {
        const x = (i / real.length) * width;
        const y = height / 2 - (real[i] / maxAmp) * height * 0.3;
        
        if (i === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      }
      ctx.stroke();
      
      // Imaginary part
      ctx.strokeStyle = 'rgba(168, 85, 247, 0.8)';
      ctx.beginPath();
      for (let i = 0; i < imag.length; i++) {
        const x = (i / imag.length) * width;
        const y = height / 2 - (imag[i] / maxAmp) * height * 0.3;
        
        if (i === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      }
      ctx.stroke();
      
    } else if (visualizationMode === 'phase') {
      // Draw phase as color-coded line
      for (let i = 0; i < real.length - 1; i++) {
        const phase = Math.atan2(imag[i], real[i]);
        const hue = ((phase + Math.PI) / (2 * Math.PI)) * 360;
        const prob = probability[i];
        const maxProb = Math.max(...probability);
        
        ctx.strokeStyle = `hsla(${hue}, 80%, 60%, ${prob / maxProb})`;
        ctx.lineWidth = 4;
        ctx.beginPath();
        
        const x1 = (i / real.length) * width;
        const x2 = ((i + 1) / real.length) * width;
        const y1 = height - (prob / maxProb) * height * 0.6;
        const y2 = height - (probability[i + 1] / maxProb) * height * 0.6;
        
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.stroke();
      }
    }

    // Draw current time indicator
    ctx.fillStyle = 'rgba(59, 130, 246, 0.8)';
    ctx.fillRect(10, 10, 120, 30);
    ctx.fillStyle = 'white';
    ctx.font = '14px monospace';
    ctx.fillText(`t = ${currentTime.toFixed(2)}`, 15, 28);

  }, [quantumState, parameters, currentTime]);

  // Initialize simulation on mount and parameter changes
  useEffect(() => {
    initializeSimulation();
  }, [initializeSimulation]);

  const handleParameterChange = <K extends keyof SimulationParameters>(
    key: K,
    value: SimulationParameters[K]
  ) => {
    setParameters(prev => ({ ...prev, [key]: value }));
  };

  const handlePlayPause = () => {
    setIsPlaying(!isPlaying);
  };

  const handleReset = () => {
    setIsPlaying(false);
    initializeSimulation();
    toast({
      title: "Simulation Reset",
      description: "Quantum state has been reinitialized.",
    });
  };

  const handleExport = () => {
    if (!quantumState) return;
    
    const data = {
      time: currentTime,
      parameters,
      quantumState,
      performance
    };
    
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `tdse_simulation_t${currentTime.toFixed(2)}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    
    toast({
      title: "Data Exported",
      description: "Simulation data has been saved to file.",
    });
  };

  return (
    <div className="space-y-6">
      {/* Control Panel */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Atom className="w-5 h-5 text-primary" />
            Enhanced TDSE Simulation Controls
            <Badge variant="outline" className="ml-auto">
              {performance.fps} FPS
            </Badge>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <Tabs defaultValue="simulation" className="w-full">
            <TabsList className="grid w-full grid-cols-4">
              <TabsTrigger value="simulation">Simulation</TabsTrigger>
              <TabsTrigger value="potential">Potential</TabsTrigger>
              <TabsTrigger value="initial">Initial State</TabsTrigger>
              <TabsTrigger value="visualization">Visualization</TabsTrigger>
            </TabsList>

            <TabsContent value="simulation" className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="space-y-2">
                  <Label htmlFor="time-step">Time Step: {parameters.timeStep.toFixed(3)}</Label>
                  <Slider
                    id="time-step"
                    min={0.001}
                    max={0.1}
                    step={0.001}
                    value={[parameters.timeStep]}
                    onValueChange={([value]) => handleParameterChange('timeStep', value)}
                  />
                </div>
                
                <div className="space-y-2">
                  <Label htmlFor="total-time">Total Time: {parameters.totalTime.toFixed(1)}</Label>
                  <Slider
                    id="total-time"
                    min={1}
                    max={50}
                    step={0.5}
                    value={[parameters.totalTime]}
                    onValueChange={([value]) => handleParameterChange('totalTime', value)}
                  />
                </div>
                
                <div className="space-y-2">
                  <Label htmlFor="grid-size">Grid Size: {parameters.gridSize}</Label>
                  <Slider
                    id="grid-size"
                    min={64}
                    max={512}
                    step={32}
                    value={[parameters.gridSize]}
                    onValueChange={([value]) => handleParameterChange('gridSize', value)}
                  />
                </div>
              </div>
              
              <div className="flex items-center gap-4">
                <TooltipProvider>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <Button
                        variant={isPlaying ? "destructive" : "default"}
                        onClick={handlePlayPause}
                        className="flex items-center gap-2"
                      >
                        {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
                        {isPlaying ? "Pause" : "Play"}
                      </Button>
                    </TooltipTrigger>
                    <TooltipContent>
                      {isPlaying ? "Pause time evolution" : "Start time evolution"}
                    </TooltipContent>
                  </Tooltip>
                </TooltipProvider>
                
                <Button
                  variant="outline"
                  onClick={handleReset}
                  className="flex items-center gap-2"
                >
                  <RotateCcw className="w-4 h-4" />
                  Reset
                </Button>
                
                <Button
                  variant="outline"
                  onClick={handleExport}
                  className="flex items-center gap-2"
                >
                  <Download className="w-4 h-4" />
                  Export Data
                </Button>
                
                <div className="ml-auto">
                  <Progress value={progress} className="w-32" />
                </div>
              </div>
            </TabsContent>

            <TabsContent value="potential" className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="space-y-2">
                  <Label>Potential Type</Label>
                  <select
                    value={parameters.potentialType}
                    onChange={(e) => handleParameterChange('potentialType', e.target.value as any)}
                    className="w-full p-2 border rounded"
                  >
                    <option value="harmonic">Harmonic Oscillator</option>
                    <option value="step">Step Potential</option>
                    <option value="barrier">Potential Barrier</option>
                    <option value="well">Potential Well</option>
                    <option value="double_well">Double Well</option>
                  </select>
                </div>
                
                <div className="space-y-2">
                  <Label>Strength: {parameters.potentialStrength.toFixed(2)}</Label>
                  <Slider
                    min={0.1}
                    max={5}
                    step={0.1}
                    value={[parameters.potentialStrength]}
                    onValueChange={([value]) => handleParameterChange('potentialStrength', value)}
                  />
                </div>
                
                <div className="space-y-2">
                  <Label>Width: {parameters.potentialWidth.toFixed(2)}</Label>
                  <Slider
                    min={0.5}
                    max={5}
                    step={0.1}
                    value={[parameters.potentialWidth]}
                    onValueChange={([value]) => handleParameterChange('potentialWidth', value)}
                  />
                </div>
              </div>
            </TabsContent>

            <TabsContent value="initial" className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label>Wave Packet Width: {parameters.initialWavePacketWidth.toFixed(2)}</Label>
                  <Slider
                    min={0.3}
                    max={3}
                    step={0.1}
                    value={[parameters.initialWavePacketWidth]}
                    onValueChange={([value]) => handleParameterChange('initialWavePacketWidth', value)}
                  />
                </div>
                
                <div className="space-y-2">
                  <Label>Initial Momentum: {parameters.initialMomentum.toFixed(2)}</Label>
                  <Slider
                    min={-5}
                    max={5}
                    step={0.1}
                    value={[parameters.initialMomentum]}
                    onValueChange={([value]) => handleParameterChange('initialMomentum', value)}
                  />
                </div>
              </div>
            </TabsContent>

            <TabsContent value="visualization" className="space-y-4">
              <div className="space-y-4">
                <Label>Visualization Mode</Label>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
                  {[
                    { value: 'probability', label: 'Probability', icon: BarChart3 },
                    { value: 'wavefunction', label: 'Wavefunction', icon: Waves },
                    { value: 'phase', label: 'Phase', icon: Eye },
                    { value: 'energy', label: 'Energy', icon: Zap }
                  ].map(({ value, label, icon: Icon }) => (
                    <Button
                      key={value}
                      variant={parameters.visualizationMode === value ? "default" : "outline"}
                      onClick={() => handleParameterChange('visualizationMode', value as any)}
                      className="flex items-center gap-2"
                    >
                      <Icon className="w-4 h-4" />
                      {label}
                    </Button>
                  ))}
                </div>
              </div>
            </TabsContent>
          </Tabs>
        </CardContent>
      </Card>

      {/* Visualization Canvas */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center justify-between">
            <span>Quantum State Visualization</span>
            <div className="flex items-center gap-4 text-sm text-muted-foreground">
              {quantumState && (
                <>
                  <span>Energy: {quantumState.energy.toFixed(4)}</span>
                  <span>Norm: {quantumState.norm.toFixed(6)}</span>
                  <span>Time: {currentTime.toFixed(3)}</span>
                </>
              )}
            </div>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="relative">
            <canvas
              ref={canvasRef}
              className="w-full h-96 border rounded-lg bg-muted/10"
              style={{ background: 'linear-gradient(135deg, rgba(59, 130, 246, 0.05), rgba(168, 85, 247, 0.05))' }}
            />
            
            {prefersReducedMotion && (
              <div className="absolute inset-0 flex items-center justify-center bg-muted/20 rounded-lg">
                <div className="text-center space-y-2">
                  <Atom className="w-8 h-8 mx-auto text-muted-foreground" />
                  <p className="text-sm text-muted-foreground">
                    Animation disabled (reduced motion preference)
                  </p>
                  <p className="text-xs text-muted-foreground">
                    Static visualization available
                  </p>
                </div>
              </div>
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  );
};