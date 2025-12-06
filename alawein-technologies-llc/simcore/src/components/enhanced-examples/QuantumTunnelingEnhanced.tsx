/**
 * Enhanced Quantum Tunneling Simulation
 * Advanced quantum mechanics with interactive controls
 */

import React, { useState, useEffect, useCallback } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Slider } from '@/components/ui/slider';
import { Switch } from '@/components/ui/switch';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Label } from '@/components/ui/label';
import { Progress } from '@/components/ui/progress';
import { ScientificPlot } from '@/components/ScientificPlot';
import { MLAnalysis } from '@/components/MLAnalysis';
import { SimulationExportSystem } from '@/components/SimulationExportSystem';
import { 
  Play, 
  Pause, 
  RotateCcw, 
  Zap, 
  Atom, 
  BarChart3,
  Settings,
  Download,
  Brain
} from 'lucide-react';

interface TunnelingParams {
  barrierHeight: number;
  barrierWidth: number;
  particleEnergy: number;
  particleMass: number;
  wavePacketWidth: number;
  animationSpeed: number;
}

interface QuantumState {
  wavefunction: Float32Array;
  probability: Float32Array;
  position: Float32Array;
  potential: Float32Array;
  time: number;
  transmissionCoeff: number;
  reflectionCoeff: number;
}

export function QuantumTunnelingEnhanced() {
  const [params, setParams] = useState<TunnelingParams>({
    barrierHeight: 2.0,
    barrierWidth: 1.0,
    particleEnergy: 1.5,
    particleMass: 1.0,
    wavePacketWidth: 0.3,
    animationSpeed: 1.0
  });

  const [state, setState] = useState<QuantumState>({
    wavefunction: new Float32Array(512),
    probability: new Float32Array(512),
    position: new Float32Array(512),
    potential: new Float32Array(512),
    time: 0,
    transmissionCoeff: 0,
    reflectionCoeff: 0
  });

  const [isRunning, setIsRunning] = useState(false);
  const [showReal, setShowReal] = useState(true);
  const [showImaginary, setShowImaginary] = useState(true);
  const [showProbability, setShowProbability] = useState(true);
  const [tunnelAnalysis, setTunnelAnalysis] = useState<any[]>([]);

  // Initialize quantum system
  const initializeSystem = useCallback(() => {
    const N = 512;
    const dx = 20.0 / N;
    const position = new Float32Array(N);
    const potential = new Float32Array(N);
    const wavefunction = new Float32Array(N * 2); // Complex: [real, imag, real, imag, ...]
    const probability = new Float32Array(N);

    // Create position array
    for (let i = 0; i < N; i++) {
      position[i] = (i - N/2) * dx;
    }

    // Create potential barrier
    const barrierStart = -params.barrierWidth / 2;
    const barrierEnd = params.barrierWidth / 2;
    
    for (let i = 0; i < N; i++) {
      const x = position[i];
      if (x >= barrierStart && x <= barrierEnd) {
        potential[i] = params.barrierHeight;
      } else {
        potential[i] = 0;
      }
    }

    // Initialize Gaussian wave packet
    const x0 = -5; // Initial position
    const k0 = Math.sqrt(2 * params.particleMass * params.particleEnergy); // Initial momentum
    const sigma = params.wavePacketWidth;

    for (let i = 0; i < N; i++) {
      const x = position[i];
      const gaussian = Math.exp(-((x - x0) ** 2) / (2 * sigma ** 2));
      const phase = k0 * x;
      
      wavefunction[i * 2] = gaussian * Math.cos(phase); // Real part
      wavefunction[i * 2 + 1] = gaussian * Math.sin(phase); // Imaginary part
      
      probability[i] = gaussian ** 2;
    }

    // Normalize wavefunction
    let norm = 0;
    for (let i = 0; i < N; i++) {
      norm += probability[i] * dx;
    }
    const normFactor = 1 / Math.sqrt(norm);
    
    for (let i = 0; i < N; i++) {
      wavefunction[i * 2] *= normFactor;
      wavefunction[i * 2 + 1] *= normFactor;
      probability[i] = wavefunction[i * 2] ** 2 + wavefunction[i * 2 + 1] ** 2;
    }

    setState(prev => ({
      ...prev,
      wavefunction,
      probability,
      position,
      potential,
      time: 0
    }));
  }, [params]);

  // Time evolution using split-operator method
  const evolveWavefunction = useCallback(() => {
    const N = 512;
    const dx = 20.0 / N;
    const dt = 0.01 * params.animationSpeed;
    const hbar = 1; // Natural units

    // Create a copy of the current wavefunction
    const psi = new Float32Array(state.wavefunction);
    
    // Apply potential operator (multiplication in position space)
    for (let i = 0; i < N; i++) {
      const potentialFactor = Math.exp(-1 * state.potential[i] * dt / (2 * hbar));
      const real = psi[i * 2];
      const imag = psi[i * 2 + 1];
      
      // Complex multiplication with exp(-i * V * dt / 2ℏ)
      const cosV = Math.cos(state.potential[i] * dt / (2 * hbar));
      const sinV = -Math.sin(state.potential[i] * dt / (2 * hbar));
      
      psi[i * 2] = real * cosV - imag * sinV;
      psi[i * 2 + 1] = real * sinV + imag * cosV;
    }

    // Apply kinetic operator (would require FFT for exact implementation)
    // Simplified finite difference approximation
    const kineticFactor = -hbar * dt / (2 * params.particleMass * dx * dx);
    const newPsi = new Float32Array(psi.length);
    
    for (let i = 1; i < N - 1; i++) {
      const realPrev = psi[(i - 1) * 2];
      const imagPrev = psi[(i - 1) * 2 + 1];
      const realCurr = psi[i * 2];
      const imagCurr = psi[i * 2 + 1];
      const realNext = psi[(i + 1) * 2];
      const imagNext = psi[(i + 1) * 2 + 1];
      
      // Second derivative approximation
      const d2Real = realPrev - 2 * realCurr + realNext;
      const d2Imag = imagPrev - 2 * imagCurr + imagNext;
      
      // Apply kinetic energy operator: i * ℏ * ∇² / (2m)
      newPsi[i * 2] = realCurr + kineticFactor * d2Imag;
      newPsi[i * 2 + 1] = imagCurr - kineticFactor * d2Real;
    }

    // Handle boundaries
    newPsi[0] = 0;
    newPsi[1] = 0;
    newPsi[(N - 1) * 2] = 0;
    newPsi[(N - 1) * 2 + 1] = 0;

    // Calculate new probability density
    const newProbability = new Float32Array(N);
    for (let i = 0; i < N; i++) {
      newProbability[i] = newPsi[i * 2] ** 2 + newPsi[i * 2 + 1] ** 2;
    }

    // Calculate transmission and reflection coefficients
    const leftRegion = Math.floor(N * 0.2);
    const rightRegion = Math.floor(N * 0.8);
    
    let leftProb = 0, rightProb = 0;
    for (let i = 0; i < leftRegion; i++) {
      leftProb += newProbability[i];
    }
    for (let i = rightRegion; i < N; i++) {
      rightProb += newProbability[i];
    }
    
    const totalProb = leftProb + rightProb;
    const reflectionCoeff = totalProb > 0 ? leftProb / totalProb : 0;
    const transmissionCoeff = totalProb > 0 ? rightProb / totalProb : 0;

    setState(prev => ({
      ...prev,
      wavefunction: newPsi,
      probability: newProbability,
      time: prev.time + dt,
      transmissionCoeff,
      reflectionCoeff
    }));
  }, [state, params]);

  // Animation loop
  useEffect(() => {
    let animationFrame: number;
    
    if (isRunning) {
      const animate = () => {
        evolveWavefunction();
        animationFrame = requestAnimationFrame(animate);
      };
      animationFrame = requestAnimationFrame(animate);
    }
    
    return () => {
      if (animationFrame) {
        cancelAnimationFrame(animationFrame);
      }
    };
  }, [isRunning, evolveWavefunction]);

  // Initialize system when parameters change
  useEffect(() => {
    initializeSystem();
  }, [initializeSystem]);

  // Calculate theoretical transmission coefficient
  const theoreticalTransmission = useCallback(() => {
    const k = Math.sqrt(2 * params.particleMass * params.particleEnergy);
    const kappa = Math.sqrt(2 * params.particleMass * (params.barrierHeight - params.particleEnergy));
    const a = params.barrierWidth;
    
    if (params.particleEnergy >= params.barrierHeight) {
      return 1; // Classical case
    }
    
    const transmission = 1 / (1 + (params.barrierHeight ** 2 * Math.sinh(kappa * a) ** 2) / (4 * params.particleEnergy * (params.barrierHeight - params.particleEnergy)));
    return Math.min(1, Math.max(0, transmission));
  }, [params]);

  const updateParam = <K extends keyof TunnelingParams>(key: K, value: TunnelingParams[K]) => {
    setParams(prev => ({ ...prev, [key]: value }));
  };

  // Prepare plot data
  const wavefunctionPlotData = {
    x: Array.from(state.position),
    datasets: [
      ...(showReal ? [{
        y: Array.from(state.wavefunction).filter((_, i) => i % 2 === 0),
        label: 'Real Part',
        color: '#3b82f6'
      }] : []),
      ...(showImaginary ? [{
        y: Array.from(state.wavefunction).filter((_, i) => i % 2 === 1),
        label: 'Imaginary Part', 
        color: '#ef4444'
      }] : []),
      ...(showProbability ? [{
        y: Array.from(state.probability),
        label: 'Probability Density',
        color: '#10b981'
      }] : []),
      {
        y: Array.from(state.potential).map(v => v * 0.1),
        label: 'Potential (scaled)',
        color: '#8b5cf6'
      }
    ]
  };

  return (
    <div className="w-full space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Atom className="h-5 w-5" />
            Enhanced Quantum Tunneling Simulation
          </CardTitle>
          <CardDescription>
            Interactive quantum tunneling with real-time wavefunction evolution
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-6">
            {/* Control Panel */}
            <div className="flex items-center gap-4 p-4 bg-muted rounded-lg">
              <Button
                onClick={() => setIsRunning(!isRunning)}
                className="flex items-center gap-2"
              >
                {isRunning ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
                {isRunning ? 'Pause' : 'Start'}
              </Button>
              
              <Button
                onClick={initializeSystem}
                variant="outline"
                className="flex items-center gap-2"
              >
                <RotateCcw className="h-4 w-4" />
                Reset
              </Button>
              
              <div className="flex items-center gap-2">
                <Badge variant="outline">
                  Time: {state.time.toFixed(2)}
                </Badge>
                <Badge variant="outline" className="text-green-600">
                  T: {(state.transmissionCoeff * 100).toFixed(1)}%
                </Badge>
                <Badge variant="outline" className="text-red-600">
                  R: {(state.reflectionCoeff * 100).toFixed(1)}%
                </Badge>
              </div>
            </div>

            {/* Main Visualization */}
            <div className="h-96 border rounded-lg">
              <ScientificPlot
                title="Quantum Tunneling Wavefunction"
                data={wavefunctionPlotData}
                plotType="2d"
                xLabel="Position"
                yLabel="Amplitude"
                showGrid={true}
                showLegend={true}
              />
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Parameter Controls */}
      <Tabs defaultValue="parameters" className="w-full">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="parameters">Parameters</TabsTrigger>
          <TabsTrigger value="visualization">Visualization</TabsTrigger>
          <TabsTrigger value="analysis">ML Analysis</TabsTrigger>
          <TabsTrigger value="export">Export</TabsTrigger>
        </TabsList>

        <TabsContent value="parameters" className="space-y-4">
          <div className="grid gap-4 md:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Barrier Properties</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <Label>Barrier Height: {params.barrierHeight.toFixed(2)} eV</Label>
                  <Slider
                    value={[params.barrierHeight]}
                    onValueChange={([value]) => updateParam('barrierHeight', value)}
                    min={0.5}
                    max={5.0}
                    step={0.1}
                  />
                </div>
                
                <div className="space-y-2">
                  <Label>Barrier Width: {params.barrierWidth.toFixed(2)} nm</Label>
                  <Slider
                    value={[params.barrierWidth]}
                    onValueChange={([value]) => updateParam('barrierWidth', value)}
                    min={0.1}
                    max={3.0}
                    step={0.1}
                  />
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Particle Properties</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <Label>Particle Energy: {params.particleEnergy.toFixed(2)} eV</Label>
                  <Slider
                    value={[params.particleEnergy]}
                    onValueChange={([value]) => updateParam('particleEnergy', value)}
                    min={0.1}
                    max={4.0}
                    step={0.1}
                  />
                </div>
                
                <div className="space-y-2">
                  <Label>Wave Packet Width: {params.wavePacketWidth.toFixed(2)}</Label>
                  <Slider
                    value={[params.wavePacketWidth]}
                    onValueChange={([value]) => updateParam('wavePacketWidth', value)}
                    min={0.1}
                    max={1.0}
                    step={0.05}
                  />
                </div>
              </CardContent>
            </Card>
          </div>
          
          <Card>
            <CardHeader>
              <CardTitle className="text-lg">Theoretical Analysis</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid gap-4 md:grid-cols-3">
                <div className="text-center p-4 bg-muted rounded-lg">
                  <div className="text-2xl font-bold text-primary">
                    {(theoreticalTransmission() * 100).toFixed(1)}%
                  </div>
                  <div className="text-sm text-muted-foreground">
                    Theoretical Transmission
                  </div>
                </div>
                
                <div className="text-center p-4 bg-muted rounded-lg">
                  <div className="text-2xl font-bold text-green-600">
                    {(state.transmissionCoeff * 100).toFixed(1)}%
                  </div>
                  <div className="text-sm text-muted-foreground">
                    Simulated Transmission
                  </div>
                </div>
                
                <div className="text-center p-4 bg-muted rounded-lg">
                  <div className="text-2xl font-bold text-blue-600">
                    {Math.abs(theoreticalTransmission() - state.transmissionCoeff).toFixed(3)}
                  </div>
                  <div className="text-sm text-muted-foreground">
                    Difference
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="visualization" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <BarChart3 className="h-4 w-4" />
                Visualization Controls
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid gap-4 md:grid-cols-2">
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <Label>Show Real Part</Label>
                    <Switch
                      checked={showReal}
                      onCheckedChange={setShowReal}
                    />
                  </div>
                  
                  <div className="flex items-center justify-between">
                    <Label>Show Imaginary Part</Label>
                    <Switch
                      checked={showImaginary}
                      onCheckedChange={setShowImaginary}
                    />
                  </div>
                  
                  <div className="flex items-center justify-between">
                    <Label>Show Probability Density</Label>
                    <Switch
                      checked={showProbability}
                      onCheckedChange={setShowProbability}
                    />
                  </div>
                </div>
                
                <div className="space-y-2">
                  <Label>Animation Speed: {params.animationSpeed.toFixed(1)}x</Label>
                  <Slider
                    value={[params.animationSpeed]}
                    onValueChange={([value]) => updateParam('animationSpeed', value)}
                    min={0.1}
                    max={3.0}
                    step={0.1}
                  />
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="analysis">
          <MLAnalysis
            data={[Array.from(state.probability)]}
            parameters={{
              barrierHeight: params.barrierHeight,
              barrierWidth: params.barrierWidth,
              particleEnergy: params.particleEnergy,
              transmissionCoeff: state.transmissionCoeff
            }}
            simulationType="quantum_tunneling"
          />
        </TabsContent>

        <TabsContent value="export">
          <SimulationExportSystem
            simulationData={{
              parameters: params,
              results: [
                Array.from(state.wavefunction),
                Array.from(state.probability),
                Array.from(state.position),
                Array.from(state.potential)
              ],
              metadata: {
                created: new Date().toISOString(),
                type: 'quantum_tunneling',
                version: '1.0'
              },
              analysis: tunnelAnalysis
            }}
            simulationType="quantum_tunneling"
          />
        </TabsContent>
      </Tabs>
    </div>
  );
}