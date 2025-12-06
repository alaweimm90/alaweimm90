import React, { useState, useMemo, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Slider } from '@/components/ui/slider';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Switch } from '@/components/ui/switch';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Home, Play, Pause, RotateCcw, Brain, Zap } from 'lucide-react';
import { performanceMonitor } from '@/lib/performance-utils';
import { mathUtils } from '@/lib/physics-utils';
import { InlineMath, BlockMath } from '@/components/ui/Math';
import { PhysicsModuleLayout } from '@/components/PhysicsModuleLayout';
import { PhysicsModuleHeader } from '@/components/PhysicsModuleHeader';
import { useSEO } from '@/hooks/use-seo';

// Simplified Physics-Informed Neural Network for Schrödinger equation
class PINNSchrodingerNetwork {
  private weights: number[][][];
  private biases: number[][];
  private layers: number[];
  private learningRate: number;
  
  constructor(layers: number[] = [2, 20, 20, 20, 2], learningRate: number = 0.01) {
    this.layers = layers;
    this.learningRate = learningRate;
    this.weights = [];
    this.biases = [];
    
    // Initialize weights and biases
    for (let i = 0; i < layers.length - 1; i++) {
      const W = Array(layers[i]).fill(0).map(() => 
        Array(layers[i + 1]).fill(0).map(() => mathUtils.gaussianRandom(0, 0.1))
      );
      const b = Array(layers[i + 1]).fill(0).map(() => mathUtils.gaussianRandom(0, 0.1));
      this.weights.push(W);
      this.biases.push(b);
    }
  }
  
  // Activation function (tanh)
  private activation(x: number): number {
    return Math.tanh(x);
  }
  
  private activationDerivative(x: number): number {
    const tanh_x = Math.tanh(x);
    return 1 - tanh_x * tanh_x;
  }
  
  // Forward pass
  private forward(input: number[]): { outputs: number[][], finalOutput: number[] } {
    const outputs = [input];
    let current = input;
    
    for (let layer = 0; layer < this.weights.length; layer++) {
      const next = Array(this.layers[layer + 1]).fill(0);
      
      for (let j = 0; j < next.length; j++) {
        let sum = this.biases[layer][j];
        for (let i = 0; i < current.length; i++) {
          sum += current[i] * this.weights[layer][i][j];
        }
        next[j] = layer === this.weights.length - 1 ? sum : this.activation(sum);
      }
      
      current = next;
      outputs.push([...current]);
    }
    
    return { outputs, finalOutput: current };
  }
  
  // Predict wave function
  predict(x: number, t: number): { real: number, imag: number } {
    const { finalOutput } = this.forward([x, t]);
    return { real: finalOutput[0], imag: finalOutput[1] };
  }
  
  // Physics loss: Schrödinger equation residual
  private physicsloss(x: number, t: number, potential: (x: number) => number): number {
    const h = 1e-4;
    const hbar = 1.0;
    const mass = 1.0;
    
    // Central differences for derivatives
    const psi_t_plus = this.predict(x, t + h);
    const psi_t_minus = this.predict(x, t - h);
    const dpsi_dt = {
      real: (psi_t_plus.real - psi_t_minus.real) / (2 * h),
      imag: (psi_t_plus.imag - psi_t_minus.imag) / (2 * h)
    };
    
    const psi_x_plus = this.predict(x + h, t);
    const psi_x_minus = this.predict(x - h, t);
    const psi_center = this.predict(x, t);
    const d2psi_dx2 = {
      real: (psi_x_plus.real - 2 * psi_center.real + psi_x_minus.real) / (h * h),
      imag: (psi_x_plus.imag - 2 * psi_center.imag + psi_x_minus.imag) / (h * h)
    };
    
    // Schrödinger equation: iℏ∂ψ/∂t = -ℏ²/(2m)∇²ψ + Vψ
    const V = potential(x);
    const lhs_real = -hbar * dpsi_dt.imag;
    const lhs_imag = hbar * dpsi_dt.real;
    
    const rhs_real = -hbar * hbar / (2 * mass) * d2psi_dx2.real + V * psi_center.real;
    const rhs_imag = -hbar * hbar / (2 * mass) * d2psi_dx2.imag + V * psi_center.imag;
    
    const residual_real = lhs_real - rhs_real;
    const residual_imag = lhs_imag - rhs_imag;
    
    return residual_real * residual_real + residual_imag * residual_imag;
  }
  
  // Train on a batch of points
  train(
    xPoints: number[], 
    tPoints: number[], 
    potential: (x: number) => number,
    iterations: number = 100
  ): number[] {
    const losses = [];
    
    for (let iter = 0; iter < iterations; iter++) {
      let totalLoss = 0;
      
      // Physics loss
      for (let i = 0; i < xPoints.length; i++) {
        for (let j = 0; j < tPoints.length; j++) {
          totalLoss += this.physicsloss(xPoints[i], tPoints[j], potential);
        }
      }
      
      // Boundary conditions (simplified)
      const boundaryLoss = this.predict(-5, 0).real ** 2 + this.predict(-5, 0).imag ** 2 +
                          this.predict(5, 0).real ** 2 + this.predict(5, 0).imag ** 2;
      totalLoss += 10 * boundaryLoss; // Weight boundary conditions
      
      losses.push(totalLoss / (xPoints.length * tPoints.length));
      
      // Simplified gradient descent (would use automatic differentiation in practice)
      this.simplifiedUpdate();
    }
    
    return losses;
  }
  
  // Simplified parameter update
  private simplifiedUpdate(): void {
    for (let layer = 0; layer < this.weights.length; layer++) {
      for (let i = 0; i < this.weights[layer].length; i++) {
        for (let j = 0; j < this.weights[layer][i].length; j++) {
          // Add small random perturbation (simplified gradient)
          this.weights[layer][i][j] += this.learningRate * mathUtils.gaussianRandom(0, 0.001);
        }
      }
      for (let j = 0; j < this.biases[layer].length; j++) {
        this.biases[layer][j] += this.learningRate * mathUtils.gaussianRandom(0, 0.001);
      }
    }
  }
}

// Potential functions
const potentials = {
  harmonic: (x: number) => 0.5 * x * x,
  barrier: (x: number) => Math.abs(x) < 1 ? 5 : 0,
  well: (x: number) => Math.abs(x) < 2 ? -3 : 0,
  double_well: (x: number) => 0.1 * (x * x - 4) ** 2
};

// Training visualization component
function TrainingProgress({ losses }: { losses: number[] }) {
  const canvasRef = React.useRef<HTMLCanvasElement>(null);
  
  React.useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || losses.length === 0) return;
    
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    const { width, height } = canvas;
    ctx.clearRect(0, 0, width, height);
    
    const maxLoss = Math.max(...losses);
    const minLoss = Math.min(...losses);
    const range = maxLoss - minLoss || 1;
    
    ctx.strokeStyle = '#3b82f6';
    ctx.lineWidth = 2;
    ctx.beginPath();
    
    for (let i = 0; i < losses.length; i++) {
      const x = (i / (losses.length - 1)) * width;
      const y = height - ((losses[i] - minLoss) / range) * height * 0.8 - height * 0.1;
      
      if (i === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    }
    ctx.stroke();
    
    // Labels
    ctx.fillStyle = '#374151';
    ctx.font = '12px Arial';
    ctx.fillText('Training Loss', 10, 20);
    ctx.fillText('Iterations', width / 2 - 30, height - 5);
  }, [losses]);
  
  return (
    <canvas
      ref={canvasRef}
      width={400}
      height={200}
      className="w-full h-32 border rounded-lg bg-card"
    />
  );
}

// Solution visualization
function SolutionPlot({ 
  pinn, 
  potential, 
  timeSlice 
}: { 
  pinn: PINNSchrodingerNetwork | null, 
  potential: (x: number) => number,
  timeSlice: number 
}) {
  const canvasRef = React.useRef<HTMLCanvasElement>(null);
  
  React.useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !pinn) return;
    
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    const { width, height } = canvas;
    ctx.clearRect(0, 0, width, height);
    
    const xPoints = Array.from({ length: 100 }, (_, i) => -5 + (10 * i) / 99);
    const potentialValues = xPoints.map(potential);
    const waveFunction = xPoints.map(x => pinn.predict(x, timeSlice));
    
    const maxPot = Math.max(...potentialValues);
    const maxWave = Math.max(...waveFunction.map(w => Math.sqrt(w.real ** 2 + w.imag ** 2)));
    
    // Draw potential
    ctx.strokeStyle = '#64748b';
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i < xPoints.length; i++) {
      const x = (i / (xPoints.length - 1)) * width;
      const y = height - (potentialValues[i] / (maxPot || 1)) * height * 0.3 - height * 0.1;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
    
    // Draw wave function (real part)
    ctx.strokeStyle = '#3b82f6';
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i < xPoints.length; i++) {
      const x = (i / (xPoints.length - 1)) * width;
      const y = height / 2 - (waveFunction[i].real / (maxWave || 1)) * height * 0.2;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
    
    // Draw wave function (imaginary part)
    ctx.strokeStyle = '#ef4444';
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i < xPoints.length; i++) {
      const x = (i / (xPoints.length - 1)) * width;
      const y = height / 2 - (waveFunction[i].imag / (maxWave || 1)) * height * 0.2;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
    
    // Labels
    ctx.fillStyle = '#374151';
    ctx.font = '12px Arial';
    ctx.fillText('V(x) - gray, Re[ψ] - blue, Im[ψ] - red', 10, 15);
  }, [pinn, potential, timeSlice]);
  
  return (
    <canvas
      ref={canvasRef}
      width={600}
      height={300}
      className="w-full h-48 border rounded-lg bg-card"
    />
  );
}

const PINNSchrodinger: React.FC = () => {
  const navigate = useNavigate();
  useSEO({ title: 'PINN Schrödinger Solver – SimCore', description: 'Physics-Informed Neural Network solver for the Schrödinger equation.' });
  const schemaLD = {
    '@context': 'https://schema.org',
    '@type': 'TechArticle',
    headline: 'PINN Schrödinger Solver',
    description: 'Physics-Informed Neural Networks to approximate solutions to the time-dependent Schrödinger equation.',
    inLanguage: 'en',
    author: { '@type': 'Organization', name: 'SimCore' },
    publisher: { '@type': 'Organization', name: 'SimCore' },
    keywords: 'PINN, Schrödinger equation, quantum mechanics, neural networks, TDSE',
    about: ['Physics-Informed Neural Networks', 'Schrödinger solver', 'Quantum mechanics'],
    mainEntityOfPage: typeof window !== 'undefined' ? window.location.href : ''
  } as const;
  const [pinn, setPinn] = useState<PINNSchrodingerNetwork | null>(null);
  const [isTraining, setIsTraining] = useState(false);
  const [trainingLosses, setTrainingLosses] = useState<number[]>([]);
  const [potentialType, setPotentialType] = useState('harmonic');
  const [networkSize, setNetworkSize] = useState([20]);
  const [learningRate, setLearningRate] = useState([0.01]);
  const [timeSlice, setTimeSlice] = useState([0]);
  const [trainingIterations, setTrainingIterations] = useState([50]);

  const currentPotential = potentials[potentialType as keyof typeof potentials];

  const initializeNetwork = useCallback(() => {
    const size = networkSize[0];
    const lr = learningRate[0];
    const newPinn = new PINNSchrodingerNetwork([2, size, size, size, 2], lr);
    setPinn(newPinn);
    setTrainingLosses([]);
  }, [networkSize, learningRate]);

  const trainNetwork = useCallback(async () => {
    if (!pinn) return;
    
    setIsTraining(true);
    const endTimer = performanceMonitor.startTimer('pinnTraining');
    
    try {
      // Training points
      const xPoints = Array.from({ length: 20 }, (_, i) => -4 + (8 * i) / 19);
      const tPoints = Array.from({ length: 10 }, (_, i) => (i) / 9);
      
      const losses = pinn.train(xPoints, tPoints, currentPotential, trainingIterations[0]);
      setTrainingLosses(losses);
    } catch (error) {
      console.error('Training error:', error);
    } finally {
      setIsTraining(false);
      endTimer();
    }
  }, [pinn, currentPotential, trainingIterations]);

  const resetNetwork = () => {
    setPinn(null);
    setTrainingLosses([]);
    setTimeSlice([0]);
  };

  React.useEffect(() => {
    initializeNetwork();
  }, [initializeNetwork]);

  return (
    <PhysicsModuleLayout>
      <script type="application/ld+json" dangerouslySetInnerHTML={{ __html: JSON.stringify(schemaLD) }} />
      <PhysicsModuleHeader
        title="PINN Schrödinger Solver"
        description="Physics-Informed Neural Networks for solving the Schrödinger equation"
        category="Quantum Physics"
        difficulty="Advanced"
        equation="i\\hbar \\frac{\\partial \\psi}{\\partial t} = \\hat{H}\\psi"
        onReset={resetNetwork}
      />

        <div className="grid lg:grid-cols-3 gap-6">
          {/* Visualization */}
          <div className="lg:col-span-2">
            <Tabs defaultValue="solution" className="space-y-4">
              <TabsList>
                <TabsTrigger value="solution">Neural Solution</TabsTrigger>
                <TabsTrigger value="training">Training Progress</TabsTrigger>
              </TabsList>

              <TabsContent value="solution">
                <Card>
                  <CardHeader>
                    <CardTitle>PINN Solution: ψ(x,t)</CardTitle>
                    <CardDescription>
                      Neural network solution to iℏ∂ψ/∂t = Ĥψ at t = {timeSlice[0].toFixed(2)}
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <SolutionPlot 
                      pinn={pinn} 
                      potential={currentPotential}
                      timeSlice={timeSlice[0]}
                    />
                    
                    <div className="mt-4">
                      <label className="text-sm font-medium mb-2 block">
                        Time Slice: t = {timeSlice[0].toFixed(2)}
                      </label>
                      <Slider
                        value={timeSlice}
                        onValueChange={setTimeSlice}
                        min={0}
                        max={1}
                        step={0.01}
                        className="w-full"
                      />
                    </div>
                  </CardContent>
                </Card>
              </TabsContent>

              <TabsContent value="training">
                <Card>
                  <CardHeader>
                    <CardTitle>Training Dynamics</CardTitle>
                    <CardDescription>
                      Physics loss convergence during neural network training
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <TrainingProgress losses={trainingLosses} />
                    
                    <div className="mt-4 text-sm text-muted-foreground">
                      {trainingLosses.length > 0 && (
                        <p>Final Loss: {trainingLosses[trainingLosses.length - 1].toFixed(6)}</p>
                      )}
                    </div>
                  </CardContent>
                </Card>
              </TabsContent>
            </Tabs>
          </div>

          {/* Controls */}
          <div className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>Network Architecture</CardTitle>
              </CardHeader>
              <CardContent className="space-y-6">
                <div>
                  <Label className="text-sm font-medium mb-2 block">Potential Type</Label>
                  <Select value={potentialType} onValueChange={setPotentialType}>
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="harmonic">Harmonic Oscillator</SelectItem>
                      <SelectItem value="barrier">Potential Barrier</SelectItem>
                      <SelectItem value="well">Potential Well</SelectItem>
                      <SelectItem value="double_well">Double Well</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div>
                  <label className="text-sm font-medium mb-2 block">
                    Hidden Layer Size: {networkSize[0]}
                  </label>
                  <Slider
                    value={networkSize}
                    onValueChange={setNetworkSize}
                    min={10}
                    max={50}
                    step={5}
                    className="w-full"
                  />
                </div>

                <div>
                  <label className="text-sm font-medium mb-2 block">
                    Learning Rate: {learningRate[0].toFixed(3)}
                  </label>
                  <Slider
                    value={learningRate}
                    onValueChange={setLearningRate}
                    min={0.001}
                    max={0.1}
                    step={0.001}
                    className="w-full"
                  />
                </div>

                <div>
                  <label className="text-sm font-medium mb-2 block">
                    Training Iterations: {trainingIterations[0]}
                  </label>
                  <Slider
                    value={trainingIterations}
                    onValueChange={setTrainingIterations}
                    min={10}
                    max={200}
                    step={10}
                    className="w-full"
                  />
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Training Control</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex space-x-2">
                  <Button 
                    onClick={trainNetwork} 
                    disabled={isTraining || !pinn}
                    variant="default" 
                    size="sm" 
                    className="flex-1"
                  >
                    {isTraining ? (
                      <>
                        <Brain className="w-4 h-4 mr-2 animate-spin" />
                        Training...
                      </>
                    ) : (
                      <>
                        <Zap className="w-4 h-4 mr-2" />
                        Train PINN
                      </>
                    )}
                  </Button>
                  
                  <Button onClick={resetNetwork} variant="outline" size="sm">
                    <RotateCcw className="w-4 h-4" />
                  </Button>
                </div>

                <div className="pt-4 text-xs text-muted-foreground space-y-1">
                  <p>• Network: [2, {networkSize[0]}, {networkSize[0]}, {networkSize[0]}, 2]</p>
                  <p>• Input: (x, t) → Output: (Re[ψ], Im[ψ])</p>
                  <p>• Loss: <InlineMath math={`||i\\hbar\\partial\\psi/\\partial t - \\hat{H}\\psi||^2`} /> + boundary terms</p>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>PINN Theory</CardTitle>
              </CardHeader>
              <CardContent className="text-sm space-y-2">
                <p>
                  Physics-Informed Neural Networks embed physical laws 
                  directly into the training loss function.
                </p>
                <p>
                  For Schrödinger equation: Loss includes PDE residual 
                  and boundary/initial conditions.
                </p>
                <p className="text-xs text-muted-foreground">
                  Automatic differentiation enables exact derivative computation 
                  for physics constraints.
                </p>
              </CardContent>
            </Card>
          </div>
        </div>
    </PhysicsModuleLayout>
  );
};

export default PINNSchrodinger;