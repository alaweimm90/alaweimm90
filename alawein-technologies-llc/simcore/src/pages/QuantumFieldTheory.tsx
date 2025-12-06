import React, { useState, useMemo, useRef, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Slider } from '@/components/ui/slider';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Switch } from '@/components/ui/switch';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Home, Play, Pause, RotateCcw, Zap, Atom } from 'lucide-react';
import { PhysicsModuleLayout } from '@/components/PhysicsModuleLayout';
import { PhysicsModuleHeader } from '@/components/PhysicsModuleHeader';
import { Complex, errorUtils } from '@/lib/physics-utils';
import { performanceMonitor, AnimationController } from '@/lib/performance-utils';
import { InlineMath, BlockMath } from '@/components/ui/Math';
import { useSEO } from '@/hooks/use-seo';

// Quantum field representation
interface QuantumField {
  position: number[];
  fieldValue: Complex[];
  vacuumFluctuations: number[];
  particleNumber: number[];
  energy: number[];
}

// Field quantization calculations
const quantizeField = (
  x: number[],
  t: number,
  couplingConstant: number = 1.0,
  vacuumEnergy: number = 0.5,
  particleCount: number = 0
) => {
  const endTimer = performanceMonitor.startTimer('fieldQuantization');
  
  try {
    const fieldValue: Complex[] = [];
    const vacuumFluctuations: number[] = [];
    const particleNumber: number[] = [];
    const energy: number[] = [];
    
    const L = x.length;
    const dx = x[1] - x[0];
    
    for (let i = 0; i < L; i++) {
      const xi = x[i];
      
      // Mode expansion: φ(x,t) = Σₖ [aₖe^(ikx-iωₖt) + a†ₖe^(-ikx+iωₖt)]
      let fieldReal = 0;
      let fieldImag = 0;
      let vacuum = 0;
      let particles = 0;
      let localEnergy = 0;
      
      // Sum over momentum modes
      const maxModes = 20;
      for (let k = 1; k <= maxModes; k++) {
        const momentum = (2 * Math.PI * k) / L;
        const frequency = Math.sqrt(momentum * momentum + couplingConstant * couplingConstant);
        
        // Creation/annihilation operator amplitudes
        const amplitude = 1 / Math.sqrt(2 * frequency);
        
        // Field contribution from mode k
        const phase = momentum * xi - frequency * t;
        const creationTerm = Math.cos(phase);
        const annihilationTerm = Math.cos(-phase);
        
        fieldReal += amplitude * (creationTerm + annihilationTerm);
        fieldImag += amplitude * (Math.sin(phase) - Math.sin(-phase));
        
        // Vacuum fluctuations (zero-point energy)
        vacuum += 0.5 * frequency * amplitude * amplitude * Math.abs(Math.sin(momentum * xi + frequency * t));
        
        // Particle number density (simplified)
        if (particleCount > 0) {
          const particlePhase = momentum * xi - frequency * t + Math.PI * k / 4;
          particles += (particleCount / maxModes) * Math.exp(-0.1 * (xi * xi)) * Math.abs(Math.cos(particlePhase));
        }
        
        // Energy density
        const gradientTerm = momentum * momentum * amplitude * amplitude;
        const potentialTerm = 0.5 * couplingConstant * couplingConstant * fieldReal * fieldReal;
        localEnergy += 0.5 * (gradientTerm + potentialTerm);
      }
      
      fieldValue.push(new Complex(fieldReal, fieldImag));
      vacuumFluctuations.push(vacuumEnergy + vacuum);
      particleNumber.push(particles);
      energy.push(localEnergy);
    }
    
    return {
      position: x,
      fieldValue,
      vacuumFluctuations,
      particleNumber,
      energy
    };
  } catch (error) {
    console.error('Field quantization error:', error);
    return {
      position: x,
      fieldValue: x.map(() => Complex.zero()),
      vacuumFluctuations: x.map(() => 0),
      particleNumber: x.map(() => 0),
      energy: x.map(() => 0)
    };
  } finally {
    endTimer();
  }
};

// Calculate field observables
const calculateObservables = (field: QuantumField) => {
  const totalEnergy = field.energy.reduce((sum, e) => sum + e, 0);
  const totalParticles = field.particleNumber.reduce((sum, n) => sum + n, 0);
  const fieldNorm = field.fieldValue.reduce((sum, f) => sum + f.magnitude() ** 2, 0);
  const vacuumExpectation = field.vacuumFluctuations.reduce((sum, v) => sum + v, 0) / field.vacuumFluctuations.length;
  
  return {
    totalEnergy,
    totalParticles,
    fieldNorm,
    vacuumExpectation
  };
};

// Canvas visualization component
function FieldVisualization({ field, visualizationType }: { 
  field: QuantumField, 
  visualizationType: string 
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  
  const drawField = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas || !field) return;
    
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    const { width, height } = canvas;
    ctx.clearRect(0, 0, width, height);
    
    const data = visualizationType === 'field' ? field.fieldValue.map(f => f.magnitude()) :
                 visualizationType === 'vacuum' ? field.vacuumFluctuations :
                 visualizationType === 'particles' ? field.particleNumber :
                 field.energy;
    
    const maxVal = Math.max(...data);
    const minVal = Math.min(...data);
    const range = maxVal - minVal || 1;
    
    // Draw field/data
    ctx.strokeStyle = visualizationType === 'field' ? '#3b82f6' :
                     visualizationType === 'vacuum' ? '#8b5cf6' :
                     visualizationType === 'particles' ? '#ef4444' :
                     '#10b981';
    ctx.lineWidth = 2;
    ctx.beginPath();
    
    for (let i = 0; i < data.length; i++) {
      const x = (i / (data.length - 1)) * width;
      const y = height - ((data[i] - minVal) / range) * height * 0.8 - height * 0.1;
      
      if (i === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    }
    ctx.stroke();
    
    // Add labels
    ctx.fillStyle = '#374151';
    ctx.font = '14px Arial';
    ctx.fillText(visualizationType === 'field' ? 'Field φ(x,t)' :
                visualizationType === 'vacuum' ? 'Vacuum Fluctuations' :
                visualizationType === 'particles' ? 'Particle Density' :
                'Energy Density', 10, 20);
    
    ctx.fillText('Position', width / 2 - 20, height - 10);
  }, [field, visualizationType]);
  
  // Redraw when field or type changes
  React.useEffect(() => {
    drawField();
  }, [drawField]);
  
  return (
    <canvas
      ref={canvasRef}
      width={800}
      height={300}
      className="w-full h-64 border rounded-lg bg-card"
    />
  );
}

const QuantumFieldTheory: React.FC = () => {
  const navigate = useNavigate();
  useSEO({ title: 'Quantum Field Theory Simulator – SimCore', description: 'Interactive field quantization with particle operations and vacuum fluctuations.' });
  const schemaLD = {
    '@context': 'https://schema.org',
    '@type': 'TechArticle',
    headline: 'Quantum Field Theory Simulator',
    description: 'Interactive field quantization with particle creation/annihilation, vacuum fluctuations, and field observables.',
    inLanguage: 'en',
    author: { '@type': 'Organization', name: 'SimCore' },
    publisher: { '@type': 'Organization', name: 'SimCore' },
    keywords: 'quantum field theory, field quantization, vacuum fluctuations, particle creation, annihilation operators',
    about: ['Quantum fields', 'QFT', 'Vacuum fluctuations', 'Creation and annihilation operators'],
    mainEntityOfPage: typeof window !== 'undefined' ? window.location.href : ''
  } as const;
  const animationControllerRef = useRef<AnimationController>(new AnimationController(30));
  
  const [isRunning, setIsRunning] = useState(false);
  const [time, setTime] = useState(0);
  const [couplingConstant, setCouplingConstant] = useState([1.0]);
  const [vacuumEnergy, setVacuumEnergy] = useState([0.5]);
  const [particleCount, setParticleCount] = useState([0]);
  const [visualizationType, setVisualizationType] = useState('field');
  const [fieldLength, setFieldLength] = useState([20]);
  
  // Spatial grid
  const spatialGrid = useMemo(() => {
    const L = fieldLength[0];
    const N = 200;
    return Array.from({ length: N }, (_, i) => -L/2 + (L * i) / (N - 1));
  }, [fieldLength]);
  
  // Calculate quantum field
  const quantumField = useMemo(() => 
    quantizeField(spatialGrid, time, couplingConstant[0], vacuumEnergy[0], particleCount[0]),
    [spatialGrid, time, couplingConstant, vacuumEnergy, particleCount]
  );
  
  const observables = useMemo(() => calculateObservables(quantumField), [quantumField]);
  
  // Animation control
  React.useEffect(() => {
    const controller = animationControllerRef.current;
    
    if (isRunning) {
      controller.start((deltaTime) => {
        setTime(prev => prev + deltaTime);
      });
    } else {
      controller.stop();
    }
    
    return () => controller.stop();
  }, [isRunning]);
  
  const toggleSimulation = () => setIsRunning(!isRunning);
  
  const resetSimulation = () => {
    setIsRunning(false);
    setTime(0);
    animationControllerRef.current.stop();
  };
  
  const createParticle = () => {
    setParticleCount(prev => [prev[0] + 1]);
  };
  
  const annihilateParticle = () => {
    setParticleCount(prev => [Math.max(0, prev[0] - 1)]);
  };
  
  const exportData = () => {
    console.log('Export QFT data');
  };


  return (
    <PhysicsModuleLayout>
      <script type="application/ld+json" dangerouslySetInnerHTML={{ __html: JSON.stringify(schemaLD) }} />
      <PhysicsModuleHeader
        title="Quantum Field Theory Simulator"
        description="Interactive field quantization with particle creation/annihilation and vacuum fluctuations"
        category="Quantum Field Theory"
        difficulty="Advanced"
        equation="\\hat{H} = \\sum_k \\hbar\\omega_k(a_k^\\dagger a_k + \\frac{1}{2})"
        isRunning={isRunning}
        onReset={resetSimulation}
        onExport={exportData}
      />

      <div className="w-full">
          {/* Visualization */}
          <div className="lg:col-span-2">
            <Tabs defaultValue="field" className="space-y-4">
              <TabsList>
                <TabsTrigger value="field">Quantum Field</TabsTrigger>
                <TabsTrigger value="observables">Observables</TabsTrigger>
              </TabsList>

              <TabsContent value="field">
                <Card>
                  <CardHeader>
                    <CardTitle>Quantum Field Visualization</CardTitle>
                    <CardDescription>
                      Real-time evolution of quantized field <InlineMath math="\\phi(x,t)" /> and associated properties
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-2">
                        <Button onClick={toggleSimulation} variant="outline">
                          {isRunning ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
                          {isRunning ? 'Pause' : 'Start'}
                        </Button>
                        <Button onClick={resetSimulation} variant="outline">
                          <RotateCcw className="w-4 h-4" />
                          Reset
                        </Button>
                      </div>
                      <div className="text-sm text-muted-foreground">
                        Time: <InlineMath math={`${time.toFixed(2)} \\hbar/c`} />
                      </div>
                    </div>
                    
                    <FieldVisualization field={quantumField} visualizationType={visualizationType} />
                    
                    <div className="flex items-center space-x-4">
                      <Label className="text-sm">View:</Label>
                      <Select value={visualizationType} onValueChange={setVisualizationType}>
                        <SelectTrigger className="w-40">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="field">Field <InlineMath math={`\\phi(x,t)`} errorColor={'#cc0000'} throwOnError={false} /></SelectItem>
                          <SelectItem value="vacuum">Vacuum Fluctuations</SelectItem>
                          <SelectItem value="particles">Particle Density</SelectItem>
                          <SelectItem value="energy">Energy Density</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                  </CardContent>
                </Card>
              </TabsContent>

              <TabsContent value="observables">
                <Card>
                  <CardHeader>
                    <CardTitle>Field Observables</CardTitle>
                    <CardDescription>
                      Quantum field theory observables and vacuum properties
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="grid grid-cols-2 gap-6">
                      <div className="space-y-4">
                        <div className="text-center">
                          <div className="text-2xl font-bold text-primary">
                            {observables.totalEnergy.toFixed(3)}
                          </div>
                          <div className="text-sm text-muted-foreground">Total Energy (<InlineMath math={`\\hbar c`} />)</div>
                        </div>
                        
                        <div className="text-center">
                          <div className="text-2xl font-bold text-secondary">
                            {observables.totalParticles.toFixed(1)}
                          </div>
                          <div className="text-sm text-muted-foreground">Particle Number</div>
                        </div>
                      </div>
                      
                      <div className="space-y-4">
                        <div className="text-center">
                          <div className="text-2xl font-bold text-green-500">
                            {observables.fieldNorm.toFixed(3)}
                          </div>
                          <div className="text-sm text-muted-foreground">Field Norm <InlineMath math={`\\langle\\phi^2\\rangle`} /></div>
                        </div>
                        
                        <div className="text-center">
                          <div className="text-2xl font-bold text-purple-500">
                            {observables.vacuumExpectation.toFixed(3)}
                          </div>
                          <div className="text-sm text-muted-foreground">Vacuum Expectation</div>
                        </div>
                      </div>
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
                <CardTitle>Field Parameters</CardTitle>
              </CardHeader>
              <CardContent className="space-y-6">
                <div>
                  <label className="text-sm font-medium mb-2 block">
                    Coupling Constant: {couplingConstant[0].toFixed(2)}
                  </label>
                  <Slider
                    value={couplingConstant}
                    onValueChange={setCouplingConstant}
                    min={0.1}
                    max={3.0}
                    step={0.1}
                    className="w-full"
                  />
                </div>

                <div>
                  <label className="text-sm font-medium mb-2 block">
                    Vacuum Energy: <InlineMath math={`${vacuumEnergy[0].toFixed(2)} \\hbar\\omega`} />
                  </label>
                  <Slider
                    value={vacuumEnergy}
                    onValueChange={setVacuumEnergy}
                    min={0.1}
                    max={2.0}
                    step={0.1}
                    className="w-full"
                  />
                </div>

                <div>
                  <label className="text-sm font-medium mb-2 block">
                    Field Length: {fieldLength[0]} units
                  </label>
                  <Slider
                    value={fieldLength}
                    onValueChange={setFieldLength}
                    min={10}
                    max={50}
                    step={5}
                    className="w-full"
                  />
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Particle Operations</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="text-center space-y-2">
                  <div className="text-lg font-semibold">
                    Particles: {particleCount[0]}
                  </div>
                  
                  <div className="flex space-x-2">
                    <Button onClick={createParticle} variant="outline" size="sm" className="flex-1">
                      <Zap className="w-4 h-4 mr-1" />
                      Create <InlineMath math={`a^\\dagger`} />
                    </Button>
                    <Button onClick={annihilateParticle} variant="outline" size="sm" className="flex-1">
                      <Atom className="w-4 h-4 mr-1" />
                      Annihilate <InlineMath math="a" />
                    </Button>
                  </div>
                </div>

                <div className="pt-4 border-t text-xs text-muted-foreground space-y-1">
                  <p>• <InlineMath math={`a^\\dagger`} /> creates particles in the field</p>
                  <p>• <InlineMath math={`a`} /> annihilates particles from the field</p>
                  <p>• <InlineMath math={`[a, a^\\dagger] = 1`} /> (canonical commutation)</p>
                </div>
              </CardContent>
            </Card>

             <Tabs defaultValue="overview" className="h-full">
               <TabsList className="grid w-full grid-cols-4">
                 <TabsTrigger value="overview">Overview</TabsTrigger>
                 <TabsTrigger value="mathematics">Mathematics</TabsTrigger>
                 <TabsTrigger value="applications">Applications</TabsTrigger>
                 <TabsTrigger value="references">References</TabsTrigger>
               </TabsList>

               <TabsContent value="overview" className="space-y-4">
                 <Card>
                   <CardHeader>
                     <CardTitle>Quantum Field Theory Fundamentals</CardTitle>
                   </CardHeader>
                   <CardContent className="space-y-3">
                     <p className="text-sm leading-relaxed">
                       Quantum field theory (QFT) describes particles as quantized excitations 
                       of underlying quantum fields that permeate all of spacetime.
                     </p>
                     
                     <div className="space-y-2">
                       <h4 className="font-semibold">Key Concepts:</h4>
                       <ul className="text-sm space-y-1 ml-4">
                         <li>• <strong>Field Quantization:</strong> Classical fields become operators</li>
                         <li>• <strong>Vacuum State:</strong> Lowest energy state with zero particles</li>
                         <li>• <strong>Particle Creation:</strong> Acting with creation operators</li>
                         <li>• <strong>Vacuum Fluctuations:</strong> Inherent quantum uncertainty</li>
                       </ul>
                     </div>

                     <div className="p-3 bg-muted/30 rounded-lg">
                       <h5 className="font-medium mb-2">Field Expansion:</h5>
                       <BlockMath math="\\phi(x,t) = \\sum_k \\left[ a_k e^{i(kx - \\omega_k t)} + a_k^\\dagger e^{-i(kx - \\omega_k t)} \\right]" />
                       <p className="text-xs text-muted-foreground mt-2">
                         where <InlineMath math="a_k" /> and <InlineMath math="a_k^\\dagger" /> are annihilation and creation operators.
                       </p>
                     </div>
                   </CardContent>
                 </Card>
               </TabsContent>

               <TabsContent value="mathematics" className="space-y-4">
                 <Card>
                   <CardHeader>
                     <CardTitle>Mathematical Framework</CardTitle>
                   </CardHeader>
                   <CardContent className="space-y-4">
                     <div>
                       <h4 className="font-semibold mb-2">Canonical Commutation Relations:</h4>
                       <BlockMath math="[a_k, a_{k'}^\\dagger] = \\delta_{k,k'}" />
                       <BlockMath math="[a_k, a_{k'}] = [a_k^\\dagger, a_{k'}^\\dagger] = 0" />
                     </div>

                     <div>
                       <h4 className="font-semibold mb-2">Hamiltonian:</h4>
                       <BlockMath math="H = \\sum_k \\hbar \\omega_k \\left( a_k^\\dagger a_k + \\frac{1}{2} \\right)" />
                       <p className="text-sm text-muted-foreground">
                         The <InlineMath math="\\frac{1}{2}" /> term represents zero-point energy of vacuum fluctuations.
                       </p>
                     </div>

                     <div>
                       <h4 className="font-semibold mb-2">Number Operator:</h4>
                       <BlockMath math="N_k = a_k^\\dagger a_k" />
                       <p className="text-sm text-muted-foreground">
                         Eigenvalues give particle number in mode k.
                       </p>
                     </div>

                     <div>
                       <h4 className="font-semibold mb-2">Vacuum State Properties:</h4>
                       <BlockMath math="a_k |0\\rangle = 0" />
                       <BlockMath math="\\langle 0 | \\phi^2 | 0 \\rangle \\neq 0" />
                       <p className="text-sm text-muted-foreground">
                         The vacuum has zero particles but non-zero field fluctuations.
                       </p>
                     </div>
                   </CardContent>
                 </Card>
               </TabsContent>

               <TabsContent value="applications" className="space-y-4">
                 <Card>
                   <CardHeader>
                     <CardTitle>Applications & Phenomena</CardTitle>
                   </CardHeader>
                   <CardContent className="space-y-4">
                     <div>
                       <h4 className="font-semibold mb-2">Casimir Effect:</h4>
                       <p className="text-sm mb-2">
                         Vacuum fluctuations between conducting plates create attractive force.
                       </p>
                       <BlockMath math="F = -\\frac{\\pi^2 \\hbar c}{240 d^4} A" />
                     </div>

                     <div>
                       <h4 className="font-semibold mb-2">Hawking Radiation:</h4>
                       <p className="text-sm mb-2">
                         Black holes emit radiation due to vacuum fluctuations near event horizon.
                       </p>
                       <BlockMath math="T_H = \\frac{\\hbar g}{2\\pi c k_B}" />
                     </div>

                     <div>
                       <h4 className="font-semibold mb-2">Spontaneous Symmetry Breaking:</h4>
                       <p className="text-sm mb-2">
                         Vacuum state doesn't respect symmetries of the Lagrangian.
                       </p>
                       <BlockMath math="\\langle 0 | \\phi | 0 \\rangle = v \\neq 0" />
                     </div>

                     <div>
                       <h4 className="font-semibold mb-2">Feynman Diagrams:</h4>
                       <p className="text-sm">
                         Graphical representation of particle interactions and scattering processes.
                       </p>
                     </div>
                   </CardContent>
                 </Card>
               </TabsContent>

               <TabsContent value="references" className="space-y-4">
                 <Card>
                   <CardHeader>
                     <CardTitle>References & Further Reading</CardTitle>
                   </CardHeader>
                   <CardContent className="space-y-4">
                     <div>
                       <h4 className="font-semibold mb-2">Textbooks:</h4>
                       <ul className="text-sm space-y-1">
                         <li>• Peskin & Schroeder, "An Introduction to Quantum Field Theory" (1995)</li>
                         <li>• Weinberg, "The Quantum Theory of Fields" (1995)</li>
                         <li>• Srednicki, "Quantum Field Theory" (2007)</li>
                         <li>• Tong, "Quantum Field Theory" (Cambridge Lecture Notes)</li>
                       </ul>
                     </div>

                     <div>
                       <h4 className="font-semibold mb-2">Historical Papers:</h4>
                       <ul className="text-sm space-y-1">
                         <li>• Dirac, "The quantum theory of emission and absorption" (1927)</li>
                         <li>• Heisenberg & Pauli, "Zur Quantendynamik der Wellenfelder" (1929)</li>
                         <li>• Feynman, "Space-time approach to quantum electrodynamics" (1949)</li>
                       </ul>
                     </div>

                     <div>
                       <h4 className="font-semibold mb-2">Modern Reviews:</h4>
                       <ul className="text-sm space-y-1">
                         <li>• Weinberg, "What is quantum field theory?" Rev. Mod. Phys. (1997)</li>
                         <li>• Wilczek, "Quantum field theory" Rev. Mod. Phys. (1999)</li>
                       </ul>
                     </div>

                     <div className="p-3 bg-muted/30 rounded-lg">
                       <p className="text-xs text-muted-foreground">
                         <strong>Note:</strong> This simulation demonstrates basic concepts of field quantization, 
                         vacuum fluctuations, and particle creation/annihilation in a simplified 1D scalar field theory.
                       </p>
                     </div>
                   </CardContent>
                 </Card>
                </TabsContent>
               </Tabs>
             </div>
         </div>
    </PhysicsModuleLayout>
  );
};

export default QuantumFieldTheory;