import React, { useState, useMemo, useRef, useCallback, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Line, Text, Sphere } from '@react-three/drei';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Slider } from '@/components/ui/slider';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Switch } from '@/components/ui/switch';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { ArrowLeft, Download, Settings, Info, RotateCcw, BookOpen, Zap } from 'lucide-react';
import { PhysicsModuleLayout } from '@/components/PhysicsModuleLayout';
import { PhysicsModuleHeader } from '@/components/PhysicsModuleHeader';
import * as THREE from 'three';
import { Complex, quantum } from '@/lib/physics-utils';
import { InlineMath, BlockMath } from '@/components/ui/Math';
import { useSEO } from '@/hooks/use-seo';

// Quantum state representation
interface QuantumState {
  alpha: Complex;  // |0⟩ amplitude
  beta: Complex;   // |1⟩ amplitude
}

// Convert quantum state to Bloch vector
const stateToBloch = (state: QuantumState): [number, number, number] => {
  const { alpha, beta } = state;
  
  // Bloch vector components
  const x = 2 * (alpha.real * beta.real + alpha.imag * beta.imag);
  const y = 2 * (alpha.imag * beta.real - alpha.real * beta.imag);
  const z = alpha.magnitude() ** 2 - beta.magnitude() ** 2;
  
  return [x, y, z];
};

// Enhanced quantum gate library with educational categories
const quantumGates = {
  // Single-qubit Pauli gates
  'X': {
    matrix: quantum.PauliX,
    description: 'Bit flip: |0⟩ ↔ |1⟩',
    category: 'pauli',
    duration: 1.0
  },
  'Y': {
    matrix: quantum.PauliY,
    description: 'Combined bit-phase flip',
    category: 'pauli', 
    duration: 1.0
  },
  'Z': {
    matrix: quantum.PauliZ,
    description: 'Phase flip: |1⟩ → -|1⟩',
    category: 'pauli',
    duration: 1.0
  },
  
  // Hadamard and phase gates
  'H': {
    matrix: [
      [new Complex(1/Math.sqrt(2), 0), new Complex(1/Math.sqrt(2), 0)],
      [new Complex(1/Math.sqrt(2), 0), new Complex(-1/Math.sqrt(2), 0)]
    ],
    description: 'Creates superposition',
    category: 'hadamard',
    duration: 1.0
  },
  'S': {
    matrix: [
      [new Complex(1, 0), new Complex(0, 0)],
      [new Complex(0, 0), new Complex(0, 1)]
    ],
    description: 'Phase gate: π/2 phase shift',
    category: 'phase',
    duration: 0.5
  },
  'T': {
    matrix: [
      [new Complex(1, 0), new Complex(0, 0)],
      [new Complex(0, 0), new Complex(Math.cos(Math.PI/4), Math.sin(Math.PI/4))]
    ],
    description: 'T gate: π/4 phase shift', 
    category: 'phase',
    duration: 0.25
  },
  
  // Identity (for timing)
  'I': {
    matrix: [
      [new Complex(1, 0), new Complex(0, 0)],
      [new Complex(0, 0), new Complex(1, 0)]
    ],
    description: 'Identity: no change',
    category: 'identity',
    duration: 0.1
  }
};

// Parametric gates for advanced users
const createParametricGate = (type: string, angle: number) => {
  const theta = angle * Math.PI / 180; // Convert to radians
  
  switch (type) {
    case 'Rx':
      return {
        matrix: [
          [new Complex(Math.cos(theta/2), 0), new Complex(0, -Math.sin(theta/2))],
          [new Complex(0, -Math.sin(theta/2)), new Complex(Math.cos(theta/2), 0)]
        ],
        description: `X rotation by ${angle}°`,
        category: 'rotation'
      };
    case 'Ry':
      return {
        matrix: [
          [new Complex(Math.cos(theta/2), 0), new Complex(-Math.sin(theta/2), 0)],
          [new Complex(Math.sin(theta/2), 0), new Complex(Math.cos(theta/2), 0)]
        ],
        description: `Y rotation by ${angle}°`,
        category: 'rotation'
      };
    case 'Rz':
      return {
        matrix: [
          [new Complex(Math.cos(theta/2), -Math.sin(theta/2)), new Complex(0, 0)],
          [new Complex(0, 0), new Complex(Math.cos(theta/2), Math.sin(theta/2))]
        ],
        description: `Z rotation by ${angle}°`,
        category: 'rotation'
      };
    default:
      return quantumGates['I'];
  }
};

// Decoherence and noise models
interface DecoherenceParams {
  T1: number;     // Relaxation time (μs)
  T2: number;     // Dephasing time (μs) 
  temperature: number; // Environment temperature (K)
  noiseStrength: number; // Overall noise level
}

// Apply decoherence effects to quantum state
const applyDecoherence = (
  state: QuantumState,
  dt: number,
  decoherence: DecoherenceParams
): QuantumState => {
  const { T1, T2, temperature, noiseStrength } = decoherence;
  
  // T1 relaxation: energy dissipation to environment
  const gammaRelax = 1 / T1;
  const relaxationRate = Math.exp(-dt * gammaRelax);
  
  // Thermal population (simplified)
  const kB = 8.617e-5; // Boltzmann constant (eV/K)
  const thermalFactor = temperature > 0 ? Math.exp(-1.0 / (kB * temperature)) : 0;
  
  // T2 dephasing: pure dephasing
  const gammaDephase = 1 / T2 - gammaRelax / 2;
  const dephasingRate = Math.exp(-dt * gammaDephase);
  
  // Apply T1 relaxation
  let newAlpha = state.alpha.scale(Math.sqrt(relaxationRate));
  let newBeta = state.beta.scale(Math.sqrt(relaxationRate));
  
  // Add thermal noise
  if (temperature > 0) {
    const thermalNoise = Math.sqrt((1 - relaxationRate) * thermalFactor);
    newBeta = newBeta.add(new Complex(thermalNoise, 0));
  }
  
  // Apply T2 dephasing (random phase)
  const randomPhase = (Math.random() - 0.5) * noiseStrength * dt;
  const phaseNoise = new Complex(Math.cos(randomPhase), Math.sin(randomPhase));
  newBeta = newBeta.multiply(phaseNoise.scale(dephasingRate));
  
  // Renormalize
  const norm = Math.sqrt(newAlpha.magnitude() ** 2 + newBeta.magnitude() ** 2);
  if (norm > 0) {
    newAlpha = new Complex(newAlpha.real / norm, newAlpha.imag / norm);
    newBeta = new Complex(newBeta.real / norm, newBeta.imag / norm);
  }
  
  return { alpha: newAlpha, beta: newBeta };
};

// Educational progression levels
const learningLevels = {
  beginner: {
    title: 'Level 1: Classical vs Quantum',
    description: 'Understand the difference between classical bits and qubits',
    availableGates: ['X', 'Z', 'I'],
    concepts: [
      'Classical bit: definite 0 or 1',
      'Qubit: superposition of |0⟩ and |1⟩',
      'Bloch sphere representation',
      'Measurement collapses superposition'
    ],
    exercises: [
      'Apply X gate to |0⟩',
      'Apply Z gate to |1⟩', 
      'Observe measurement outcomes'
    ]
  },
  intermediate: {
    title: 'Level 2: Superposition & Phase',
    description: 'Learn about quantum superposition and relative phases',
    availableGates: ['X', 'Y', 'Z', 'H', 'S', 'T'],
    concepts: [
      'Hadamard creates equal superposition',
      'Phase gates modify relative phase',
      'Global vs relative phase',
      'Bloch vector evolution'
    ],
    exercises: [
      'Create |+⟩ = H|0⟩',
      'Apply phase gates to |+⟩',
      'Trace paths on Bloch sphere'
    ]
  },
  advanced: {
    title: 'Level 3: Quantum Algorithms',
    description: 'Explore quantum gate sequences and simple algorithms',
    availableGates: ['X', 'Y', 'Z', 'H', 'S', 'T', 'Rx', 'Ry', 'Rz'],
    concepts: [
      'Gate composition and circuits',
      'Parametric rotations',
      'Quantum interference',
      'Measurement statistics'
    ],
    exercises: [
      'Build arbitrary rotations',
      'Create specific target states',
      'Optimize gate sequences'
    ]
  }
};

// Bloch sphere visualization component
function BlochSphere({ 
  blochVector, 
  trajectory, 
  showTrajectory,
  showAxes,
  showStates 
}: {
  blochVector: [number, number, number],
  trajectory: [number, number, number][],
  showTrajectory: boolean,
  showAxes: boolean,
  showStates: boolean
}) {
  // Validate blochVector to prevent undefined errors
  const safeBlochVector: [number, number, number] = [
    isFinite(blochVector[0]) ? blochVector[0] : 0,
    isFinite(blochVector[1]) ? blochVector[1] : 0,
    isFinite(blochVector[2]) ? blochVector[2] : 1
  ];

  // Validate trajectory points
  const safeTrajectory = trajectory.filter(point => 
    Array.isArray(point) && 
    point.length === 3 && 
    point.every(coord => isFinite(coord))
  );

  return (
    <group>
      {/* Bloch sphere */}
      <mesh>
        <sphereGeometry args={[1, 64, 64]} />
        <meshBasicMaterial 
          color="#1e40af" 
          transparent 
          opacity={0.1} 
          wireframe={false}
        />
      </mesh>
      
      {/* Sphere wireframe */}
      <mesh>
        <sphereGeometry args={[1, 16, 16]} />
        <meshBasicMaterial 
          color="#3b82f6" 
          transparent 
          opacity={0.3} 
          wireframe
        />
      </mesh>
      
      {/* Coordinate axes */}
      {showAxes && (
        <group>
          {/* X axis */}
          <mesh rotation={[0, 0, Math.PI / 2]}>
            <cylinderGeometry args={[0.01, 0.01, 2.4]} />
            <meshBasicMaterial color="#ef4444" />
          </mesh>
          
          {/* Y axis */}
          <mesh>
            <cylinderGeometry args={[0.01, 0.01, 2.4]} />
            <meshBasicMaterial color="#22c55e" />
          </mesh>
          
          {/* Z axis */}
          <mesh rotation={[Math.PI / 2, 0, 0]}>
            <cylinderGeometry args={[0.01, 0.01, 2.4]} />
            <meshBasicMaterial color="#3b82f6" />
          </mesh>
          
          <Text position={[1.3, 0, 0]} fontSize={0.15} color="#ef4444">X</Text>
          <Text position={[0, 1.3, 0]} fontSize={0.15} color="#22c55e">Y</Text>
          <Text position={[0, 0, 1.3]} fontSize={0.15} color="#3b82f6">Z</Text>
        </group>
      )}
      
      {/* Computational basis states */}
      {showStates && (
        <group>
          <mesh position={[0, 0, 1]}>
            <sphereGeometry args={[0.05, 16, 16]} />
            <meshBasicMaterial color="#fbbf24" />
          </mesh>
          <Text position={[0.1, 0.1, 1]} fontSize={0.12} color="#fbbf24">|0⟩</Text>
          
          <mesh position={[0, 0, -1]}>
            <sphereGeometry args={[0.05, 16, 16]} />
            <meshBasicMaterial color="#f97316" />
          </mesh>
          <Text position={[0.1, 0.1, -1]} fontSize={0.12} color="#f97316">|1⟩</Text>
        </group>
      )}
      
      {/* State vector as cylinder */}
      <group>
        {/* Vector line */}
        <mesh position={[safeBlochVector[0]/2, safeBlochVector[1]/2, safeBlochVector[2]/2]}>
          <cylinderGeometry args={[0.02, 0.02, Math.sqrt(
            safeBlochVector[0]**2 + safeBlochVector[1]**2 + safeBlochVector[2]**2
          )]} />
          <meshBasicMaterial color="#ec4899" />
        </mesh>
        
        {/* Vector endpoint */}
        <mesh position={safeBlochVector}>
          <sphereGeometry args={[0.06, 16, 16]} />
          <meshPhongMaterial color="#ec4899" />
        </mesh>
        
        <Text 
          position={[
            safeBlochVector[0] + 0.15, 
            safeBlochVector[1] + 0.15, 
            safeBlochVector[2] + 0.15
          ]} 
          fontSize={0.12} 
          color="#ec4899"
        >
          |ψ⟩
        </Text>
      </group>
      
      {/* Trajectory as connected spheres */}
      {showTrajectory && safeTrajectory.length > 1 && (
        <group>
          {safeTrajectory.slice(-50).map((point, index) => (
            <mesh key={index} position={point}>
              <sphereGeometry args={[0.01, 8, 8]} />
              <meshBasicMaterial 
                color="#8b5cf6" 
                transparent 
                opacity={0.3 + (index / safeTrajectory.length) * 0.7} 
              />
            </mesh>
          ))}
        </group>
      )}
    </group>
  );
}

export default function BlochSphereDynamics() {
  const navigate = useNavigate();
  useSEO({ title: 'Bloch Sphere Dynamics – SimCore', description: 'Interactive quantum gate operations and qubit evolution on the Bloch sphere with decoherence modeling.' });
  const schemaLD = useMemo(() => ({
    '@context': 'https://schema.org',
    '@type': 'TechArticle',
    headline: 'Bloch Sphere Dynamics',
    description: 'Interactive quantum gate operations and qubit evolution on the Bloch sphere with decoherence modeling.',
    inLanguage: 'en',
    author: { '@type': 'Organization', name: 'SimCore' },
    publisher: { '@type': 'Organization', name: 'SimCore' },
    keywords: 'Bloch sphere, qubit, quantum gates, decoherence',
    about: ['Bloch sphere', 'Qubits', 'Quantum gates'],
    mainEntityOfPage: typeof window !== 'undefined' ? window.location.href : ''
  }), []);
  const breadcrumbsLD = useMemo(() => ({
    '@context': 'https://schema.org',
    '@type': 'BreadcrumbList',
    itemListElement: [
      { '@type': 'ListItem', position: 1, name: 'Home', item: typeof window !== 'undefined' ? window.location.origin : '/' },
      { '@type': 'ListItem', position: 2, name: 'Modules', item: typeof window !== 'undefined' ? `${window.location.origin}/#modules-section` : '/#modules-section' },
      { '@type': 'ListItem', position: 3, name: 'Bloch Sphere Dynamics', item: typeof window !== 'undefined' ? window.location.href : '' }
    ]
  }), []);
  const [selectedGate, setSelectedGate] = useState('H');
  const [showTrajectory, setShowTrajectory] = useState(true);
  const [showAxes, setShowAxes] = useState(true);
  const [showStates, setShowStates] = useState(true);
  
  // Enhanced controls
  const [learningLevel, setLearningLevel] = useState('beginner');
  const [enableDecoherence, setEnableDecoherence] = useState(false);
  const [parametricGateType, setParametricGateType] = useState('Rx');
  const [rotationAngle, setRotationAngle] = useState([90]);
  const [gateSequence, setGateSequence] = useState<string[]>([]);
  
  // Decoherence parameters
  const [T1, setT1] = useState([100]); // μs
  const [T2, setT2] = useState([50]);  // μs
  const [temperature, setTemperature] = useState([0.01]); // K
  const [noiseStrength, setNoiseStrength] = useState([0.1]);
  
  // Performance metrics
  const [fidelity, setFidelity] = useState(1.0);
  const [gateErrors, setGateErrors] = useState(0);
  
  const currentLevel = learningLevels[learningLevel as keyof typeof learningLevels];
  const availableGates = currentLevel.availableGates;
  
  // Enhanced state management with decoherence
  const [quantumState, setQuantumState] = useState<QuantumState>({
    alpha: new Complex(1, 0), // Start at |0⟩
    beta: new Complex(0, 0)
  });
  
  const [initialState, setInitialState] = useState<QuantumState>(quantumState);
  const [targetState, setTargetState] = useState<QuantumState | null>(null);
  
  const blochVector = useMemo(() => stateToBloch(quantumState), [quantumState]);
  const [trajectory, setTrajectory] = useState<Array<[number, number, number]>>([blochVector]);
  
  // Apply gate with optional decoherence
  const applyGate = useCallback((gateName: string, customAngle?: number) => {
    setQuantumState(prevState => {
      let gate;
      
      if (['Rx', 'Ry', 'Rz'].includes(gateName)) {
        gate = createParametricGate(gateName, customAngle || rotationAngle[0]);
      } else {
        gate = quantumGates[gateName as keyof typeof quantumGates];
      }
      
      if (!gate) return prevState;
      
      // Apply gate matrix
      const gateMatrix = gate.matrix;
      const newAlpha = gateMatrix[0][0].multiply(prevState.alpha).add(
        gateMatrix[0][1].multiply(prevState.beta)
      );
      const newBeta = gateMatrix[1][0].multiply(prevState.alpha).add(
        gateMatrix[1][1].multiply(prevState.beta)
      );
      
      let newState = { alpha: newAlpha, beta: newBeta };
      
      // Apply decoherence if enabled
      if (enableDecoherence) {
        const decoherenceParams = {
          T1: T1[0],
          T2: T2[0],
          temperature: temperature[0],
          noiseStrength: noiseStrength[0]
        };
        newState = applyDecoherence(newState, gate.duration || 1.0, decoherenceParams);
      }
      
      // Calculate fidelity if target state exists
      if (targetState) {
        const overlap = newState.alpha.multiply(targetState.alpha.conjugate()).add(
          newState.beta.multiply(targetState.beta.conjugate())
        );
        setFidelity(overlap.magnitude() ** 2);
      }
      
      return newState;
    });
  }, [rotationAngle, enableDecoherence, T1, T2, temperature, noiseStrength, targetState]);
  
  // Reset to initial state
  const resetState = useCallback(() => {
    setQuantumState(initialState);
    setTrajectory([stateToBloch(initialState)]);
    setGateErrors(0);
    setFidelity(1.0);
  }, [initialState]);
  
  // Update trajectory when state changes
  useEffect(() => {
    if (showTrajectory) {
      setTrajectory(prev => {
        const newPoint = stateToBloch(quantumState);
        const updated = [...prev, newPoint];
        return updated.length > 50 ? updated.slice(-50) : updated; // Keep last 50 points
      });
    }
  }, [quantumState, showTrajectory]);

  const exportData = () => {
    const data = {
      quantumState: {
        alpha: { real: quantumState.alpha.real, imag: quantumState.alpha.imag },
        beta: { real: quantumState.beta.real, imag: quantumState.beta.imag }
      },
      blochVector,
      trajectory,
      learningLevel,
      parameters: { T1: T1[0], T2: T2[0], temperature: temperature[0] },
      timestamp: new Date().toISOString()
    };
    
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `bloch_sphere_${learningLevel}_${Date.now()}.json`;
    a.click();
  };

  return (
    <PhysicsModuleLayout>
      <script type="application/ld+json" dangerouslySetInnerHTML={{ __html: JSON.stringify(schemaLD) }} />
      
      <PhysicsModuleHeader
        title="Bloch Sphere Dynamics"
        description="Interactive quantum gate operations and qubit evolution on the Bloch sphere with decoherence modeling"
        category="Quantum Dynamics"
        difficulty="Advanced"
        equation="|\\psi\\rangle = \\alpha|0\\rangle + \\beta|1\\rangle"
        isRunning={false}
        onExport={exportData}
        onReset={resetState}
      />

      <div className="w-full">
          <Tabs defaultValue="simulation" className="h-full flex flex-col">
            <TabsList className="grid w-full grid-cols-3 mx-6">
              <TabsTrigger value="simulation" className="flex items-center gap-2">
                <Settings className="w-4 h-4" />
                Simulation
              </TabsTrigger>
              <TabsTrigger value="theory" className="flex items-center gap-2">
                <Info className="w-4 h-4" />
                Theory
              </TabsTrigger>
              <TabsTrigger value="analysis" className="flex items-center gap-2">
                <Zap className="w-4 h-4" />
                Analysis
              </TabsTrigger>
            </TabsList>

            <TabsContent value="simulation" className="flex-1 overflow-hidden px-6">
              <div className="h-full grid grid-cols-1 lg:grid-cols-4 gap-6">
                {/* Controls Panel */}
                <Card className="lg:col-span-1 p-6 bg-card/95 backdrop-blur-sm h-full overflow-y-auto">
                <div className="space-y-6">
                  <div>
                    <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                      <Settings className="w-5 h-5" />
                      Quantum Controls
                    </h3>
                    
                    {/* Learning Level Selection */}
                    <div className="space-y-2">
                      <Label className="text-sm font-medium">Learning Level</Label>
                      <Select value={learningLevel} onValueChange={setLearningLevel}>
                        <SelectTrigger>
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="beginner">Level 1: Classical vs Quantum</SelectItem>
                          <SelectItem value="intermediate">Level 2: Superposition & Phase</SelectItem>
                          <SelectItem value="advanced">Level 3: Quantum Algorithms</SelectItem>
                        </SelectContent>
                      </Select>
                      <p className="text-xs text-muted-foreground">
                        {currentLevel.description}
                      </p>
                    </div>

                    {/* Gate Selection */}
                    <div className="space-y-2">
                      <Label className="text-sm font-medium">Quantum Gate</Label>
                      <Select value={selectedGate} onValueChange={setSelectedGate}>
                        <SelectTrigger>
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          {availableGates.map(gateName => {
                            const gate = quantumGates[gateName as keyof typeof quantumGates];
                            
                            // Handle parametric gates separately
                            if (['Rx', 'Ry', 'Rz'].includes(gateName)) {
                              const descriptions = {
                                'Rx': 'X rotation by θ',
                                'Ry': 'Y rotation by θ', 
                                'Rz': 'Z rotation by θ'
                              };
                              return (
                                <SelectItem key={gateName} value={gateName}>
                                  {gateName} - {descriptions[gateName as keyof typeof descriptions]}
                                </SelectItem>
                              );
                            }
                            
                            // Safety check for regular gates
                            if (!gate) {
                              console.warn(`Gate ${gateName} not found in quantumGates`);
                              return null;
                            }
                            
                            return (
                              <SelectItem key={gateName} value={gateName}>
                                {gateName} - {gate.description}
                              </SelectItem>
                            );
                          })}
                          {learningLevel === 'advanced' && (
                            <>
                              {/* These are now handled above, but keeping for backwards compatibility */}
                            </>
                          )}
                        </SelectContent>
                      </Select>
                    </div>

                    {/* Parametric gate angle (advanced level) */}
                    {learningLevel === 'advanced' && ['Rx', 'Ry', 'Rz'].includes(selectedGate) && (
                      <div className="space-y-2">
                        <Label className="text-sm">
                          Rotation Angle: <InlineMath math={`${rotationAngle[0]}^\\\circ`} />
                        </Label>
                        <Slider
                          value={rotationAngle}
                          onValueChange={setRotationAngle}
                          min={0}
                          max={360}
                          step={15}
                          className="w-full"
                        />
                      </div>
                    )}

                    {/* Gate sequence builder */}
                    {learningLevel !== 'beginner' && (
                      <div className="space-y-2">
                        <Label className="text-sm font-medium">Gate Sequence</Label>
                        <div className="flex gap-2">
                          <Button 
                            size="sm" 
                            variant="outline"
                            onClick={() => setGateSequence(prev => [...prev, selectedGate])}
                          >
                            Add {selectedGate}
                          </Button>
                          <Button 
                            size="sm" 
                            variant="outline"
                            onClick={() => setGateSequence([])}
                          >
                            Clear
                          </Button>
                        </div>
                        <div className="text-xs bg-muted p-2 rounded">
                          Sequence: {gateSequence.join(' → ') || 'Empty'}
                        </div>
                      </div>
                    )}

                    {/* Action buttons */}
                    <div className="space-y-2">
                      <Button 
                        onClick={() => applyGate(selectedGate)} 
                        className="w-full"
                      >
                        Apply {selectedGate} Gate
                      </Button>
                      
                      <Button 
                        onClick={resetState} 
                        variant="outline" 
                        className="w-full"
                      >
                        <RotateCcw className="w-4 h-4 mr-2" />
                        Reset to |0⟩
                      </Button>
                    </div>
                  </div>

                  {/* Decoherence Controls (Intermediate+) */}
                  {learningLevel !== 'beginner' && (
                    <div className="border-t pt-4">
                      <h4 className="font-medium mb-3">Decoherence & Noise</h4>
                      
                      <div className="flex items-center space-x-2 mb-3">
                        <Switch 
                          checked={enableDecoherence} 
                          onCheckedChange={setEnableDecoherence}
                          id="decoherence"
                        />
                        <Label htmlFor="decoherence" className="text-sm">Enable Environmental Effects</Label>
                      </div>

                      {enableDecoherence && (
                        <div className="space-y-3">
                          <div>
                            <Label className="text-sm">T₁ Relaxation: {T1[0]} μs</Label>
                            <Slider
                              value={T1}
                              onValueChange={setT1}
                              min={10}
                              max={500}
                              step={10}
                              className="w-full"
                            />
                          </div>
                          
                          <div>
                            <Label className="text-sm">T₂ Dephasing: {T2[0]} μs</Label>
                            <Slider
                              value={T2}
                              onValueChange={setT2}
                              min={5}
                              max={200}
                              step={5}
                              className="w-full"
                            />
                          </div>
                          
                          <div>
                            <Label className="text-sm">Temperature: {temperature[0]} K</Label>
                            <Slider
                              value={temperature}
                              onValueChange={setTemperature}
                              min={0}
                              max={1}
                              step={0.01}
                              className="w-full"
                            />
                          </div>
                        </div>
                      )}
                    </div>
                  )}

                  {/* Visualization Controls */}
                  <div className="border-t pt-4">
                    <h4 className="font-medium mb-3">Visualization</h4>
                    
                    <div className="space-y-2">
                      <div className="flex items-center space-x-2">
                        <Switch 
                          checked={showTrajectory} 
                          onCheckedChange={setShowTrajectory}
                          id="trajectory"
                        />
                        <Label htmlFor="trajectory" className="text-sm">Show Trajectory</Label>
                      </div>
                      
                      <div className="flex items-center space-x-2">
                        <Switch 
                          checked={showAxes} 
                          onCheckedChange={setShowAxes}
                          id="axes"
                        />
                        <Label htmlFor="axes" className="text-sm">Show Axes</Label>
                      </div>
                      
                      <div className="flex items-center space-x-2">
                        <Switch 
                          checked={showStates} 
                          onCheckedChange={setShowStates}
                          id="states"
                        />
                        <Label htmlFor="states" className="text-sm">Show |0⟩, |1⟩</Label>
                      </div>
                    </div>
                  </div>
                </div>
              </Card>

                {/* Visualization Panel */}
                <div className="lg:col-span-3 space-y-4 overflow-y-auto">
                  {/* 3D Bloch Sphere */}
                  <Card className="p-6 h-[60vh]">
                    <h3 className="text-lg font-semibold mb-4">Interactive Bloch Sphere</h3>
                  <div className="h-full bg-muted/20 rounded-lg border">
                    <Canvas camera={{ position: [2, 2, 2], fov: 50 }}>
                      <ambientLight intensity={0.6} />
                      <pointLight position={[5, 5, 5]} intensity={0.8} />
                      
                      <BlochSphere 
                        blochVector={blochVector}
                        trajectory={trajectory}
                        showTrajectory={showTrajectory}
                        showAxes={showAxes}
                        showStates={showStates}
                      />
                      
                      <OrbitControls 
                        enablePan={true} 
                        enableZoom={true} 
                        enableRotate={true}
                        enableDamping={true}
                        dampingFactor={0.05}
                        minDistance={1.5}
                        maxDistance={10}
                      />
                    </Canvas>
                    </div>
                  </Card>

                  {/* State Information */}
                  <Card className="p-6 h-[35vh]">
                    <h3 className="text-lg font-semibold mb-4">Quantum State Information</h3>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="bg-muted/30 p-4 rounded-lg">
                      <h4 className="font-medium mb-2">Amplitudes</h4>
                      <div className="space-y-1 text-sm font-mono">
                        <div>α = {quantumState.alpha.real.toFixed(3)} + {quantumState.alpha.imag.toFixed(3)}i</div>
                        <div>β = {quantumState.beta.real.toFixed(3)} + {quantumState.beta.imag.toFixed(3)}i</div>
                      </div>
                    </div>
                    
                    <div className="bg-muted/30 p-4 rounded-lg">
                      <h4 className="font-medium mb-2">Probabilities</h4>
                      <div className="space-y-1 text-sm">
                        <div>P(|0⟩) = {(quantumState.alpha.magnitude() ** 2).toFixed(3)}</div>
                        <div>P(|1⟩) = {(quantumState.beta.magnitude() ** 2).toFixed(3)}</div>
                      </div>
                    </div>
                    
                    <div className="bg-muted/30 p-4 rounded-lg">
                      <h4 className="font-medium mb-2">Bloch Coordinates</h4>
                      <div className="space-y-1 text-sm">
                        <div>x = {blochVector[0].toFixed(3)}</div>
                        <div>y = {blochVector[1].toFixed(3)}</div>
                        <div>z = {blochVector[2].toFixed(3)}</div>
                      </div>
                    </div>
                    
                    {enableDecoherence && (
                      <div className="bg-muted/30 p-4 rounded-lg">
                        <h4 className="font-medium mb-2">Fidelity</h4>
                        <div className="text-2xl font-bold">
                          {(fidelity * 100).toFixed(1)}%
                        </div>
                      </div>
                    )}
                  </div>
                  </Card>
                </div>
              </div>
            </TabsContent>

          <TabsContent value="theory">
            {/* Educational Theory Panel */}
            <Card className="p-6">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <BookOpen className="w-4 h-4" />
                  {currentLevel.title}
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-6">
                <div>
                  <h4 className="font-medium mb-2">Key Concepts:</h4>
                  <ul className="list-disc list-inside space-y-1 text-muted-foreground">
                    {currentLevel.concepts.map((concept, idx) => (
                      <li key={idx}>{concept}</li>
                    ))}
                  </ul>
                </div>
                
                <div>
                  <h4 className="font-medium mb-2">Try These Exercises:</h4>
                  <ul className="list-disc list-inside space-y-1 text-muted-foreground">
                    {currentLevel.exercises.map((exercise, idx) => (
                      <li key={idx}>{exercise}</li>
                    ))}
                  </ul>
                </div>

                {learningLevel !== 'beginner' && (
                  <div>
                    <h4 className="font-medium mb-2">Mathematical Framework:</h4>
                    <div className="space-y-2">
                      <div>
                        <h5 className="text-xs font-medium">Qubit State:</h5>
                        <BlockMath math="|ψ⟩ = α|0⟩ + β|1⟩" />
                      </div>
                      <div>
                        <h5 className="text-xs font-medium">Bloch Vector:</h5>
                        <BlockMath math="\vec{r} = (⟨σ_x⟩, ⟨σ_y⟩, ⟨σ_z⟩)" />
                      </div>
                      {enableDecoherence && (
                        <div>
                          <h5 className="text-xs font-medium">Decoherence:</h5>
                          <BlockMath math="T_1: \text{energy relaxation}, T_2: \text{dephasing}" />
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="analysis" className="space-y-6">
            <Card className="p-6">
              <h3 className="text-lg font-semibold mb-4">Performance Analysis</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                <div className="bg-muted/30 p-4 rounded-lg">
                  <h4 className="font-medium mb-2">Current Level</h4>
                  <div className="text-2xl font-bold capitalize">
                    {learningLevel}
                  </div>
                </div>
                <div className="bg-muted/30 p-4 rounded-lg">
                  <h4 className="font-medium mb-2">Gates Applied</h4>
                  <div className="text-2xl font-bold">
                    {trajectory.length - 1}
                  </div>
                </div>
                <div className="bg-muted/30 p-4 rounded-lg">
                  <h4 className="font-medium mb-2">State Purity</h4>
                  <div className="text-2xl font-bold">
                    {(quantumState.alpha.magnitude() ** 2 + quantumState.beta.magnitude() ** 2).toFixed(3)}
                  </div>
                </div>
                <div className="bg-muted/30 p-4 rounded-lg">
                  <h4 className="font-medium mb-2">Decoherence</h4>
                  <div className="text-2xl font-bold">
                    {enableDecoherence ? 'ON' : 'OFF'}
                  </div>
                </div>
              </div>
            </Card>
            </TabsContent>
          </Tabs>
        </div>
    </PhysicsModuleLayout>
  );
}

