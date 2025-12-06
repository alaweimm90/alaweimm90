import React, { useState, useEffect, useMemo, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Line, Text } from '@react-three/drei';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Slider } from '@/components/ui/slider';
import { Badge } from '@/components/ui/badge';
import { Switch } from '@/components/ui/switch';
import { Label } from '@/components/ui/label';
import { Progress } from '@/components/ui/progress';
import { Separator } from '@/components/ui/separator';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { ScientificPlot } from '@/components/ScientificPlot';
import { PhysicsTheory } from '@/components/PhysicsTheory';
import { PerformanceBenchmark } from '@/components/PerformanceBenchmark';
import { ArrowLeft, Download, Play, Pause, RotateCcw, Shuffle, BarChart3, Calculator, TrendingUp, Settings, Activity, Info, Database, Layers } from 'lucide-react';
import { InlineMath, BlockMath } from '@/components/ui/Math';
import { useToast } from '@/hooks/use-toast';
import * as THREE from 'three';
import { PhysicsModuleLayout } from '@/components/PhysicsModuleLayout';
import { PhysicsModuleHeader } from '@/components/PhysicsModuleHeader';
import { useSEO } from '@/hooks/use-seo';

type SystemType = 'ideal_gas' | 'spin_system' | 'harmonic_chain' | 'lattice_gas' | 'information_system' | 'custom';
type EnsembleType = 'microcanonical' | 'canonical' | 'grand_canonical';

interface Particle {
  id: number;
  position: THREE.Vector3;
  velocity: THREE.Vector3;
  energy: number;
  compartment: number;
  spin?: number;
  state?: string;
}

interface MicrostateData {
  id: number;
  configuration: number[];
  energy: number;
  entropy: number;
  probability: number;
  degeneracy: number;
  description: string;
}

interface EntropyCalculations {
  boltzmann: number;
  shannon: number;
  thermodynamic: number;
  maxEntropy: number;
  relativeEntropy: number;
  mutualInformation: number;
  conditionalEntropy: number;
}

// Advanced entropy calculation engine
class EntropyEngine {
  private systemType: SystemType;
  private ensembleType: EnsembleType;
  private numParticles: number;
  private numStates: number;
  private temperature: number;
  private parameters: any;
  
  // Physical constants
  private readonly kB = 1.381e-23; // J/K
  private readonly ln2 = Math.log(2);
  
  constructor(
    systemType: SystemType, 
    ensembleType: EnsembleType,
    numParticles: number, 
    numStates: number, 
    temperature: number,
    parameters: any
  ) {
    this.systemType = systemType;
    this.ensembleType = ensembleType;
    this.numParticles = numParticles;
    this.numStates = numStates;
    this.temperature = temperature;
    this.parameters = parameters;
  }
  
  // Generate all possible microstates for the system
  generateMicrostates(): MicrostateData[] {
    switch (this.systemType) {
      case 'ideal_gas':
        return this.idealGasMicrostates();
      case 'spin_system':
        return this.spinSystemMicrostates();
      case 'harmonic_chain':
        return this.harmonicChainMicrostates();
      case 'lattice_gas':
        return this.latticeGasMicrostates();
      case 'information_system':
        return this.informationSystemMicrostates();
      default:
        return this.idealGasMicrostates();
    }
  }
  
  private idealGasMicrostates(): MicrostateData[] {
    const microstates: MicrostateData[] = [];
    
    // Generate all ways to distribute N particles among M compartments
    const distributions = this.generateDistributions(this.numParticles, this.numStates);
    
    distributions.forEach((dist, id) => {
      const energy = this.calculateIdealGasEnergy(dist);
      const degeneracy = this.calculateMultinomialCoefficient(this.numParticles, dist);
      const entropy = Math.log(degeneracy);
      
      microstates.push({
        id,
        configuration: dist,
        energy,
        entropy,
        probability: 0, // Will be calculated later
        degeneracy,
        description: `Distribution: [${dist.join(', ')}]`
      });
    });
    
    return this.calculateProbabilities(microstates);
  }
  
  private spinSystemMicrostates(): MicrostateData[] {
    const microstates: MicrostateData[] = [];
    const magneticField = this.parameters.magneticField || 0;
    
    // For N spins, generate all 2^N configurations
    const totalConfigs = Math.pow(2, this.numParticles);
    
    for (let config = 0; config < Math.min(totalConfigs, 1000); config++) {
      const spins = [];
      let magnetization = 0;
      
      for (let i = 0; i < this.numParticles; i++) {
        const spin = (config >> i) & 1 ? 1 : -1;
        spins.push(spin);
        magnetization += spin;
      }
      
      const energy = -magneticField * magnetization; // Zeeman energy
      const entropy = 0; // Each microstate has equal weight initially
      
      microstates.push({
        id: config,
        configuration: spins,
        energy,
        entropy,
        probability: 0,
        degeneracy: 1,
        description: `M = ${magnetization}, Config: ${spins.join('')}`
      });
    }
    
    return this.calculateProbabilities(microstates);
  }
  
  private harmonicChainMicrostates(): MicrostateData[] {
    const microstates: MicrostateData[] = [];
    const omega = this.parameters.frequency || 1.0;
    const maxQuanta = this.parameters.maxQuanta || 10;
    
    // Distribute energy quanta among oscillators
    const distributions = this.generateDistributions(maxQuanta, this.numParticles);
    
    distributions.forEach((dist, id) => {
      const energy = omega * dist.reduce((sum, n) => sum + n, 0);
      const degeneracy = 1; // Each configuration is unique
      const entropy = Math.log(degeneracy);
      
      microstates.push({
        id,
        configuration: dist,
        energy,
        entropy,
        probability: 0,
        degeneracy,
        description: `Quanta: [${dist.join(', ')}]`
      });
    });
    
    return this.calculateProbabilities(microstates);
  }
  
  private latticeGasMicrostates(): MicrostateData[] {
    const microstates: MicrostateData[] = [];
    const interactionStrength = this.parameters.interaction || 0;
    
    // Generate configurations for lattice gas
    const sites = Math.min(this.numStates, 20); // Limit for computational feasibility
    const totalConfigs = Math.pow(2, sites);
    
    for (let config = 0; config < Math.min(totalConfigs, 1000); config++) {
      const occupation = [];
      let numParticles = 0;
      let energy = 0;
      
      for (let i = 0; i < sites; i++) {
        const occupied = (config >> i) & 1;
        occupation.push(occupied);
        numParticles += occupied;
        
        // Nearest neighbor interactions
        if (i > 0 && occupied && occupation[i-1]) {
          energy += interactionStrength;
        }
      }
      
      // Only include configurations with correct particle number
      if (numParticles <= this.numParticles) {
        microstates.push({
          id: config,
          configuration: occupation,
          energy,
          entropy: 0,
          probability: 0,
          degeneracy: 1,
          description: `N = ${numParticles}, E_int = ${energy.toFixed(2)}`
        });
      }
    }
    
    return this.calculateProbabilities(microstates);
  }
  
  private informationSystemMicrostates(): MicrostateData[] {
    const microstates: MicrostateData[] = [];
    const alphabet = this.parameters.alphabetSize || this.numStates;
    
    // Generate information-theoretic microstates
    const distributions = this.generateDistributions(this.numParticles, alphabet);
    
    distributions.forEach((dist, id) => {
      const informationContent = this.calculateInformationContent(dist);
      const degeneracy = this.calculateMultinomialCoefficient(this.numParticles, dist);
      const entropy = Math.log(degeneracy);
      
      microstates.push({
        id,
        configuration: dist,
        energy: informationContent,
        entropy,
        probability: 0,
        degeneracy,
        description: `Symbol counts: [${dist.join(', ')}]`
      });
    });
    
    return this.calculateProbabilities(microstates);
  }
  
  private calculateProbabilities(microstates: MicrostateData[]): MicrostateData[] {
    const beta = 1 / (this.kB * this.temperature);
    
    // Calculate partition function
    let Z = 0;
    microstates.forEach(state => {
      if (this.ensembleType === 'canonical') {
        Z += state.degeneracy * Math.exp(-beta * state.energy);
      } else {
        Z += state.degeneracy; // Microcanonical
      }
    });
    
    // Prevent division by zero
    if (Z === 0 || !isFinite(Z)) {
      Z = 1;
    }
    
    // Calculate probabilities
    microstates.forEach(state => {
      if (this.ensembleType === 'canonical') {
        const boltzmannFactor = Math.exp(-beta * state.energy);
        if (!isFinite(boltzmannFactor)) {
          state.probability = 0;
        } else {
          state.probability = (state.degeneracy * boltzmannFactor) / Z;
        }
      } else {
        state.probability = state.degeneracy / Z; // Microcanonical
      }
      
      // Ensure probability is finite and non-negative
      if (!isFinite(state.probability) || state.probability < 0) {
        state.probability = 0;
      }
    });
    
    // Normalize probabilities to ensure they sum to 1
    const totalProb = microstates.reduce((sum, state) => sum + state.probability, 0);
    if (totalProb > 0) {
      microstates.forEach(state => {
        state.probability /= totalProb;
      });
    }
    
    return microstates.sort((a, b) => b.probability - a.probability);
  }
  
  private generateDistributions(total: number, bins: number): number[][] {
    const distributions: number[][] = [];
    
    function generate(remaining: number, currentBin: number, current: number[]): void {
      if (currentBin === bins - 1) {
        distributions.push([...current, remaining]);
        return;
      }
      
      for (let i = 0; i <= remaining; i++) {
        generate(remaining - i, currentBin + 1, [...current, i]);
      }
    }
    
    if (bins > 0) {
      generate(total, 0, []);
    }
    
    return distributions.slice(0, 1000); // Limit for performance
  }
  
  private calculateMultinomialCoefficient(n: number, distribution: number[]): number {
    let result = this.factorial(n);
    distribution.forEach(ni => {
      result /= this.factorial(ni);
    });
    return result;
  }
  
  private factorial(n: number): number {
    if (n <= 1) return 1;
    let result = 1;
    for (let i = 2; i <= Math.min(n, 170); i++) { // Prevent overflow
      result *= i;
    }
    return result;
  }
  
  private calculateIdealGasEnergy(distribution: number[]): number {
    // Simple model: energy proportional to local density
    return distribution.reduce((sum, n, i) => sum + n * (i + 1) * 0.1, 0);
  }
  
  private calculateInformationContent(distribution: number[]): number {
    const total = distribution.reduce((sum, n) => sum + n, 0);
    if (total === 0) return 0;
    
    return -distribution.reduce((sum, n) => {
      if (n > 0) {
        const p = n / total;
        return sum + n * Math.log2(p);
      }
      return sum;
    }, 0);
  }
  
  // Advanced entropy calculations
  calculateAdvancedEntropies(microstates: MicrostateData[]): EntropyCalculations {
    // Boltzmann entropy
    const totalDegeneracy = microstates.reduce((sum, state) => sum + state.degeneracy, 0);
    const boltzmann = Math.log(totalDegeneracy);
    
    // Shannon entropy
    const shannon = -microstates.reduce((sum, state) => {
      if (state.probability > 0) {
        return sum + state.probability * Math.log2(state.probability);
      }
      return sum;
    }, 0);
    
    // Thermodynamic entropy (from average energy)
    const avgEnergy = microstates.reduce((sum, state) => sum + state.energy * state.probability, 0);
    const thermodynamic = boltzmann + avgEnergy / this.temperature;
    
    // Maximum entropy (equiprobable states)
    const maxEntropy = Math.log2(microstates.length);
    
    // Relative entropy (KL divergence from uniform)
    const uniformProb = 1 / microstates.length;
    const relativeEntropy = microstates.reduce((sum, state) => {
      if (state.probability > 0) {
        return sum + state.probability * Math.log2(state.probability / uniformProb);
      }
      return sum;
    }, 0);
    
    return {
      boltzmann,
      shannon,
      thermodynamic,
      maxEntropy,
      relativeEntropy,
      mutualInformation: maxEntropy - shannon, // Simplified
      conditionalEntropy: shannon * 0.8 // Placeholder
    };
  }
}

// Enhanced 3D visualization component
interface AdvancedMicrostateVisualizerProps {
  microstates: MicrostateData[];
  currentStateIndex: number;
  systemType: SystemType;
  showProbabilities: boolean;
}

const AdvancedMicrostateVisualizer: React.FC<AdvancedMicrostateVisualizerProps> = ({ 
  microstates, 
  currentStateIndex,
  systemType,
  showProbabilities
}) => {
  const groupRef = React.useRef<THREE.Group>(null);
  
  useFrame((state) => {
    if (groupRef.current) {
      groupRef.current.rotation.y += 0.005;
    }
  });
  
  if (!microstates[currentStateIndex]) return null;
  
  const currentState = microstates[currentStateIndex];
  
  const renderConfiguration = () => {
    const elements = [];
    
    if (systemType === 'ideal_gas' || systemType === 'information_system') {
      // Bar chart representation
      currentState.configuration.forEach((count, i) => {
        const height = Math.max(count * 0.5, 0.1);
        elements.push(
          <mesh key={i} position={[i * 2 - 4, height/2, 0]}>
            <boxGeometry args={[0.8, height, 0.8]} />
            <meshPhongMaterial 
              color={showProbabilities ? 
                new THREE.Color().setHSL(currentState.probability, 0.8, 0.6) :
                new THREE.Color().setHSL(i / currentState.configuration.length, 0.8, 0.6)
              } 
            />
          </mesh>
        );
      });
    } else if (systemType === 'spin_system') {
      // Spin chain representation
      currentState.configuration.forEach((spin, i) => {
        elements.push(
          <mesh key={i} position={[i * 1.5 - 4, 0, 0]} rotation={[0, 0, spin > 0 ? 0 : Math.PI]}>
            <coneGeometry args={[0.3, 1, 8]} />
            <meshPhongMaterial color={spin > 0 ? '#ff4444' : '#4444ff'} />
          </mesh>
        );
      });
    } else if (systemType === 'lattice_gas') {
      // Lattice site representation
      currentState.configuration.forEach((occupied, i) => {
        const x = (i % 5) * 1.5 - 3;
        const y = Math.floor(i / 5) * 1.5 - 2;
        
        elements.push(
          <mesh key={i} position={[x, y, 0]}>
            <sphereGeometry args={[occupied ? 0.4 : 0.2, 8, 6]} />
            <meshPhongMaterial 
              color={occupied ? '#00ff00' : '#888888'} 
              opacity={occupied ? 1.0 : 0.3}
              transparent
            />
          </mesh>
        );
      });
    }
    
    return elements;
  };
  
  return (
    <group ref={groupRef}>
      {renderConfiguration()}
      
      {/* Information display */}
      <Text
        position={[0, 4, 0]}
        fontSize={0.5}
        color="white"
        anchorX="center"
        anchorY="middle"
      >
        {currentState.description}
      </Text>
      
      <Text
        position={[0, 3, 0]}
        fontSize={0.3}
        color="yellow"
        anchorX="center"
        anchorY="middle"
      >
        P = {currentState.probability.toFixed(4)}
      </Text>
    </group>
  );
};

export default function MicrostatesEntropy() {
  const navigate = useNavigate();
  const { toast } = useToast();
  
  useSEO({ title: 'Microstates & Entropy – SimCore', description: 'Analyze microstates, entropy measures, and information-theoretic properties across systems.' });
  const schemaLD = {
    '@context': 'https://schema.org',
    '@type': 'TechArticle',
    headline: 'Microstates & Entropy Engine',
    description: 'Advanced entropy analysis with microstate visualization and information-theoretic metrics.',
    inLanguage: 'en',
    author: { '@type': 'Organization', name: 'SimCore' },
    publisher: { '@type': 'Organization', name: 'SimCore' },
    keywords: 'entropy, microstates, statistical physics, information theory, Shannon entropy',
    about: ['Boltzmann entropy', 'Shannon entropy', 'Microcanonical and canonical ensembles'],
    mainEntityOfPage: typeof window !== 'undefined' ? window.location.href : ''
  } as const;
  
  // Enhanced state management
  const [activeTab, setActiveTab] = useState('simulation');
  const [systemType, setSystemType] = useState<SystemType>('ideal_gas');
  const [ensembleType, setEnsembleType] = useState<EnsembleType>('canonical');
  const [numParticles, setNumParticles] = useState([6]);
  const [numStates, setNumStates] = useState([4]);
  const [temperature, setTemperature] = useState([300]);
  
  // Animation and visualization
  const [isAnimating, setIsAnimating] = useState(false);
  const [currentStateIndex, setCurrentStateIndex] = useState(0);
  const [showProbabilities, setShowProbabilities] = useState(true);
  const [show3D, setShow3D] = useState(true);
  const [animationSpeed, setAnimationSpeed] = useState([500]);
  
  // System parameters
  const [parameters, setParameters] = useState({
    magneticField: 0.1,
    frequency: 1.0,
    maxQuanta: 10,
    interaction: 0.1,
    alphabetSize: 4
  });
  
  // Analysis data
  const [entropyEvolution, setEntropyEvolution] = useState<number[]>([]);
  const [probabilityHistory, setProbabilityHistory] = useState<number[][]>([]);
  
  // Create entropy engine
  const entropyEngine = useMemo(() => 
    new EntropyEngine(
      systemType, 
      ensembleType, 
      numParticles[0], 
      numStates[0], 
      temperature[0],
      parameters
    ),
    [systemType, ensembleType, numParticles, numStates, temperature, parameters]
  );
  
  // Generate microstates
  const microstates = useMemo(() => {
    try {
      return entropyEngine.generateMicrostates();
    } catch (error) {
      console.error('Error generating microstates:', error);
      return [];
    }
  }, [entropyEngine]);
  
  // Calculate advanced entropies
  const entropies = useMemo(() => {
    if (microstates.length === 0) return null;
    return entropyEngine.calculateAdvancedEntropies(microstates);
  }, [entropyEngine, microstates]);
  
  // Animation logic
  useEffect(() => {
    if (!isAnimating || microstates.length === 0) return;
    
    const interval = setInterval(() => {
      setCurrentStateIndex(prev => {
        const next = (prev + 1) % microstates.length;
        
        // Record entropy evolution
        if (microstates[next]) {
          setEntropyEvolution(prevEvolution => 
            [...prevEvolution, microstates[next].entropy].slice(-100)
          );
          
          setProbabilityHistory(prevHistory => {
            const newEntry = microstates.map(state => state.probability);
            return [...prevHistory, newEntry].slice(-50);
          });
        }
        
        return next;
      });
    }, animationSpeed[0]);
    
    return () => clearInterval(interval);
  }, [isAnimating, microstates, animationSpeed]);
  
  // Reset simulation
  const resetSimulation = () => {
    setCurrentStateIndex(0);
    setEntropyEvolution([]);
    setProbabilityHistory([]);
    setIsAnimating(false);
  };
  
  // Export data
  const exportData = () => {
    const data = {
      systemType,
      ensembleType,
      parameters: {
        numParticles: numParticles[0],
        numStates: numStates[0],
        temperature: temperature[0],
        ...parameters
      },
      microstates,
      entropies,
      entropyEvolution,
      timestamp: new Date().toISOString()
    };
    
    const dataStr = JSON.stringify(data, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'microstates_entropy_analysis.json';
    link.click();
  };
  
  // Get system-specific parameter controls
  const getParameterControls = () => {
    switch (systemType) {
      case 'spin_system':
        return (
          <div className="space-y-2">
            <Label className="text-sm font-medium">
              Magnetic Field: {parameters.magneticField} T
            </Label>
            <Slider
              value={[parameters.magneticField]}
              onValueChange={(value) => setParameters(prev => ({ ...prev, magneticField: value[0] }))}
              min={0}
              max={1}
              step={0.01}
              className="w-full"
            />
          </div>
        );
      
      case 'harmonic_chain':
        return (
          <div className="space-y-4">
            <div className="space-y-2">
              <Label className="text-sm font-medium">
                Frequency: {parameters.frequency} ω₀
              </Label>
              <Slider
                value={[parameters.frequency]}
                onValueChange={(value) => setParameters(prev => ({ ...prev, frequency: value[0] }))}
                min={0.1}
                max={5}
                step={0.1}
                className="w-full"
              />
            </div>
            <div className="space-y-2">
              <Label className="text-sm font-medium">
                Max Quanta: {parameters.maxQuanta}
              </Label>
              <Slider
                value={[parameters.maxQuanta]}
                onValueChange={(value) => setParameters(prev => ({ ...prev, maxQuanta: value[0] }))}
                min={5}
                max={20}
                step={1}
                className="w-full"
              />
            </div>
          </div>
        );
      
      case 'lattice_gas':
        return (
          <div className="space-y-2">
            <Label className="text-sm font-medium">
              Interaction: {parameters.interaction} J
            </Label>
            <Slider
              value={[parameters.interaction]}
              onValueChange={(value) => setParameters(prev => ({ ...prev, interaction: value[0] }))}
              min={-1}
              max={1}
              step={0.01}
              className="w-full"
            />
          </div>
        );
      
      case 'information_system':
        return (
          <div className="space-y-2">
            <Label className="text-sm font-medium">
              Alphabet Size: {parameters.alphabetSize}
            </Label>
            <Slider
              value={[parameters.alphabetSize]}
              onValueChange={(value) => setParameters(prev => ({ ...prev, alphabetSize: value[0] }))}
              min={2}
              max={10}
              step={1}
              className="w-full"
            />
          </div>
        );
      
      default:
        return null;
    }
  };

  return (
    <PhysicsModuleLayout>
      <script type="application/ld+json" dangerouslySetInnerHTML={{ __html: JSON.stringify(schemaLD) }} />
      <PhysicsModuleHeader
        title="Microstates & Entropy Engine"
        description="Advanced entropy analysis and microstate visualization for statistical physics systems"
        category="Statistical Physics"
        difficulty="Advanced"
        equation="S = k_B \\ln(\\Omega)"
        onReset={() => window.location.reload()}
        onExport={exportData}
      />
      <div className="max-w-7xl mx-auto p-6">

        <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
          <TabsList className="grid w-full grid-cols-4 mb-8">
            <TabsTrigger value="simulation" className="flex items-center gap-2">
              <Activity className="w-4 h-4" />
              Microstates
            </TabsTrigger>
            <TabsTrigger value="analysis" className="flex items-center gap-2">
              <BarChart3 className="w-4 h-4" />
              Entropy Analysis
            </TabsTrigger>
            <TabsTrigger value="theory" className="flex items-center gap-2">
              <Calculator className="w-4 h-4" />
              Theory
            </TabsTrigger>
            <TabsTrigger value="systems" className="flex items-center gap-2">
              <Layers className="w-4 h-4" />
              System Types
            </TabsTrigger>
          </TabsList>

          {/* Main Simulation Tab */}
          <TabsContent value="simulation" className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              {/* Enhanced Controls */}
              <Card className="p-6">
                <div className="flex items-center gap-2 mb-4">
                  <Settings className="w-5 h-5 text-primary" />
                  <h3 className="text-xl font-semibold font-serif">System Configuration</h3>
                </div>
                
                <div className="space-y-6">
                  {/* System Type Selection */}
                  <div>
                    <Label className="text-sm font-medium">System Type</Label>
                    <Select value={systemType} onValueChange={(value: SystemType) => setSystemType(value)}>
                      <SelectTrigger className="mt-1">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="ideal_gas">Ideal Gas</SelectItem>
                        <SelectItem value="spin_system">Spin System</SelectItem>
                        <SelectItem value="harmonic_chain">Harmonic Chain</SelectItem>
                        <SelectItem value="lattice_gas">Lattice Gas</SelectItem>
                        <SelectItem value="information_system">Information System</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>

                  {/* Ensemble Type */}
                  <div>
                    <Label className="text-sm font-medium">Statistical Ensemble</Label>
                    <Select value={ensembleType} onValueChange={(value: EnsembleType) => setEnsembleType(value)}>
                      <SelectTrigger className="mt-1">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="microcanonical">Microcanonical (E fixed)</SelectItem>
                        <SelectItem value="canonical">Canonical (T fixed)</SelectItem>
                        <SelectItem value="grand_canonical">Grand Canonical (μ, T fixed)</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>

                  {/* Basic Parameters */}
                  <div className="space-y-4">
                    <div className="space-y-2">
                      <Label className="text-sm font-medium">
                        Number of Particles: {numParticles[0]}
                      </Label>
                      <Slider
                        value={numParticles}
                        onValueChange={setNumParticles}
                        min={2}
                        max={12}
                        step={1}
                        className="w-full"
                      />
                    </div>

                    <div className="space-y-2">
                      <Label className="text-sm font-medium">
                        Number of States/Compartments: {numStates[0]}
                      </Label>
                      <Slider
                        value={numStates}
                        onValueChange={setNumStates}
                        min={2}
                        max={8}
                        step={1}
                        className="w-full"
                      />
                    </div>

                    <div className="space-y-2">
                      <Label className="text-sm font-medium">
                        Temperature: {temperature[0]} K
                      </Label>
                      <Slider
                        value={temperature}
                        onValueChange={setTemperature}
                        min={100}
                        max={1000}
                        step={10}
                        className="w-full"
                      />
                    </div>
                  </div>

                  {/* System-specific parameters */}
                  {getParameterControls()}

                  {/* Animation Controls */}
                  <Separator />
                  <div className="space-y-4">
                    <div className="space-y-2">
                      <Label className="text-sm font-medium">
                        Animation Speed: {animationSpeed[0]}ms
                      </Label>
                      <Slider
                        value={animationSpeed}
                        onValueChange={setAnimationSpeed}
                        min={100}
                        max={2000}
                        step={100}
                        className="w-full"
                      />
                    </div>

                    <div className="flex gap-2">
                      <Button
                        onClick={() => setIsAnimating(!isAnimating)}
                        className="flex-1"
                        variant={isAnimating ? "destructive" : "default"}
                      >
                        {isAnimating ? (
                          <>
                            <Pause className="w-4 h-4 mr-2" />
                            Pause
                          </>
                        ) : (
                          <>
                            <Play className="w-4 h-4 mr-2" />
                            Animate
                          </>
                        )}
                      </Button>
                      <Button onClick={resetSimulation} variant="outline">
                        <RotateCcw className="w-4 h-4" />
                      </Button>
                    </div>

                    <div className="flex items-center justify-between">
                      <Label className="text-sm font-medium">Show 3D Visualization</Label>
                      <Switch checked={show3D} onCheckedChange={setShow3D} />
                    </div>

                    <div className="flex items-center justify-between">
                      <Label className="text-sm font-medium">Color by Probability</Label>
                      <Switch checked={showProbabilities} onCheckedChange={setShowProbabilities} />
                    </div>
                  </div>

                  {/* Current State Info */}
                  <Separator />
                  <div className="space-y-2 text-sm">
                    <div className="font-medium text-center">Current Microstate</div>
                    {microstates[currentStateIndex] && (
                      <div className="p-3 bg-muted/30 rounded space-y-1">
                        <div>State: {currentStateIndex + 1}/{microstates.length}</div>
                        <div>Probability: {microstates[currentStateIndex].probability.toFixed(6)}</div>
                        <div>Degeneracy: {microstates[currentStateIndex].degeneracy}</div>
                        <div>Energy: {microstates[currentStateIndex].energy.toFixed(3)}</div>
                      </div>
                    )}
                  </div>
                </div>
              </Card>

              {/* 3D Visualization */}
              <Card className="lg:col-span-2 p-6">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-xl font-semibold">Microstate Visualization</h3>
                  <div className="text-sm text-muted-foreground">
                    {systemType.charAt(0).toUpperCase() + systemType.slice(1).replace('_', ' ')} System
                  </div>
                </div>
                
                {show3D && microstates.length > 0 ? (
                  <div className="h-96 bg-muted/10 rounded-lg">
                    <Canvas camera={{ position: [0, 0, 15], fov: 60 }}>
                      <ambientLight intensity={0.5} />
                      <directionalLight position={[10, 10, 5]} intensity={1} />
                      <AdvancedMicrostateVisualizer
                        microstates={microstates}
                        currentStateIndex={currentStateIndex}
                        systemType={systemType}
                        showProbabilities={showProbabilities}
                      />
                      <OrbitControls enableZoom={true} enablePan={true} />
                    </Canvas>
                  </div>
                ) : (
                  <div className="h-96 flex items-center justify-center bg-muted/10 rounded-lg">
                    <div className="text-center">
                      <Database className="w-12 h-12 mx-auto mb-4 text-muted-foreground" />
                      <p className="text-muted-foreground">
                        {microstates.length === 0 ? 'Generating microstates...' : '3D visualization disabled'}
                      </p>
                    </div>
                  </div>
                )}
              </Card>
            </div>

            {/* Microstate Browser */}
            {microstates.length > 0 && (
              <Card className="p-6">
                <h3 className="text-lg font-semibold mb-4">Microstate Browser</h3>
                <div className="space-y-4">
                  <div className="flex items-center gap-4">
                    <Label className="text-sm font-medium">Microstate:</Label>
                    <Slider
                      value={[currentStateIndex]}
                      onValueChange={(value) => setCurrentStateIndex(value[0])}
                      min={0}
                      max={microstates.length - 1}
                      step={1}
                      className="flex-1"
                    />
                    <span className="text-sm font-mono min-w-20">
                      {currentStateIndex + 1}/{microstates.length}
                    </span>
                  </div>
                  
                  {microstates[currentStateIndex] && (
                    <div className="p-4 bg-muted/30 rounded-lg">
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                        <div>
                          <div className="text-muted-foreground">Description</div>
                          <div className="font-mono">{microstates[currentStateIndex].description}</div>
                        </div>
                        <div>
                          <div className="text-muted-foreground">Probability</div>
                          <div className="font-mono">{microstates[currentStateIndex].probability.toFixed(6)}</div>
                        </div>
                        <div>
                          <div className="text-muted-foreground">Degeneracy</div>
                          <div className="font-mono">{microstates[currentStateIndex].degeneracy}</div>
                        </div>
                        <div>
                          <div className="text-muted-foreground">Entropy</div>
                          <div className="font-mono">{microstates[currentStateIndex].entropy.toFixed(3)} kB</div>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              </Card>
            )}
          </TabsContent>

          {/* Entropy Analysis Tab */}
          <TabsContent value="analysis" className="space-y-6">
            {/* Entropy Metrics */}
            {entropies && (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                <Card className="p-4">
                  <div className="text-center">
                    <div className="text-2xl font-bold text-blue-400">{entropies.boltzmann.toFixed(3)}</div>
                    <div className="text-sm text-muted-foreground">Boltzmann Entropy (kB)</div>
                    <div className="text-xs mt-1">S = ln(Ω)</div>
                  </div>
                </Card>
                
                <Card className="p-4">
                  <div className="text-center">
                    <div className="text-2xl font-bold text-green-400">{entropies.shannon.toFixed(3)}</div>
                    <div className="text-sm text-muted-foreground">Shannon Entropy (bits)</div>
                    <div className="text-xs mt-1">H = -Σ p log₂(p)</div>
                  </div>
                </Card>
                
                <Card className="p-4">
                  <div className="text-center">
                    <div className="text-2xl font-bold text-purple-400">{entropies.maxEntropy.toFixed(3)}</div>
                    <div className="text-sm text-muted-foreground">Maximum Entropy (bits)</div>
                    <div className="text-xs mt-1">H_max = log₂(N)</div>
                  </div>
                </Card>
                
                <Card className="p-4">
                  <div className="text-center">
                    <div className="text-2xl font-bold text-orange-400">{((1 - entropies.relativeEntropy/entropies.maxEntropy) * 100).toFixed(1)}%</div>
                    <div className="text-sm text-muted-foreground">Efficiency</div>
                    <div className="text-xs mt-1">H/H_max</div>
                  </div>
                </Card>
              </div>
            )}

            {/* Entropy Visualizations */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <Card className="p-6">
                <h3 className="text-lg font-semibold mb-4">Microstate Probabilities</h3>
                <ScientificPlot
                  title="Probability Distribution"
                  data={microstates.slice(0, 20).map((state, i) => ({ x: i, y: state.probability }))}
                  plotType="1d"
                  xLabel="Microstate Index"
                  yLabel="Probability"
                  showGrid={true}
                  showAxes={true}
                />
              </Card>

              <Card className="p-6">
                <h3 className="text-lg font-semibold mb-4">Energy Distribution</h3>
                <ScientificPlot
                  title="Energy vs Microstate"
                  data={microstates.slice(0, 20).map((state, i) => ({ x: i, y: state.energy }))}
                  plotType="1d"
                  xLabel="Microstate Index"
                  yLabel="Energy"
                  showGrid={true}
                  showAxes={true}
                />
              </Card>

              {entropyEvolution.length > 0 && (
                <Card className="p-6">
                  <h3 className="text-lg font-semibold mb-4">Entropy Evolution</h3>
                  <ScientificPlot
                    title="Entropy vs Time"
                    data={entropyEvolution.map((entropy, i) => ({ x: i, y: entropy }))}
                    plotType="1d"
                    xLabel="Time Steps"
                    yLabel="Entropy (kB)"
                    showGrid={true}
                    showAxes={true}
                  />
                </Card>
              )}

              <Card className="p-6">
                <h3 className="text-lg font-semibold mb-4">Degeneracy Distribution</h3>
                <ScientificPlot
                  title="Degeneracy vs Microstate"
                  data={microstates.slice(0, 20).map((state, i) => ({ x: i, y: Math.log10(state.degeneracy) }))}
                  plotType="1d"
                  xLabel="Microstate Index"
                  yLabel="log₁₀(Degeneracy)"
                  showGrid={true}
                  showAxes={true}
                />
              </Card>
            </div>

            {/* Information Theory Analysis */}
            {entropies && (
              <Card className="p-6">
                <h3 className="text-lg font-semibold mb-4">Information Theory Analysis</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div>
                    <h4 className="font-medium mb-3">Entropy Measures</h4>
                    <div className="space-y-2 text-sm">
                      <div className="flex justify-between">
                        <span>Boltzmann Entropy:</span>
                        <span className="font-mono">{entropies.boltzmann.toFixed(4)} kB</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Shannon Entropy:</span>
                        <span className="font-mono">{entropies.shannon.toFixed(4)} bits</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Max Entropy:</span>
                        <span className="font-mono">{entropies.maxEntropy.toFixed(4)} bits</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Relative Entropy:</span>
                        <span className="font-mono">{entropies.relativeEntropy.toFixed(4)} bits</span>
                      </div>
                    </div>
                  </div>
                  
                  <div>
                    <h4 className="font-medium mb-3">Information Content</h4>
                    <div className="space-y-2 text-sm">
                      <div className="flex justify-between">
                        <span>Total Microstates:</span>
                        <span className="font-mono">{microstates.length}</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Effective States:</span>
                        <span className="font-mono">{Math.pow(2, entropies.shannon).toFixed(1)}</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Compression Ratio:</span>
                        <span className="font-mono">{(entropies.shannon / entropies.maxEntropy).toFixed(3)}</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Redundancy:</span>
                        <span className="font-mono">{((1 - entropies.shannon / entropies.maxEntropy) * 100).toFixed(1)}%</span>
                      </div>
                    </div>
                  </div>
                </div>
              </Card>
            )}
          </TabsContent>

          {/* Theory Tab */}
          <TabsContent value="theory" className="space-y-6">
            <PhysicsTheory module={{
              title: "Advanced Microstates & Entropy",
              description: "Statistical foundations, information theory, and thermodynamic connections",
              category: "Statistical Physics",
              difficulty: "Beginner" as const,
              theory: {
                overview: "Entropy connects microscopic disorder to macroscopic thermodynamic properties through statistical mechanics and information theory.",
                mathematics: [
                  "Boltzmann entropy: S = kB ln(Ω)",
                  "Shannon entropy: H = -Σ pi log₂(pi)",
                  "Multinomial coefficient: Ω = N!/(n₁!n₂!...nM!)",
                  "Relative entropy: D(P||Q) = Σ pi log(pi/qi)",
                  "Maximum entropy principle: maximize S subject to constraints",
                  "Fluctuation-dissipation: ⟨δX²⟩ = kBT ∂²S/∂X²"
                ],
                references: [
                  "Boltzmann, L. (1877). Über die Beziehung zwischen dem zweiten Hauptsatze der mechanischen Wärmetheorie",
                  "Shannon, C. E. (1948). A Mathematical Theory of Communication",
                  "Jaynes, E. T. (1957). Information Theory and Statistical Mechanics"
                ]
              }
            }} />
          </TabsContent>

          {/* System Types Tab */}
          <TabsContent value="systems" className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <Card className="p-6">
                <h3 className="text-lg font-semibold mb-4">Ideal Gas System</h3>
                <div className="space-y-3 text-sm">
                  <div><strong>Configuration:</strong> Particles distributed among compartments</div>
                  <div><strong>Microstates:</strong> All possible distributions N → (N₁, N₂, ..., NM)</div>
                  <div><strong>Degeneracy:</strong> Ω = N!/(N₁!N₂!...NM!)</div>
                  <div><strong>Applications:</strong> Gas expansion, mixing entropy</div>
                  <div className="p-2 bg-muted/30 rounded text-xs">
                    Classic example of configurational entropy in thermodynamics
                  </div>
                </div>
              </Card>

              <Card className="p-6">
                <h3 className="text-lg font-semibold mb-4">Spin System</h3>
                <div className="space-y-3 text-sm">
                  <div><strong>Configuration:</strong> N spins, each ±1 in magnetic field</div>
                  <div><strong>Microstates:</strong> 2ᴺ total configurations</div>
                  <div><strong>Energy:</strong> E = -B·M (Zeeman interaction)</div>
                  <div><strong>Applications:</strong> Magnetic materials, Ising model</div>
                  <div className="p-2 bg-muted/30 rounded text-xs">
                    Shows phase transitions and critical phenomena
                  </div>
                </div>
              </Card>

              <Card className="p-6">
                <h3 className="text-lg font-semibold mb-4">Harmonic Chain</h3>
                <div className="space-y-3 text-sm">
                  <div><strong>Configuration:</strong> Energy quanta distributed among oscillators</div>
                  <div><strong>Microstates:</strong> Integer partitions of total energy</div>
                  <div><strong>Energy:</strong> E = ℏω(n₁ + n₂ + ... + nN)</div>
                  <div><strong>Applications:</strong> Phonons, molecular vibrations</div>
                  <div className="p-2 bg-muted/30 rounded text-xs">
                    Foundation of solid state physics and lattice dynamics
                  </div>
                </div>
              </Card>

              <Card className="p-6">
                <h3 className="text-lg font-semibold mb-4">Lattice Gas</h3>
                <div className="space-y-3 text-sm">
                  <div><strong>Configuration:</strong> Particles on lattice sites with interactions</div>
                  <div><strong>Microstates:</strong> All occupation patterns</div>
                  <div><strong>Energy:</strong> Kinetic + interaction terms</div>
                  <div><strong>Applications:</strong> Adsorption, liquid-gas transitions</div>
                  <div className="p-2 bg-muted/30 rounded text-xs">
                    Bridge between statistical mechanics and condensed matter
                  </div>
                </div>
              </Card>

              <Card className="p-6">
                <h3 className="text-lg font-semibold mb-4">Information System</h3>
                <div className="space-y-3 text-sm">
                  <div><strong>Configuration:</strong> Symbols from finite alphabet</div>
                  <div><strong>Microstates:</strong> All possible message configurations</div>
                  <div><strong>Information:</strong> I = -log₂(p) bits per symbol</div>
                  <div><strong>Applications:</strong> Data compression, communication</div>
                  <div className="p-2 bg-muted/30 rounded text-xs">
                    Connects physics with information theory and computation
                  </div>
                </div>
              </Card>

              <Card className="p-6">
                <h3 className="text-lg font-semibold mb-4">Statistical Ensembles</h3>
                <div className="space-y-3 text-sm">
                  <div><strong>Microcanonical:</strong> Fixed E, N, V - equal probability</div>
                  <div><strong>Canonical:</strong> Fixed T, N, V - Boltzmann weights</div>
                  <div><strong>Grand Canonical:</strong> Fixed T, μ, V - particle exchange</div>
                  <div className="p-2 bg-muted/30 rounded text-xs">
                    Different constraints lead to different statistical distributions
                  </div>
                </div>
              </Card>
            </div>

            <Card className="p-6">
              <h3 className="text-lg font-semibold mb-4">Entropy Connections</h3>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="p-4 bg-muted/30 rounded-lg">
                  <h4 className="font-medium text-sm mb-2">Thermodynamic Entropy</h4>
                  <div className="text-xs text-muted-foreground">
                    S = ∫ dQ/T (reversible process)
                    <br />
                    Connects to heat and temperature
                    <br />
                    Macroscopic observable
                  </div>
                </div>
                
                <div className="p-4 bg-muted/30 rounded-lg">
                  <h4 className="font-medium text-sm mb-2">Statistical Entropy</h4>
                  <div className="text-xs text-muted-foreground">
                    S = kB ln(Ω)
                    <br />
                    Counts microscopic configurations
                    <br />
                    Links micro and macro
                  </div>
                </div>
                
                <div className="p-4 bg-muted/30 rounded-lg">
                  <h4 className="font-medium text-sm mb-2">Information Entropy</h4>
                  <div className="text-xs text-muted-foreground">
                    H = -Σ p log₂(p)
                    <br />
                    Measures uncertainty/information
                     <br />
                     Basis of information theory
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