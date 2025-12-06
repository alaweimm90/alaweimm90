import React, { useState, useEffect, useMemo, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
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
import { ArrowLeft, Download, Thermometer, Atom, BarChart, Calculator, TrendingUp, Settings, Activity, Target, Layers, Play, Pause } from 'lucide-react';
import { InlineMath, BlockMath } from '@/components/ui/Math';
import { useToast } from '@/hooks/use-toast';
import { PhysicsModuleLayout } from '@/components/PhysicsModuleLayout';
import { PhysicsModuleHeader } from '@/components/PhysicsModuleHeader';
import Plot from 'react-plotly.js';
import { useSEO } from '@/hooks/use-seo';
import { UNIFIED_SCIENTIFIC_PLOT_SYSTEM as PLOT, PHYSICS_AXIS_LABELS } from '@/lib/scientific-plot-system';

type SystemType = 'harmonic' | 'particle_box' | 'rigid_rotor' | 'hydrogen' | 'diatomic' | 'custom';

interface EnergyLevel {
  energy: number;
  degeneracy: number;
  probability: number;
  population: number;
  quantumNumbers?: string;
  description?: string;
}

interface ThermodynamicProperties {
  partitionFunction: number;
  averageEnergy: number;
  heatCapacity: number;
  entropy: number;
  freeEnergy: number;
  variance: number;
  fluctuations: number;
  stdDevEnergy: number;
}

interface TemperatureSweepData {
  temperatures: number[];
  heatCapacities: number[];
  entropies: number[];
  averageEnergies: number[];
  freeEnergies: number[];
  fluctuations: number[];
  magnetizations?: number[];
}

// Enhanced statistical mechanics engine
class StatisticalMechanicsEngine {
  private systemType: SystemType;
  private temperature: number;
  private maxLevels: number;
  private parameters: any;
  
  // Physical constants
  private readonly kB = 8.617e-5; // eV/K
  private readonly hc = 1.24e-4; // eV·cm
  private readonly hbar = 6.582e-16; // eV·s
  private readonly me = 0.511e6; // electron rest mass in eV/c²
  private readonly rydberg = 13.6; // eV
  
  constructor(systemType: SystemType, temperature: number, maxLevels: number, parameters: any) {
    this.systemType = systemType;
    this.temperature = temperature;
    this.maxLevels = maxLevels;
    this.parameters = parameters;
  }
  
  generateEnergyLevels(): EnergyLevel[] {
    switch (this.systemType) {
      case 'harmonic':
        return this.harmonicOscillator();
      case 'particle_box':
        return this.particleInBox();
      case 'rigid_rotor':
        return this.rigidRotor();
      case 'hydrogen':
        return this.hydrogenAtom();
      case 'diatomic':
        return this.diatomicMolecule();
      case 'custom':
        return this.customSystem();
      default:
        return this.harmonicOscillator();
    }
  }
  
  private harmonicOscillator(): EnergyLevel[] {
    const levels: EnergyLevel[] = [];
    const omega = this.parameters.frequency; // cm^-1
    const hbarOmega = this.hc * omega; // eV
    
    for (let n = 0; n < this.maxLevels; n++) {
      const energy = hbarOmega * (n + 0.5);
      levels.push({
        energy,
        degeneracy: 1,
        probability: 0,
        population: 0,
        quantumNumbers: `n=${n}`,
        description: `Vibrational level ${n}`
      });
    }
    return levels;
  }
  
  private particleInBox(): EnergyLevel[] {
    const levels: EnergyLevel[] = [];
    const L = this.parameters.boxLength * 1e-9; // nm to m
    const prefactor = (this.hbar * this.hbar * Math.PI * Math.PI) / (2 * this.me * 1.602e-19 * L * L);
    
    const levelMap = new Map<number, EnergyLevel>();
    
    for (let nx = 1; nx <= Math.ceil(Math.pow(this.maxLevels, 1/3)); nx++) {
      for (let ny = 1; ny <= Math.ceil(Math.pow(this.maxLevels, 1/3)); ny++) {
        for (let nz = 1; nz <= Math.ceil(Math.pow(this.maxLevels, 1/3)); nz++) {
          if (levels.length >= this.maxLevels) break;
          
          const energy = prefactor * (nx*nx + ny*ny + nz*nz);
          const energyKey = Math.round(energy * 1e6);
          
          const quantumNumbers = `(${nx},${ny},${nz})`;
          
          if (levelMap.has(energyKey)) {
            const existingLevel = levelMap.get(energyKey)!;
            existingLevel.degeneracy++;
            existingLevel.quantumNumbers += `, ${quantumNumbers}`;
          } else {
            levelMap.set(energyKey, {
              energy,
              degeneracy: 1,
              probability: 0,
              population: 0,
              quantumNumbers,
              description: `3D box state`
            });
          }
        }
      }
    }
    
    return Array.from(levelMap.values()).sort((a, b) => a.energy - b.energy).slice(0, this.maxLevels);
  }
  
  private rigidRotor(): EnergyLevel[] {
    const levels: EnergyLevel[] = [];
    const I = this.parameters.momentInertia * 1.66e-47; // kg⋅m²
    const B = (this.hbar * this.hbar) / (2 * I * 1.602e-19); // eV
    
    for (let J = 0; J < this.maxLevels; J++) {
      const energy = B * J * (J + 1);
      levels.push({
        energy,
        degeneracy: 2 * J + 1,
        probability: 0,
        population: 0,
        quantumNumbers: `J=${J}`,
        description: `Rotational level J=${J}`
      });
    }
    return levels;
  }
  
  private hydrogenAtom(): EnergyLevel[] {
    const levels: EnergyLevel[] = [];
    
    for (let n = 1; n <= this.maxLevels; n++) {
      const energy = -this.rydberg / (n * n);
      levels.push({
        energy,
        degeneracy: n * n,
        probability: 0,
        population: 0,
        quantumNumbers: `n=${n}`,
        description: `Principal quantum number n=${n}`
      });
    }
    return levels;
  }
  
  private diatomicMolecule(): EnergyLevel[] {
    const levels: EnergyLevel[] = [];
    const omegaE = this.parameters.vibrationalFreq; // cm^-1
    const Be = this.parameters.rotationalConstant; // cm^-1
    const hbarOmegaE = this.hc * omegaE;
    const hbarBe = this.hc * Be;
    
    let levelCount = 0;
    for (let v = 0; v < Math.ceil(Math.sqrt(this.maxLevels)) && levelCount < this.maxLevels; v++) {
      for (let J = 0; J < Math.ceil(Math.sqrt(this.maxLevels)) && levelCount < this.maxLevels; J++) {
        const vibEnergy = hbarOmegaE * (v + 0.5);
        const rotEnergy = hbarBe * J * (J + 1);
        const energy = vibEnergy + rotEnergy;
        
        levels.push({
          energy,
          degeneracy: 2 * J + 1,
          probability: 0,
          population: 0,
          quantumNumbers: `v=${v}, J=${J}`,
          description: `Vibrational-rotational state`
        });
        levelCount++;
      }
    }
    
    return levels.sort((a, b) => a.energy - b.energy).slice(0, this.maxLevels);
  }
  
  private customSystem(): EnergyLevel[] {
    const levels: EnergyLevel[] = [];
    const baseEnergy = this.parameters.baseEnergy || 0;
    const energyScale = this.parameters.energyScale || 1;
    
    for (let i = 0; i < this.maxLevels; i++) {
      const energy = baseEnergy + energyScale * i * i * 0.1;
      levels.push({
        energy,
        degeneracy: i + 1,
        probability: 0,
        population: 0,
        quantumNumbers: `i=${i}`,
        description: `Custom level ${i}`
      });
    }
    return levels;
  }
  
  calculateBoltzmannDistribution(levels: EnergyLevel[]): { levels: EnergyLevel[], thermodynamics: ThermodynamicProperties } {
    const kBT = this.kB * this.temperature;
    
    // Calculate partition function
    let Z = 0;
    levels.forEach(level => {
      Z += level.degeneracy * Math.exp(-level.energy / kBT);
    });
    
    // Calculate probabilities
    levels.forEach(level => {
      const boltzmannFactor = Math.exp(-level.energy / kBT);
      level.probability = (level.degeneracy * boltzmannFactor) / Z;
      level.population = level.probability * 1000; // Normalization
    });
    
    // Calculate thermodynamic properties
    const avgEnergy = levels.reduce((sum, level) => sum + level.energy * level.probability, 0);
    const avgEnergySquared = levels.reduce((sum, level) => sum + level.energy * level.energy * level.probability, 0);
    const variance = avgEnergySquared - avgEnergy * avgEnergy;
    const stdDevEnergy = Math.sqrt(variance);
    
    const heatCapacity = variance / (kBT * kBT);
    const entropy = Math.log(Z) + avgEnergy / kBT;
    const freeEnergy = -kBT * Math.log(Z);
    const fluctuations = Math.sqrt(variance) / Math.abs(avgEnergy);
    
    const thermodynamics: ThermodynamicProperties = {
      partitionFunction: Z,
      averageEnergy: avgEnergy,
      heatCapacity,
      entropy,
      freeEnergy,
      variance,
      fluctuations,
      stdDevEnergy
    };
    
    return { levels, thermodynamics };
  }
}

export default function BoltzmannDistribution() {
  const navigate = useNavigate();
  const { toast } = useToast();
  
  useSEO({ title: 'Quantum Boltzmann Distribution – SimCore', description: 'Analyze Boltzmann populations, partition functions, and thermodynamics across quantum systems.' });
  const schemaLD = {
    '@context': 'https://schema.org',
    '@type': 'TechArticle',
    headline: 'Quantum Boltzmann Distribution Engine',
    description: 'Thermodynamic analysis of quantum systems: energy level populations, partition functions, and temperature dependence.',
    inLanguage: 'en',
    author: { '@type': 'Organization', name: 'SimCore' },
    publisher: { '@type': 'Organization', name: 'SimCore' },
    keywords: 'Boltzmann distribution, partition function, statistical mechanics, thermodynamics',
    about: ['Partition function', 'Energy levels', 'Heat capacity', 'Entropy'],
    mainEntityOfPage: typeof window !== 'undefined' ? window.location.href : ''
  } as const;
  
  // Enhanced state management
  const [activeTab, setActiveTab] = useState('overview');
  const [temperature, setTemperature] = useState([300]); // Kelvin
  const [systemType, setSystemType] = useState<SystemType>('harmonic');
  const [showLogScale, setShowLogScale] = useState(false);
  const [showComparison, setShowComparison] = useState(false);
  const [maxLevels, setMaxLevels] = useState([20]);
  const [isAnimating, setIsAnimating] = useState(false);
  
  // Advanced system parameters
  const [parameters, setParameters] = useState({
    frequency: 1000, // cm^-1 for harmonic oscillator
    boxLength: 1, // nm for particle in box
    momentInertia: 1, // for rigid rotor
    vibrationalFreq: 1000, // cm^-1 for diatomic
    rotationalConstant: 1, // cm^-1 for diatomic
    baseEnergy: 0, // eV for custom
    energyScale: 1 // for custom
  });
  
  // Temperature sweep data
  const [temperatureSweepData, setTemperatureSweepData] = useState<TemperatureSweepData | null>(null);
  const [isRunningAnalysis, setIsRunningAnalysis] = useState(false);
  
  // Statistical mechanics engine
  const engine = useMemo(() => 
    new StatisticalMechanicsEngine(systemType, temperature[0], maxLevels[0], parameters),
    [systemType, temperature, maxLevels, parameters]
  );
  
  // Calculate energy levels and distribution
  const { energyLevels, thermodynamics } = useMemo(() => {
    const levels = engine.generateEnergyLevels();
    const result = engine.calculateBoltzmannDistribution(levels);
    return { energyLevels: result.levels, thermodynamics: result.thermodynamics };
  }, [engine]);
  
  // Temperature sweep analysis
  const runTemperatureSweep = useCallback(async () => {
    setIsRunningAnalysis(true);
    toast({
      title: "Running Temperature Analysis",
      description: "Calculating thermodynamic properties across temperature range..."
    });
    
    const temperatures = Array.from({ length: 100 }, (_, i) => 10 + i * 10); // 10K to 1000K
    const results: TemperatureSweepData = {
      temperatures,
      heatCapacities: [],
      entropies: [],
      averageEnergies: [],
      freeEnergies: [],
      fluctuations: []
    };
    
    for (const T of temperatures) {
      const tempEngine = new StatisticalMechanicsEngine(systemType, T, maxLevels[0], parameters);
      const levels = tempEngine.generateEnergyLevels();
      const result = tempEngine.calculateBoltzmannDistribution(levels);
      
      results.heatCapacities.push(result.thermodynamics.heatCapacity);
      results.entropies.push(result.thermodynamics.entropy);
      results.averageEnergies.push(result.thermodynamics.averageEnergy);
      results.freeEnergies.push(result.thermodynamics.freeEnergy);
      results.fluctuations.push(result.thermodynamics.fluctuations);
    }
    
    setTemperatureSweepData(results);
    setIsRunningAnalysis(false);
    
    toast({
      title: "Analysis Complete",
      description: "Temperature-dependent properties ready for visualization"
    });
  }, [systemType, maxLevels, parameters, toast]);
  
  // Export simulation data
  const exportData = () => {
    const data = {
      systemType,
      parameters,
      temperature: temperature[0],
      energyLevels,
      thermodynamics,
      temperatureSweepData,
      timestamp: new Date().toISOString()
    };
    
    const dataStr = JSON.stringify(data, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'boltzmann_distribution_analysis.csv';
    link.click();
  };
  
  // Animation effect for population visualization
  useEffect(() => {
    if (isAnimating) {
      const interval = setInterval(() => {
        setTemperature(prev => {
          const newTemp = prev[0] + (Math.random() - 0.5) * 10;
          return [Math.max(10, Math.min(1000, newTemp))];
        });
      }, 100);
      
      return () => clearInterval(interval);
    }
  }, [isAnimating]);
  
  // Get system-specific parameter controls
  const getParameterControls = () => {
    switch (systemType) {
      case 'harmonic':
        return (
          <div className="space-y-4">
            <div className="space-y-2">
              <Label className="text-sm font-medium">
                Frequency: {parameters.frequency} cm⁻¹
              </Label>
              <Slider
                value={[parameters.frequency]}
                onValueChange={(value) => setParameters(prev => ({ ...prev, frequency: value[0] }))}
                min={100}
                max={3000}
                step={50}
                className="w-full"
              />
            </div>
            <div className="text-xs text-muted-foreground">
              <InlineMath math="E_n = \hbar\omega(n + 1/2)" />
            </div>
          </div>
        );
      
      case 'particle_box':
        return (
          <div className="space-y-4">
            <div className="space-y-2">
              <Label className="text-sm font-medium">
                Box Length: {parameters.boxLength} nm
              </Label>
              <Slider
                value={[parameters.boxLength]}
                onValueChange={(value) => setParameters(prev => ({ ...prev, boxLength: value[0] }))}
                min={0.1}
                max={5}
                step={0.1}
                className="w-full"
              />
            </div>
            <div className="text-xs text-muted-foreground">
              <InlineMath math="E_{n_x,n_y,n_z} = \frac{\hbar^2\pi^2}{2mL^2}(n_x^2 + n_y^2 + n_z^2)" />
            </div>
          </div>
        );
      
      case 'rigid_rotor':
        return (
          <div className="space-y-4">
            <div className="space-y-2">
              <Label className="text-sm font-medium">
                Moment of Inertia: {parameters.momentInertia}
              </Label>
              <Slider
                value={[parameters.momentInertia]}
                onValueChange={(value) => setParameters(prev => ({ ...prev, momentInertia: value[0] }))}
                min={0.1}
                max={10}
                step={0.1}
                className="w-full"
              />
            </div>
            <div className="text-xs text-muted-foreground">
              <InlineMath math="E_J = BJ(J+1)" />
            </div>
          </div>
        );
      
      case 'hydrogen':
        return (
          <div className="space-y-2">
            <div className="text-sm text-muted-foreground">
              Hydrogen atom energy levels (fixed parameters)
            </div>
            <div className="text-xs text-muted-foreground">
              <InlineMath math="E_n = -\frac{13.6 \text{ eV}}{n^2}" />
            </div>
          </div>
        );
      
      case 'diatomic':
        return (
          <div className="space-y-4">
            <div className="space-y-2">
              <Label className="text-sm font-medium">
                Vibrational Frequency: {parameters.vibrationalFreq} cm⁻¹
              </Label>
              <Slider
                value={[parameters.vibrationalFreq]}
                onValueChange={(value) => setParameters(prev => ({ ...prev, vibrationalFreq: value[0] }))}
                min={200}
                max={4000}
                step={100}
                className="w-full"
              />
            </div>
            <div className="space-y-2">
              <Label className="text-sm font-medium">
                Rotational Constant: {parameters.rotationalConstant} cm⁻¹
              </Label>
              <Slider
                value={[parameters.rotationalConstant]}
                onValueChange={(value) => setParameters(prev => ({ ...prev, rotationalConstant: value[0] }))}
                min={0.1}
                max={10}
                step={0.1}
                className="w-full"
              />
            </div>
            <div className="text-xs text-muted-foreground">
              <InlineMath math="E_{v,J} = \hbar\omega_e(v + 1/2) + BJ(J+1)" />
            </div>
          </div>
        );
      
      case 'custom':
        return (
          <div className="space-y-4">
            <div className="space-y-2">
              <Label className="text-sm font-medium">
                Base Energy: {parameters.baseEnergy} eV
              </Label>
              <Slider
                value={[parameters.baseEnergy]}
                onValueChange={(value) => setParameters(prev => ({ ...prev, baseEnergy: value[0] }))}
                min={-2}
                max={2}
                step={0.1}
                className="w-full"
              />
            </div>
            <div className="space-y-2">
              <Label className="text-sm font-medium">
                Energy Scale: {parameters.energyScale}
              </Label>
              <Slider
                value={[parameters.energyScale]}
                onValueChange={(value) => setParameters(prev => ({ ...prev, energyScale: value[0] }))}
                min={0.1}
                max={5}
                step={0.1}
                className="w-full"
              />
            </div>
            <div className="text-xs text-muted-foreground">
              <InlineMath math="E_i = E_0 + \alpha \cdot i^2" />
            </div>
          </div>
        );
      
      default:
        return null;
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-background via-background to-muted/20">
      <script type="application/ld+json" dangerouslySetInnerHTML={{ __html: JSON.stringify(schemaLD) }} />
      <div className="container mx-auto px-4 py-6">
        <PhysicsModuleHeader
          title="Quantum Boltzmann Distribution Engine"
          description="Statistical mechanics and thermodynamic analysis of quantum energy systems. Explore energy level populations, partition functions, and thermal properties across different quantum models."
          category="Statistical Physics"
          difficulty="Advanced"
          equation="P_n = \frac{g_n e^{-E_n/k_BT}}{Z}"
          onExport={exportData}
          onReset={() => {
            setTemperature([300]);
            setParameters({
              frequency: 1000,
              boxLength: 1,
              momentInertia: 1,
              vibrationalFreq: 1000,
              rotationalConstant: 1,
              baseEnergy: 0,
              energyScale: 1
            });
          }}
          isRunning={isAnimating}
        />

        {/* Main Content */}
        <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
          <TabsList className="grid w-full grid-cols-5">
            <TabsTrigger value="overview" className="gap-2">
              <Atom className="h-4 w-4" />
              Overview
            </TabsTrigger>
            <TabsTrigger value="thermodynamics" className="gap-2">
              <TrendingUp className="h-4 w-4" />
              Thermodynamics
            </TabsTrigger>
            <TabsTrigger value="math" className="gap-2">
              <Calculator className="h-4 w-4" />
              Mathematics
            </TabsTrigger>
            <TabsTrigger value="systems" className="gap-2">
              <Layers className="h-4 w-4" />
              Quantum Systems
            </TabsTrigger>
            <TabsTrigger value="applications" className="gap-2">
              <Target className="h-4 w-4" />
              Applications
            </TabsTrigger>
          </TabsList>

          {/* Overview Tab */}
          <TabsContent value="overview" className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              {/* Controls Panel */}
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Settings className="h-5 w-5" />
                    System Controls
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-6">
                  {/* System Type Selection */}
                  <div className="space-y-2">
                    <Label className="text-sm font-medium">Quantum System</Label>
                    <Select value={systemType} onValueChange={(value: SystemType) => setSystemType(value)}>
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="harmonic">Harmonic Oscillator</SelectItem>
                        <SelectItem value="particle_box">Particle in Box</SelectItem>
                        <SelectItem value="rigid_rotor">Rigid Rotor</SelectItem>
                        <SelectItem value="hydrogen">Hydrogen Atom</SelectItem>
                        <SelectItem value="diatomic">Diatomic Molecule</SelectItem>
                        <SelectItem value="custom">Custom System</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>

                  {/* Temperature Control */}
                  <div className="space-y-2">
                    <Label className="text-sm font-medium">
                      Temperature: {temperature[0]} K
                    </Label>
                    <div className="text-xs text-muted-foreground mb-2">
                      k<sub>B</sub>T = {(8.617e-5 * temperature[0]).toFixed(4)} eV
                    </div>
                    <Slider
                      value={temperature}
                      onValueChange={setTemperature}
                      min={10}
                      max={1000}
                      step={5}
                      className="w-full"
                    />
                  </div>

                  {/* Max Levels */}
                  <div className="space-y-2">
                    <Label className="text-sm font-medium">
                      Max Levels: {maxLevels[0]}
                    </Label>
                    <Slider
                      value={maxLevels}
                      onValueChange={setMaxLevels}
                      min={5}
                      max={50}
                      step={1}
                      className="w-full"
                    />
                  </div>

                  {/* System-specific parameters */}
                  {getParameterControls()}

                  <Separator />

                  {/* Display Options */}
                  <div className="space-y-4">
                    <div className="flex items-center justify-between">
                      <Label className="text-sm">Logarithmic Scale</Label>
                      <Switch
                        checked={showLogScale}
                        onCheckedChange={setShowLogScale}
                      />
                    </div>
                    <div className="flex items-center justify-between">
                      <Label className="text-sm">Show Classical Limit</Label>
                      <Switch
                        checked={showComparison}
                        onCheckedChange={setShowComparison}
                      />
                    </div>
                  </div>
                </CardContent>
              </Card>

              {/* Population Distribution */}
              <div className="lg:col-span-2">
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <BarChart className="h-5 w-5" />
                      Boltzmann Population Distribution
                    </CardTitle>
                    <CardDescription>
                      Energy level populations: <InlineMath math="P_n = \frac{g_n e^{-E_n/k_BT}}{Z}" />
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="h-96">
                      <Plot
                        data={[
                          {
                            x: energyLevels.map((_, i) => i),
                            y: energyLevels.map(level => level.probability),
                            type: 'bar',
                            name: 'Probability',
                            marker: {
                              color: energyLevels.map(level => `rgba(59, 130, 246, ${level.probability / Math.max(...energyLevels.map(l => l.probability))})`),
                              line: { color: 'rgb(59, 130, 246)', width: 1 }
                            },
                            hovertemplate: 
                              '<b>Level %{x}</b><br>' +
                              'Energy: %{customdata[0]:.4f} eV<br>' +
                              'Probability: %{y:.6f}<br>' +
                              'Degeneracy: %{customdata[1]}<br>' +
                              'Quantum Numbers: %{customdata[2]}<br>' +
                              '<extra></extra>',
                            customdata: energyLevels.map(level => [
                              level.energy,
                              level.degeneracy,
                              level.quantumNumbers
                            ])
                          }
                        ]}
                        layout={{
                          ...PLOT.createLayout('Energy Level Population', 'Energy Level Index (n)', 'Probability P(n)', { showLegend: false, height: 400 }),
                          yaxis: { ...PLOT.layout.yaxis, title: 'Probability P(n)', type: showLogScale ? 'log' : 'linear' }
                        }}
                        config={PLOT.config}
                        style={{ width: '100%', height: '100%' }}
                      />
                    </div>
                  </CardContent>
                </Card>
              </div>
            </div>

            {/* Thermodynamic Properties Cards */}
            <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-7 gap-4">
              <Card>
                <CardContent className="p-4 text-center">
                  <div className="text-2xl font-bold text-blue-500">
                    {thermodynamics.partitionFunction.toFixed(2)}
                  </div>
                  <div className="text-sm text-muted-foreground">
                    Partition Function (Z)
                  </div>
                </CardContent>
              </Card>
              
              <Card>
                <CardContent className="p-4 text-center">
                  <div className="text-2xl font-bold text-green-500">
                    {thermodynamics.averageEnergy.toFixed(4)}
                  </div>
                  <div className="text-sm text-muted-foreground">
                    ⟨E⟩ (eV)
                  </div>
                </CardContent>
              </Card>
              
              <Card>
                <CardContent className="p-4 text-center">
                  <div className="text-2xl font-bold text-purple-500">
                    {thermodynamics.heatCapacity.toFixed(4)}
                  </div>
                  <div className="text-sm text-muted-foreground">
                    C<sub>v</sub> (k<sub>B</sub>)
                  </div>
                </CardContent>
              </Card>
              
              <Card>
                <CardContent className="p-4 text-center">
                  <div className="text-2xl font-bold text-red-500">
                    {thermodynamics.entropy.toFixed(4)}
                  </div>
                  <div className="text-sm text-muted-foreground">
                    S (k<sub>B</sub>)
                  </div>
                </CardContent>
              </Card>
              
              <Card>
                <CardContent className="p-4 text-center">
                  <div className="text-2xl font-bold text-orange-500">
                    {thermodynamics.freeEnergy.toFixed(4)}
                  </div>
                  <div className="text-sm text-muted-foreground">
                    F (eV)
                  </div>
                </CardContent>
              </Card>
              
              <Card>
                <CardContent className="p-4 text-center">
                  <div className="text-2xl font-bold text-cyan-500">
                    {thermodynamics.stdDevEnergy.toFixed(4)}
                  </div>
                  <div className="text-sm text-muted-foreground">
                    σ<sub>E</sub> (eV)
                  </div>
                </CardContent>
              </Card>
              
              <Card>
                <CardContent className="p-4 text-center">
                  <div className="text-2xl font-bold text-yellow-500">
                    {(thermodynamics.fluctuations * 100).toFixed(1)}%
                  </div>
                  <div className="text-sm text-muted-foreground">
                    Fluctuations
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          {/* Thermodynamics Tab */}
          <TabsContent value="thermodynamics" className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <Card>
                <CardHeader>
                  <CardTitle>Temperature Sweep Analysis</CardTitle>
                  <CardDescription>
                    Calculate thermodynamic properties across temperature range
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <Button 
                    onClick={runTemperatureSweep} 
                    disabled={isRunningAnalysis}
                    className="w-full"
                  >
                    {isRunningAnalysis ? 'Running Analysis...' : 'Run Temperature Sweep'}
                  </Button>
                  
                  {isRunningAnalysis && (
                    <Progress value={50} className="w-full" />
                  )}
                  
                  {temperatureSweepData && (
                    <Alert>
                      <AlertDescription>
                        Analysis complete for {temperatureSweepData.temperatures.length} temperature points.
                        View plots below.
                      </AlertDescription>
                    </Alert>
                  )}
                </CardContent>
              </Card>

              {temperatureSweepData && (
                <Card>
                  <CardHeader>
                    <CardTitle>Heat Capacity vs Temperature</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="h-64">
                      <Plot
                        data={[
                          {
                            x: temperatureSweepData.temperatures,
                            y: temperatureSweepData.heatCapacities,
                            type: 'scatter',
                            mode: 'lines',
                            name: 'Cv',
                            line: { color: 'rgb(168, 85, 247)', width: 2 }
                          }
                        ]}
                        layout={PLOT.createLayout('Heat Capacity', 'Temperature (K)', 'Cv (kB)')}
                        config={PLOT.config}
                        style={{ width: '100%', height: '100%' }}
                      />
                    </div>
                  </CardContent>
                </Card>
              )}
            </div>

            {temperatureSweepData && (
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <Card>
                  <CardHeader>
                    <CardTitle>Average Energy vs Temperature</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="h-64">
                      <Plot
                        data={[
                          {
                            x: temperatureSweepData.temperatures,
                            y: temperatureSweepData.averageEnergies,
                            type: 'scatter',
                            mode: 'lines',
                            name: '⟨E⟩',
                            line: { color: 'rgb(34, 197, 94)', width: 2 }
                          }
                        ]}
                        layout={PLOT.createLayout('Average Energy', 'Temperature (K)', '⟨E⟩ (eV)')}
                        config={PLOT.config}
                        style={{ width: '100%', height: '100%' }}
                      />
                    </div>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader>
                    <CardTitle>Entropy vs Temperature</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="h-64">
                      <Plot
                        data={[
                          {
                            x: temperatureSweepData.temperatures,
                            y: temperatureSweepData.entropies,
                            type: 'scatter',
                            mode: 'lines',
                            name: 'S',
                            line: { color: 'rgb(239, 68, 68)', width: 2 }
                          }
                        ]}
                        layout={PLOT.createLayout('Entropy', 'Temperature (K)', 'S (kB)')}
                        config={PLOT.config}
                        style={{ width: '100%', height: '100%' }}
                      />
                    </div>
                  </CardContent>
                </Card>
              </div>
            )}
          </TabsContent>

          {/* Mathematics Tab */}
          <TabsContent value="math" className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <Card>
                <CardHeader>
                  <CardTitle>Fundamental Equations</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div>
                    <h4 className="font-medium mb-2">Partition Function</h4>
                    <BlockMath math="Z = \sum_i g_i e^{-E_i/k_B T}" />
                    <p className="text-sm text-muted-foreground mt-2">
                      Sum over all energy states with degeneracy g<sub>i</sub>
                    </p>
                  </div>
                  
                  <Separator />
                  
                  <div>
                    <h4 className="font-medium mb-2">Boltzmann Probability</h4>
                    <BlockMath math="P_i = \frac{g_i e^{-E_i/k_B T}}{Z}" />
                    <p className="text-sm text-muted-foreground mt-2">
                      Probability of finding the system in state i
                    </p>
                  </div>
                  
                  <Separator />
                  
                  <div>
                    <h4 className="font-medium mb-2">Average Energy</h4>
                    <BlockMath math="\langle E \rangle = \sum_i E_i P_i = -\frac{\partial \ln Z}{\partial \beta}" />
                    <p className="text-sm text-muted-foreground mt-2">
                      Where β = 1/(k<sub>B</sub>T)
                    </p>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Thermodynamic Properties</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div>
                    <h4 className="font-medium mb-2">Heat Capacity</h4>
                    <BlockMath math="C_V = \frac{\partial \langle E \rangle}{\partial T} = \frac{\langle E^2 \rangle - \langle E \rangle^2}{k_B T^2}" />
                    <p className="text-sm text-muted-foreground mt-2">
                      Related to energy fluctuations
                    </p>
                  </div>
                  
                  <Separator />
                  
                  <div>
                    <h4 className="font-medium mb-2">Entropy</h4>
                    <BlockMath math="S = k_B \ln Z + \frac{\langle E \rangle}{T}" />
                    <p className="text-sm text-muted-foreground mt-2">
                      Measure of thermal disorder
                    </p>
                  </div>
                  
                  <Separator />
                  
                  <div>
                    <h4 className="font-medium mb-2">Free Energy</h4>
                    <BlockMath math="F = \langle E \rangle - TS = -k_B T \ln Z" />
                    <p className="text-sm text-muted-foreground mt-2">
                      Helmholtz free energy
                    </p>
                  </div>
                  
                  <Separator />
                  
                  <div>
                    <h4 className="font-medium mb-2">Energy Fluctuations</h4>
                    <BlockMath math="\sigma_E^2 = \langle E^2 \rangle - \langle E \rangle^2 = k_B T^2 C_V" />
                    <p className="text-sm text-muted-foreground mt-2">
                      Variance in energy
                    </p>
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          {/* Quantum Systems Tab */}
          <TabsContent value="systems" className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              <Card>
                <CardHeader>
                  <CardTitle>Harmonic Oscillator</CardTitle>
                </CardHeader>
                <CardContent className="space-y-2">
                  <BlockMath math="E_n = \hbar\omega(n + 1/2)" />
                  <div className="space-y-1 text-sm">
                    <Badge variant="outline">g<sub>n</sub> = 1</Badge>
                    <p>Equally spaced energy levels</p>
                    <p>Models molecular vibrations</p>
                    <p>Zero-point energy ℏω/2</p>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Particle in a Box</CardTitle>
                </CardHeader>
                <CardContent className="space-y-2">
                  <BlockMath math="E_{n_x,n_y,n_z} = \frac{\hbar^2\pi^2}{2mL^2}(n_x^2 + n_y^2 + n_z^2)" />
                  <div className="space-y-1 text-sm">
                    <Badge variant="outline">Variable degeneracy</Badge>
                    <p>Quantum confinement</p>
                    <p>Energy ∝ n²</p>
                    <p>Models quantum dots</p>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Rigid Rotor</CardTitle>
                </CardHeader>
                <CardContent className="space-y-2">
                  <BlockMath math="E_J = BJ(J+1)" />
                  <div className="space-y-1 text-sm">
                    <Badge variant="outline">g<sub>J</sub> = 2J+1</Badge>
                    <p>Rotational motion</p>
                    <p>Increasing degeneracy</p>
                    <p>Models molecular rotation</p>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Hydrogen Atom</CardTitle>
                </CardHeader>
                <CardContent className="space-y-2">
                  <BlockMath math="E_n = -\frac{13.6 \text{ eV}}{n^2}" />
                  <div className="space-y-1 text-sm">
                    <Badge variant="outline">g<sub>n</sub> = n²</Badge>
                    <p>Bound electronic states</p>
                    <p>Negative energies</p>
                    <p>High degeneracy for large n</p>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Diatomic Molecule</CardTitle>
                </CardHeader>
                <CardContent className="space-y-2">
                  <BlockMath math="E_{v,J} = \hbar\omega_e(v + 1/2) + BJ(J+1)" />
                  <div className="space-y-1 text-sm">
                    <Badge variant="outline">g<sub>v,J</sub> = 2J+1</Badge>
                    <p>Vibration + rotation</p>
                    <p>Realistic molecules</p>
                    <p>Spectroscopic transitions</p>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Custom System</CardTitle>
                </CardHeader>
                <CardContent className="space-y-2">
                  <BlockMath math="E_i = E_0 + \alpha \cdot i^2" />
                  <div className="space-y-1 text-sm">
                    <Badge variant="outline">Adjustable parameters</Badge>
                    <p>User-defined energy levels</p>
                    <p>Variable degeneracy</p>
                    <p>Exploration tool</p>
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          {/* Applications Tab */}
          <TabsContent value="applications" className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <Card>
                <CardHeader>
                  <CardTitle>Physical Applications</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div>
                    <h4 className="font-medium mb-2">Molecular Spectroscopy</h4>
                    <ul className="text-sm space-y-1 text-muted-foreground">
                      <li>• IR absorption lines</li>
                      <li>• Population ratios at different temperatures</li>
                      <li>• Intensity calculations</li>
                      <li>• Thermal broadening effects</li>
                    </ul>
                  </div>
                  
                  <Separator />
                  
                  <div>
                    <h4 className="font-medium mb-2">Specific Heat of Solids</h4>
                    <ul className="text-sm space-y-1 text-muted-foreground">
                      <li>• Einstein model (harmonic oscillators)</li>
                      <li>• Debye model corrections</li>
                      <li>• Temperature dependence of C<sub>V</sub></li>
                      <li>• Classical limit at high T</li>
                    </ul>
                  </div>
                  
                  <Separator />
                  
                  <div>
                    <h4 className="font-medium mb-2">Stellar Atmospheres</h4>
                    <ul className="text-sm space-y-1 text-muted-foreground">
                      <li>• Excitation of atomic levels</li>
                      <li>• Line strength calculations</li>
                      <li>• Temperature determination</li>
                      <li>• Opacity calculations</li>
                    </ul>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Technological Applications</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div>
                    <h4 className="font-medium mb-2">Laser Physics</h4>
                    <ul className="text-sm space-y-1 text-muted-foreground">
                      <li>• Population inversion</li>
                      <li>• Thermal equilibrium vs. pumping</li>
                      <li>• Threshold conditions</li>
                      <li>• Temperature effects on efficiency</li>
                    </ul>
                  </div>
                  
                  <Separator />
                  
                  <div>
                    <h4 className="font-medium mb-2">Quantum Computing</h4>
                    <ul className="text-sm space-y-1 text-muted-foreground">
                      <li>• Thermal noise in qubits</li>
                      <li>• Decoherence rates</li>
                      <li>• Cooling requirements</li>
                      <li>• Error correction thresholds</li>
                    </ul>
                  </div>
                  
                  <Separator />
                  
                  <div>
                    <h4 className="font-medium mb-2">Materials Science</h4>
                    <ul className="text-sm space-y-1 text-muted-foreground">
                      <li>• Thermal properties prediction</li>
                      <li>• Phase transition analysis</li>
                      <li>• Magnetic susceptibility</li>
                      <li>• Electronic structure effects</li>
                    </ul>
                  </div>
                </CardContent>
              </Card>
            </div>

            <Card>
              <CardHeader>
                <CardTitle>Educational Impact</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div>
                    <h4 className="font-medium mb-2">Conceptual Understanding</h4>
                    <ul className="text-sm space-y-1 text-muted-foreground">
                      <li>• Statistical mechanics principles</li>
                      <li>• Quantum-classical correspondence</li>
                      <li>• Thermal equilibrium concepts</li>
                      <li>• Energy level population dynamics</li>
                    </ul>
                  </div>
                  
                  <div>
                    <h4 className="font-medium mb-2">Mathematical Skills</h4>
                    <ul className="text-sm space-y-1 text-muted-foreground">
                      <li>• Exponential functions</li>
                      <li>• Probability distributions</li>
                      <li>• Thermodynamic derivatives</li>
                      <li>• Series expansions</li>
                    </ul>
                  </div>
                  
                  <div>
                    <h4 className="font-medium mb-2">Computational Thinking</h4>
                    <ul className="text-sm space-y-1 text-muted-foreground">
                      <li>• Parameter sensitivity analysis</li>
                      <li>• Limit behavior exploration</li>
                      <li>• Approximation methods</li>
                      <li>• Data visualization techniques</li>
                    </ul>
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
}