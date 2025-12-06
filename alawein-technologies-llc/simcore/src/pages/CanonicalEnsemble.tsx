import React, { useState, useEffect, useMemo, useCallback } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Slider } from '@/components/ui/slider';
import { Badge } from '@/components/ui/badge';
import { Switch } from '@/components/ui/switch';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import Plot from 'react-plotly.js';
import { PhysicsTheory } from '@/components/PhysicsTheory';
import { PerformanceBenchmark } from '@/components/PerformanceBenchmark';
import { Thermometer, Activity, TrendingUp, Calculator, BarChart3, Target, Zap, Download, Play, BookOpen, AlertTriangle } from 'lucide-react';
import { InlineMath, BlockMath } from '@/components/ui/Math';
import { useToast } from '@/hooks/use-toast';

import { PhysicsModuleLayout } from '@/components/PhysicsModuleLayout';
import { PhysicsModuleHeader } from '@/components/PhysicsModuleHeader';
import { useSEO } from '@/hooks/use-seo';
import { UNIFIED_SCIENTIFIC_PLOT_SYSTEM as PLOT } from '@/lib/scientific-plot-system';

type SystemType = 'ideal_gas' | 'harmonic_oscillator' | 'spin_system' | 'einstein_solid' | 'fermi_gas' | 'bose_gas' | 'photon_gas' | 'paramagnet';

interface EnsembleData {
  temperature: number;
  averageEnergy: number;
  heatCapacity: number;
  entropy: number;
  partitionFunction: number;
  freeEnergy: number;
  fluctuation: number;
  pressure?: number;
  chemicalPotential?: number;
  compressibility?: number;
  magnetization?: number;
  susceptibility?: number;
}

interface ThermodynamicDerivative {
  temperature: number;
  value: number;
  derivative: number;
  secondDerivative: number;
}

interface FluctuationData {
  temperature: number;
  energyFluctuation: number;
  relativeFluctuation: number;
  correlationLength: number;
}

export default function CanonicalEnsemble() {
  useSEO({ title: 'Canonical Ensemble Simulator – SimCore', description: 'Thermodynamics in the canonical ensemble with fluctuations and temperature sweeps.' });
  const schemaLD = {
    '@context': 'https://schema.org',
    '@type': 'TechArticle',
    headline: 'Canonical Ensemble Simulator',
    description: 'Statistical mechanics of systems at constant temperature with comprehensive thermodynamic analysis.',
    inLanguage: 'en',
    author: { '@type': 'Organization', name: 'SimCore' },
    publisher: { '@type': 'Organization', name: 'SimCore' },
    keywords: 'canonical ensemble, partition function, heat capacity, entropy, fluctuations',
    about: ['Canonical ensemble', 'Partition function', 'Thermodynamic derivatives'],
    mainEntityOfPage: typeof window !== 'undefined' ? window.location.href : ''
  } as const;
  const [systemType, setSystemType] = useState<SystemType>('ideal_gas');
  const [temperature, setTemperature] = useState([300]);
  const [systemSize, setSystemSize] = useState([100]);
  const [systemParameter, setSystemParameter] = useState([1]);
  const [volume, setVolume] = useState([1e-21]); // m³
  const [mass, setMass] = useState([9.109e-31]); // kg
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [showFluctuations, setShowFluctuations] = useState(true);
  const [temperatureRange, setTemperatureRange] = useState([50, 800]);
  const [analysisMode, setAnalysisMode] = useState<'standard' | 'derivatives' | 'comparisons'>('standard');
  const [isRunning, setIsRunning] = useState(false);
  
  const [ensembleData, setEnsembleData] = useState<EnsembleData[]>([]);
  const [derivativeData, setDerivativeData] = useState<ThermodynamicDerivative[]>([]);
  const [fluctuationData, setFluctuationData] = useState<FluctuationData[]>([]);
  const { toast } = useToast();

  // Physical constants
  const kB = 1.381e-23; // J/K
  const kB_eV = 8.617e-5; // eV/K
  const hbar = 1.055e-34; // J⋅s
  const NA = 6.022e23; // Avogadro's number

  // Helper functions - defined before use
  const factorial = useCallback((n: number): number => {
    if (n <= 1) return 1;
    if (n > 170) return Infinity;
    let result = 1;
    for (let i = 2; i <= n; i++) {
      result *= i;
    }
    return result;
  }, []);

  // Advanced ensemble calculation engine
  const calculateAdvancedEnsemble = useCallback((T: number, N: number, param: number): EnsembleData => {
    const beta = 1 / (kB_eV * T);
    let result: EnsembleData;

    switch (systemType) {
      case 'ideal_gas': {
        // 3D classical ideal gas with proper thermodynamics
        const V = volume[0];
        const m = mass[0];
        const thermalWavelength = Math.sqrt(2 * Math.PI * hbar * hbar / (m * kB * T));
        const Z_single = V / Math.pow(thermalWavelength, 3);
        const Z = Math.pow(Z_single, N) / factorial(N);
        
        result = {
          temperature: T,
          averageEnergy: 1.5 * N * kB_eV * T,
          heatCapacity: 1.5 * N * kB_eV,
          entropy: N * kB_eV * (Math.log(Z_single / N) + 2.5),
          partitionFunction: Z,
          freeEnergy: -kB_eV * T * Math.log(Z),
          fluctuation: Math.sqrt(1.5 * N * Math.pow(kB_eV * T, 2)),
          pressure: N * kB * T / V * 1e-9, // GPa
          compressibility: 1 / (N * kB * T / V * 1e-9)
        };
        break;
      }

      case 'harmonic_oscillator': {
        // Quantum harmonic oscillator ensemble
        const omega = param * 1e12;
        const hbarOmega_eV = hbar * omega / 1.602e-19;
        const x = hbarOmega_eV / (kB_eV * T);
        const n_avg = 1 / (Math.exp(x) - 1);
        
        result = {
          temperature: T,
          averageEnergy: N * hbarOmega_eV * (n_avg + 0.5),
          heatCapacity: N * kB_eV * x * x * Math.exp(x) / Math.pow(Math.exp(x) - 1, 2),
          entropy: N * kB_eV * (x * n_avg - Math.log(1 - Math.exp(-x))),
          partitionFunction: Math.pow(1 / (2 * Math.sinh(x / 2)), N),
          freeEnergy: N * hbarOmega_eV / 2 - N * kB_eV * T * Math.log(1 / (2 * Math.sinh(x / 2))),
          fluctuation: Math.sqrt(N * hbarOmega_eV * hbarOmega_eV * n_avg * (n_avg + 1))
        };
        break;
      }

      case 'spin_system': {
        // Magnetic spin system with field dependence
        const muB = 5.788e-5; // eV/T
        const B = param;
        const epsilon = muB * B;
        const m_avg = Math.tanh(epsilon * beta);
        
        result = {
          temperature: T,
          averageEnergy: -N * epsilon * m_avg,
          heatCapacity: N * kB_eV * Math.pow(epsilon * beta, 2) / Math.pow(Math.cosh(epsilon * beta), 2),
          entropy: N * kB_eV * (Math.log(2 * Math.cosh(epsilon * beta)) - epsilon * beta * m_avg),
          partitionFunction: Math.pow(2 * Math.cosh(epsilon * beta), N),
          freeEnergy: -N * kB_eV * T * Math.log(2 * Math.cosh(epsilon * beta)),
          fluctuation: Math.sqrt(N * epsilon * epsilon * (1 - m_avg * m_avg)),
          magnetization: N * muB * m_avg,
          susceptibility: N * muB * muB * beta * (1 - m_avg * m_avg)
        };
        break;
      }

      case 'einstein_solid': {
        // Einstein solid model with proper phonon statistics
        const thetaE = param * 100;
        const y = thetaE / T;
        const n_phonon = 1 / (Math.exp(y) - 1);
        
        result = {
          temperature: T,
          averageEnergy: 3 * N * kB_eV * thetaE * n_phonon,
          heatCapacity: 3 * N * kB_eV * y * y * Math.exp(y) / Math.pow(Math.exp(y) - 1, 2),
          entropy: 3 * N * kB_eV * (y * n_phonon - Math.log(1 - Math.exp(-y))),
          partitionFunction: Math.pow(1 / (1 - Math.exp(-y)), 3 * N),
          freeEnergy: 3 * N * kB_eV * T * Math.log(1 - Math.exp(-y)),
          fluctuation: Math.sqrt(3 * N * Math.pow(kB_eV * thetaE, 2) * n_phonon * (n_phonon + 1))
        };
        break;
      }

      case 'fermi_gas': {
        // Fermi gas at finite temperature
        const EF = param * kB_eV * 300; // Fermi energy in eV
        const fermiFunc = (E: number) => 1 / (Math.exp((E - EF) / (kB_eV * T)) + 1);
        const avgE = 0.6 * EF * (1 + 5 * Math.pow(Math.PI * kB_eV * T / (12 * EF), 2));
        
        result = {
          temperature: T,
          averageEnergy: N * avgE,
          heatCapacity: N * kB_eV * Math.PI * Math.PI * kB_eV * T / (2 * EF),
          entropy: N * kB_eV * Math.PI * Math.PI * kB_eV * T / (3 * EF),
          partitionFunction: Math.exp(N * EF / (kB_eV * T)), // Approximation
          freeEnergy: N * (avgE - T * kB_eV * Math.PI * Math.PI * kB_eV * T / (3 * EF)),
          fluctuation: Math.sqrt(N * avgE * kB_eV * T),
          chemicalPotential: EF * (1 - Math.pow(Math.PI * kB_eV * T / (12 * EF), 2))
        };
        break;
      }

      case 'bose_gas': {
        // Ideal Bose gas
        const g = 2; // degeneracy
        const n_BE = g / (Math.exp(kB_eV * T * beta) - 1);
        
        result = {
          temperature: T,
          averageEnergy: N * 1.5 * kB_eV * T,
          heatCapacity: N * 1.5 * kB_eV,
          entropy: N * kB_eV * (2.5 - Math.log(N / (g * Math.pow(2 * Math.PI * mass[0] * kB * T / (hbar * hbar), 1.5) * volume[0]))),
          partitionFunction: Math.exp(N * 1.5),
          freeEnergy: -N * kB_eV * T * (2.5 - Math.log(N)),
          fluctuation: Math.sqrt(N * 1.5 * Math.pow(kB_eV * T, 2))
        };
        break;
      }

      case 'photon_gas': {
        // Blackbody radiation
        const sigma = 5.67e-8; // Stefan-Boltzmann constant
        const a = 7.57e-16; // Radiation constant J/(m³⋅K⁴)
        
        result = {
          temperature: T,
          averageEnergy: a * volume[0] * Math.pow(T, 4) / 1.602e-19, // Convert to eV
          heatCapacity: 4 * a * volume[0] * Math.pow(T, 3) / 1.602e-19,
          entropy: (4/3) * a * volume[0] * Math.pow(T, 3) / 1.602e-19,
          partitionFunction: Math.exp(a * volume[0] * Math.pow(T, 4) / (3 * 1.602e-19)),
          freeEnergy: -a * volume[0] * Math.pow(T, 4) / (3 * 1.602e-19),
          fluctuation: Math.sqrt(a * volume[0] * Math.pow(T, 4) / 1.602e-19),
          pressure: a * Math.pow(T, 4) / 3 * 1e-9 // GPa
        };
        break;
      }

      case 'paramagnet': {
        // Paramagnetic system with J > 1/2
        const J = param;
        const muB_param = 5.788e-5;
        const gJ = 2; // Landé g-factor
        const B_field = 1; // Tesla
        const muJ = gJ * muB_param * Math.sqrt(J * (J + 1));
        const x_param = muJ * B_field * beta;
        const brillouin = (2 * J + 1) / (2 * J) * (1 / Math.tanh((2 * J + 1) * x_param / (2 * J))) - 1 / (2 * J) * (1 / Math.tanh(x_param / (2 * J)));
        
        result = {
          temperature: T,
          averageEnergy: -N * muJ * B_field * brillouin,
          heatCapacity: N * kB_eV * Math.pow(x_param, 2) * (J + 1) / (3 * J), // High-T approximation
          entropy: N * kB_eV * Math.log(2 * J + 1), // High-T limit
          partitionFunction: Math.pow(Math.sinh((J + 0.5) * x_param) / Math.sinh(0.5 * x_param), N),
          freeEnergy: -N * kB_eV * T * Math.log(Math.sinh((J + 0.5) * x_param) / Math.sinh(0.5 * x_param)),
          fluctuation: Math.sqrt(N * muJ * muJ * (1 - brillouin * brillouin / (J * (J + 1)))),
          magnetization: N * muJ * brillouin,
          susceptibility: N * muJ * muJ * beta * (J + 1) / (3 * J)
        };
        break;
      }

      default:
        result = {
          temperature: T,
          averageEnergy: 0,
          heatCapacity: 0,
          entropy: 0,
          partitionFunction: 1,
          freeEnergy: 0,
          fluctuation: 0
        };
    }

    return result;
  }, [systemType, volume, mass, systemParameter, factorial]);

  // Calculate thermodynamic derivatives
  const calculateDerivatives = useCallback(() => {
    const data: ThermodynamicDerivative[] = [];
    const [minT, maxT] = temperatureRange;
    const steps = 100;
    const dT = 0.1; // Small temperature increment for derivatives
    
    for (let i = 0; i <= steps; i++) {
      const T = minT + (maxT - minT) * i / steps;
      const N = systemSize[0];
      const param = systemParameter[0];
      
      // Calculate function and derivatives using finite differences
      const f0 = calculateAdvancedEnsemble(T, N, param);
      const f1 = calculateAdvancedEnsemble(T + dT, N, param);
      const f2 = calculateAdvancedEnsemble(T + 2 * dT, N, param);
      
      const firstDerivative = (f1.heatCapacity - f0.heatCapacity) / dT;
      const secondDerivative = (f2.heatCapacity - 2 * f1.heatCapacity + f0.heatCapacity) / (dT * dT);
      
      data.push({
        temperature: T,
        value: f0.heatCapacity,
        derivative: firstDerivative,
        secondDerivative: secondDerivative
      });
    }
    
    setDerivativeData(data);
  }, [systemSize, systemParameter, temperatureRange, calculateAdvancedEnsemble]);

  // Calculate fluctuation analysis
  const calculateFluctuations = useCallback(() => {
    const data: FluctuationData[] = [];
    const [minT, maxT] = temperatureRange;
    const steps = 50;
    
    for (let i = 0; i <= steps; i++) {
      const T = minT + (maxT - minT) * i / steps;
      const N = systemSize[0];
      const param = systemParameter[0];
      
      const ensemble = calculateAdvancedEnsemble(T, N, param);
      const relativeFluctuation = ensemble.fluctuation / Math.abs(ensemble.averageEnergy);
      const correlationLength = Math.sqrt(N) * Math.exp(-1 / (kB_eV * T)); // Simplified model
      
      data.push({
        temperature: T,
        energyFluctuation: ensemble.fluctuation,
        relativeFluctuation: relativeFluctuation,
        correlationLength: correlationLength
      });
    }
    
    setFluctuationData(data);
  }, [systemSize, systemParameter, temperatureRange, calculateAdvancedEnsemble]);

  // Current ensemble calculation
  const currentEnsemble = useMemo(() => {
    return calculateAdvancedEnsemble(temperature[0], systemSize[0], systemParameter[0]);
  }, [temperature, systemSize, systemParameter, calculateAdvancedEnsemble]);

  // Temperature sweep calculation
  useEffect(() => {
    const data: EnsembleData[] = [];
    const [minT, maxT] = temperatureRange;
    const steps = 100;
    
    for (let i = 0; i <= steps; i++) {
      const T = minT + (maxT - minT) * i / steps;
      const result = calculateAdvancedEnsemble(T, systemSize[0], systemParameter[0]);
      data.push(result);
    }
    
    setEnsembleData(data);
    
    if (showAdvanced) {
      calculateDerivatives();
      calculateFluctuations();
    }
  }, [systemType, systemSize, systemParameter, temperatureRange, showAdvanced, calculateAdvancedEnsemble, calculateDerivatives, calculateFluctuations]);

  // Helper functions section - factorial moved above

  const exportData = useCallback(() => {
    const data = {
      systemType,
      temperature: temperature[0],
      systemSize: systemSize[0],
      parameter: systemParameter[0],
      currentState: currentEnsemble,
      temperatureSweep: ensembleData,
      derivatives: derivativeData,
      fluctuations: fluctuationData
    };
    
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `canonical_ensemble_${systemType}_${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
    
    toast({
      title: "Data Exported",
      description: "Ensemble data exported successfully",
    });
  }, [systemType, temperature, systemSize, systemParameter, currentEnsemble, ensembleData, derivativeData, fluctuationData, toast]);

  const runAnalysis = useCallback(async () => {
    setIsRunning(true);
    toast({
      title: "Analysis Started",
      description: "Running comprehensive thermodynamic analysis...",
    });
    
    // Simulate intensive calculation
    await new Promise(resolve => setTimeout(resolve, 1000));
    
    calculateDerivatives();
    calculateFluctuations();
    
    setIsRunning(false);
    toast({
      title: "Analysis Complete",
      description: "Thermodynamic analysis finished successfully",
    });
  }, [calculateDerivatives, calculateFluctuations, toast]);

  // Export CSV data
  const exportCSV = useCallback(() => {
    try {
      const csvData = [
        ['Temperature (K)', 'Average Energy (eV)', 'Heat Capacity (kB)', 'Entropy (kB)', 'Free Energy (eV)', 'Energy Fluctuation (eV)'],
        ...ensembleData.map(d => [
          d.temperature.toFixed(2),
          d.averageEnergy.toExponential(6),
          d.heatCapacity.toExponential(6),
          d.entropy.toExponential(6),
          d.freeEnergy.toExponential(6),
          d.fluctuation.toExponential(6)
        ])
      ];
      
      const csvContent = csvData.map(row => row.join(',')).join('\n');
      const blob = new Blob([csvContent], { type: 'text/csv' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `canonical_ensemble_${systemType}_${Date.now()}.csv`;
      a.click();
      URL.revokeObjectURL(url);
      
      toast({
        title: "CSV Exported",
        description: "Thermodynamic data exported successfully",
      });
    } catch (error) {
      toast({
        title: "Export Failed",
        description: "Failed to export CSV data",
        variant: "destructive"
      });
    }
  }, [ensembleData, systemType, toast]);

  // Create plot configuration
  const plotConfig = PLOT.config;

  const plotLayout = PLOT.layout;

  const moduleData = {
    title: "Canonical Ensemble",
    description: "Statistical mechanics of systems at constant temperature",
    category: "Statistical Physics",
    difficulty: "Advanced" as const,
    theory: {
      overview: "The canonical ensemble describes systems in thermal equilibrium with a heat reservoir at fixed temperature.",
      mathematics: [
        "Z = \\sum_i e^{-\\beta E_i}",
        "\\langle E \\rangle = -\\frac{\\partial \\ln Z}{\\partial \\beta}"
      ],
      references: ["Gibbs (1902)", "Boltzmann (1877)", "Planck (1900)"]
    }
  };

  return (
    <PhysicsModuleLayout>
      <script type="application/ld+json" dangerouslySetInnerHTML={{ __html: JSON.stringify(schemaLD) }} />
      <PhysicsModuleHeader
        title="Canonical Ensemble"
        description="Statistical mechanics of systems at constant temperature"
        category="Statistical Physics"
        difficulty="Advanced"
        equation="Z = \\sum_i e^{-\\beta E_i}"
        onReset={() => window.location.reload()}
        onExport={exportData}
      />
      <div className="max-w-7xl mx-auto space-y-6 p-6">
        {/* Header */}
        <div className="text-center space-y-2">
          <h1 className="text-4xl font-bold gradient-text">Canonical Ensemble</h1>
          <p className="text-xl text-muted-foreground">
            Statistical mechanics of systems at constant temperature
          </p>
          <div className="flex justify-center gap-2 flex-wrap">
            <Badge variant="secondary">Statistical Physics</Badge>
            <Badge variant="secondary">Thermodynamics</Badge>
            <Badge variant="secondary">Partition Function</Badge>
          </div>
        </div>

        <Tabs defaultValue="simulation" className="w-full">
          <TabsList className="grid w-full grid-cols-4">
            <TabsTrigger value="simulation">Simulation</TabsTrigger>
            <TabsTrigger value="theory">Theory</TabsTrigger>
            <TabsTrigger value="mathematics">Mathematics</TabsTrigger>
            <TabsTrigger value="applications">Applications</TabsTrigger>
          </TabsList>

          <TabsContent value="simulation" className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
              {/* Controls Panel */}
              <Card className="lg:col-span-1">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Calculator className="w-5 h-5" />
                    System Parameters
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="space-y-2">
                    <label className="text-sm font-medium">System Type</label>
                    <Select value={systemType} onValueChange={(value: SystemType) => setSystemType(value)}>
                      <SelectTrigger className="text-xs">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="ideal_gas">Classical Ideal Gas</SelectItem>
                        <SelectItem value="harmonic_oscillator">Quantum Harmonic Oscillator</SelectItem>
                        <SelectItem value="spin_system">Magnetic Spin System</SelectItem>
                        <SelectItem value="einstein_solid">Einstein Solid</SelectItem>
                        <SelectItem value="fermi_gas">Fermi Gas</SelectItem>
                        <SelectItem value="bose_gas">Bose Gas</SelectItem>
                        <SelectItem value="photon_gas">Photon Gas (Blackbody)</SelectItem>
                        <SelectItem value="paramagnet">Paramagnetic System</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>

                  <div className="space-y-2">
                    <label className="text-sm font-medium flex items-center gap-2">
                      <Thermometer className="w-4 h-4" />
                      Temperature: {temperature[0]} K
                    </label>
                    <Slider
                      value={temperature}
                      onValueChange={setTemperature}
                      min={temperatureRange[0]}
                      max={temperatureRange[1]}
                      step={1}
                      className="w-full"
                    />
                  </div>

                  <div className="space-y-2">
                    <label className="text-sm font-medium">
                      System Size (N): {systemSize[0]}
                    </label>
                    <Slider
                      value={systemSize}
                      onValueChange={setSystemSize}
                      min={10}
                      max={1000}
                      step={10}
                      className="w-full"
                    />
                  </div>

                  <div className="space-y-2">
                    <label className="text-sm font-medium">
                      {systemType === 'harmonic_oscillator' && `Frequency: ${systemParameter[0]} THz`}
                      {systemType === 'spin_system' && `Magnetic Field: ${systemParameter[0]} T`}
                      {systemType === 'einstein_solid' && `Einstein Temp Factor: ${systemParameter[0]}`}
                      {systemType === 'ideal_gas' && `Density Factor: ${systemParameter[0]}`}
                      {systemType === 'fermi_gas' && `Fermi Energy Factor: ${systemParameter[0]}`}
                      {systemType === 'bose_gas' && `Chemical Potential: ${systemParameter[0]}`}
                      {systemType === 'photon_gas' && `Volume Factor: ${systemParameter[0]}`}
                      {systemType === 'paramagnet' && `Angular Momentum J: ${systemParameter[0]}`}
                    </label>
                    <Slider
                      value={systemParameter}
                      onValueChange={setSystemParameter}
                      min={0.1}
                      max={10}
                      step={0.1}
                      className="w-full"
                    />
                  </div>

                  <div className="space-y-2">
                    <label className="text-sm font-medium">Temperature Range</label>
                    <div className="grid grid-cols-2 gap-2">
                      <Slider
                        value={[temperatureRange[0]]}
                        onValueChange={(value) => setTemperatureRange([value[0], temperatureRange[1]])}
                        min={1}
                        max={200}
                        step={1}
                      />
                      <Slider
                        value={[temperatureRange[1]]}
                        onValueChange={(value) => setTemperatureRange([temperatureRange[0], value[0]])}
                        min={300}
                        max={1000}
                        step={10}
                      />
                    </div>
                    <p className="text-xs text-muted-foreground">
                      {temperatureRange[0]} K to {temperatureRange[1]} K
                    </p>
                  </div>

                  <div className="flex gap-2">
                    <Button 
                      onClick={exportData}
                      variant="outline"
                      size="sm"
                      className="flex items-center gap-2 text-xs"
                    >
                      <Download className="w-3 h-3" />
                      JSON
                    </Button>
                    <Button 
                      onClick={exportCSV}
                      variant="outline"
                      size="sm"
                      className="flex items-center gap-2 text-xs"
                    >
                      <Download className="w-3 h-3" />
                      CSV
                    </Button>
                  </div>

                  {/* Current State Display - Enhanced */}
                  <Card className="bg-muted/50">
                    <CardHeader className="pb-2">
                      <CardTitle className="text-sm">Current Thermodynamic State</CardTitle>
                    </CardHeader>
                    <CardContent className="text-xs space-y-2">
                      <div className="grid grid-cols-2 gap-2">
                        <div>
                          <span className="font-medium">⟨E⟩:</span>
                          <div className="font-mono">{currentEnsemble.averageEnergy.toExponential(2)} eV</div>
                        </div>
                        <div>
                          <span className="font-medium">Cᵥ:</span>
                          <div className="font-mono">{currentEnsemble.heatCapacity.toExponential(2)} kᵦ</div>
                        </div>
                        <div>
                          <span className="font-medium">S:</span>
                          <div className="font-mono">{currentEnsemble.entropy.toExponential(2)} kᵦ</div>
                        </div>
                        <div>
                          <span className="font-medium">F:</span>
                          <div className="font-mono">{currentEnsemble.freeEnergy.toExponential(2)} eV</div>
                        </div>
                        <div>
                          <span className="font-medium">σₑ:</span>
                          <div className="font-mono">{currentEnsemble.fluctuation.toExponential(2)} eV</div>
                        </div>
                        <div>
                          <span className="font-medium">Z:</span>
                          <div className="font-mono">{currentEnsemble.partitionFunction.toExponential(2)}</div>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                </CardContent>
              </Card>

              {/* Main Plot Grid */}
              <div className="lg:col-span-3">
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 h-full">
                  {/* Average Energy Plot */}
                  <Card>
                    <CardHeader className="pb-2">
                      <CardTitle className="text-lg flex items-center gap-2">
                        <TrendingUp className="w-4 h-4" />
                        Average Energy ⟨E⟩
                      </CardTitle>
                      <div className="text-xs text-muted-foreground">
                        <InlineMath>{'\\langle E \\rangle = -\\frac{\\partial \\ln Z}{\\partial \\beta}'}</InlineMath>
                      </div>
                    </CardHeader>
                    <CardContent>
                      {ensembleData.length > 0 ? (
                        <Plot
                          data={[{
                            x: ensembleData.map(d => d.temperature),
                            y: ensembleData.map(d => d.averageEnergy),
                            type: 'scatter',
                            mode: 'lines',
                            line: { color: '#3b82f6', width: 2 },
                            name: '⟨E⟩'
                          }]}
                          layout={{
                            ...plotLayout,
                            height: 280,
                            xaxis: { ...plotLayout.xaxis, title: 'Temperature (K)' },
                            yaxis: { ...plotLayout.yaxis, title: 'Average Energy (eV)' }
                          }}
                          config={plotConfig}
                          style={{ width: '100%', height: '280px' }}
                        />
                      ) : (
                        <div className="flex items-center justify-center h-64 text-muted-foreground">
                          <AlertTriangle className="w-8 h-8 mr-2" />
                          No data available
                        </div>
                      )}
                    </CardContent>
                  </Card>

                  {/* Heat Capacity Plot */}
                  <Card>
                    <CardHeader className="pb-2">
                      <CardTitle className="text-lg flex items-center gap-2">
                        <Activity className="w-4 h-4" />
                        Heat Capacity Cᵥ
                      </CardTitle>
                      <div className="text-xs text-muted-foreground">
                        <InlineMath>{'C_V = k_B \\beta^2 \\langle (E - \\langle E \\rangle)^2 \\rangle'}</InlineMath>
                      </div>
                    </CardHeader>
                    <CardContent>
                      {ensembleData.length > 0 ? (
                        <Plot
                          data={[{
                            x: ensembleData.map(d => d.temperature),
                            y: ensembleData.map(d => d.heatCapacity),
                            type: 'scatter',
                            mode: 'lines',
                            line: { color: '#ef4444', width: 2 },
                            name: 'Cᵥ'
                          }]}
                          layout={{
                            ...plotLayout,
                            height: 280,
                            xaxis: { ...plotLayout.xaxis, title: 'Temperature (K)' },
                            yaxis: { ...plotLayout.yaxis, title: 'Heat Capacity (kᵦ)' }
                          }}
                          config={plotConfig}
                          style={{ width: '100%', height: '280px' }}
                        />
                      ) : (
                        <div className="flex items-center justify-center h-64 text-muted-foreground">
                          <AlertTriangle className="w-8 h-8 mr-2" />
                          No data available
                        </div>
                      )}
                    </CardContent>
                  </Card>

                  {/* Entropy Plot */}
                  <Card>
                    <CardHeader className="pb-2">
                      <CardTitle className="text-lg flex items-center gap-2">
                        <BarChart3 className="w-4 h-4" />
                        Entropy S
                      </CardTitle>
                      <div className="text-xs text-muted-foreground">
                        <InlineMath>{'S = k_B (\\ln Z + \\beta \\langle E \\rangle)'}</InlineMath>
                      </div>
                    </CardHeader>
                    <CardContent>
                      {ensembleData.length > 0 ? (
                        <Plot
                          data={[{
                            x: ensembleData.map(d => d.temperature),
                            y: ensembleData.map(d => d.entropy),
                            type: 'scatter',
                            mode: 'lines',
                            line: { color: '#10b981', width: 2 },
                            name: 'S'
                          }]}
                          layout={{
                            ...plotLayout,
                            height: 280,
                            xaxis: { ...plotLayout.xaxis, title: 'Temperature (K)' },
                            yaxis: { ...plotLayout.yaxis, title: 'Entropy (kᵦ)' }
                          }}
                          config={plotConfig}
                          style={{ width: '100%', height: '280px' }}
                        />
                      ) : (
                        <div className="flex items-center justify-center h-64 text-muted-foreground">
                          <AlertTriangle className="w-8 h-8 mr-2" />
                          No data available
                        </div>
                      )}
                    </CardContent>
                  </Card>

                  {/* Free Energy Plot */}
                  <Card>
                    <CardHeader className="pb-2">
                      <CardTitle className="text-lg flex items-center gap-2">
                        <Zap className="w-4 h-4" />
                        Free Energy F
                      </CardTitle>
                      <div className="text-xs text-muted-foreground">
                        <InlineMath>{'F = -k_B T \\ln Z'}</InlineMath>
                      </div>
                    </CardHeader>
                    <CardContent>
                      {ensembleData.length > 0 ? (
                        <Plot
                          data={[{
                            x: ensembleData.map(d => d.temperature),
                            y: ensembleData.map(d => d.freeEnergy),
                            type: 'scatter',
                            mode: 'lines',
                            line: { color: '#8b5cf6', width: 2 },
                            name: 'F'
                          }]}
                          layout={{
                            ...plotLayout,
                            height: 280,
                            xaxis: { ...plotLayout.xaxis, title: 'Temperature (K)' },
                            yaxis: { ...plotLayout.yaxis, title: 'Free Energy (eV)' }
                          }}
                          config={plotConfig}
                          style={{ width: '100%', height: '280px' }}
                        />
                      ) : (
                        <div className="flex items-center justify-center h-64 text-muted-foreground">
                          <AlertTriangle className="w-8 h-8 mr-2" />
                          No data available
                        </div>
                      )}
                    </CardContent>
                  </Card>
                </div>

                {/* Energy Fluctuations Plot - Full Width */}
                <Card className="mt-4">
                  <CardHeader className="pb-2">
                    <CardTitle className="text-lg flex items-center gap-2">
                      <Target className="w-4 h-4" />
                      Energy Fluctuations σₑ
                    </CardTitle>
                    <div className="text-xs text-muted-foreground">
                      <InlineMath>{'\\sigma_E = \\sqrt{\\langle (E - \\langle E \\rangle)^2 \\rangle}'}</InlineMath>
                    </div>
                  </CardHeader>
                  <CardContent>
                    {ensembleData.length > 0 ? (
                      <Plot
                        data={[{
                          x: ensembleData.map(d => d.temperature),
                          y: ensembleData.map(d => d.fluctuation),
                          type: 'scatter',
                          mode: 'lines',
                          line: { color: '#f59e0b', width: 2 },
                          name: 'σₑ'
                        }]}
                        layout={{
                          ...plotLayout,
                          height: 300,
                          xaxis: { ...plotLayout.xaxis, title: 'Temperature (K)' },
                          yaxis: { ...plotLayout.yaxis, title: 'Energy Fluctuation (eV)' }
                        }}
                        config={plotConfig}
                        style={{ width: '100%', height: '300px' }}
                      />
                    ) : (
                      <div className="flex items-center justify-center h-64 text-muted-foreground">
                        <AlertTriangle className="w-8 h-8 mr-2" />
                        No data available
                      </div>
                    )}
                  </CardContent>
                </Card>
              </div>
            </div>
          </TabsContent>

          <TabsContent value="theory">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <BookOpen className="w-5 h-5" />
                  Canonical Ensemble Theory
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-6">
                <div>
                  <h3 className="text-lg font-semibold mb-3">Physical Foundation</h3>
                  <p className="text-muted-foreground mb-4">
                    The canonical ensemble describes a system in thermal equilibrium with a heat reservoir at temperature T. 
                    Unlike the microcanonical ensemble (fixed energy), the canonical ensemble allows energy fluctuations while 
                    maintaining constant temperature, volume, and particle number.
                  </p>
                </div>

                <div>
                  <h3 className="text-lg font-semibold mb-3">Key Concepts</h3>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <Card className="p-4">
                      <h4 className="font-medium mb-2">Boltzmann Distribution</h4>
                      <p className="text-sm text-muted-foreground mb-2">
                        The probability of finding the system in state i with energy Eᵢ:
                      </p>
                      <div className="text-center">
                        <BlockMath>{'P_i = \\frac{e^{-\\beta E_i}}{Z}'}</BlockMath>
                      </div>
                    </Card>

                    <Card className="p-4">
                      <h4 className="font-medium mb-2">Partition Function</h4>
                      <p className="text-sm text-muted-foreground mb-2">
                        The normalization constant that contains all thermodynamic information:
                      </p>
                      <div className="text-center">
                        <BlockMath>{'Z = \\sum_i e^{-\\beta E_i}'}</BlockMath>
                      </div>
                    </Card>

                    <Card className="p-4">
                      <h4 className="font-medium mb-2">Energy Fluctuations</h4>
                      <p className="text-sm text-muted-foreground mb-2">
                        Unlike fixed-energy systems, canonical systems exhibit thermal fluctuations:
                      </p>
                      <div className="text-center">
                        <BlockMath>{'\\langle (\\Delta E)^2 \\rangle = k_B T^2 C_V'}</BlockMath>
                      </div>
                    </Card>

                    <Card className="p-4">
                      <h4 className="font-medium mb-2">Connection to Thermodynamics</h4>
                      <p className="text-sm text-muted-foreground mb-2">
                        The partition function directly yields thermodynamic potentials:
                      </p>
                      <div className="text-center">
                        <BlockMath>{'F = -k_B T \\ln Z'}</BlockMath>
                      </div>
                    </Card>
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="mathematics">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Calculator className="w-5 h-5" />
                  Mathematical Framework
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-6">
                <div>
                  <h3 className="text-lg font-semibold mb-3">Fundamental Relations</h3>
                  <div className="space-y-4">
                    <Card className="p-4">
                      <h4 className="font-medium mb-2">Partition Function and Probability</h4>
                      <div className="space-y-2">
                        <BlockMath>{'Z = \\sum_i e^{-\\beta E_i} \\quad \\text{where } \\beta = \\frac{1}{k_B T}'}</BlockMath>
                        <BlockMath>{'P_i = \\frac{e^{-\\beta E_i}}{Z}'}</BlockMath>
                      </div>
                    </Card>

                    <Card className="p-4">
                      <h4 className="font-medium mb-2">Thermodynamic Quantities</h4>
                      <div className="space-y-2">
                        <BlockMath>{'\\langle E \\rangle = -\\frac{\\partial \\ln Z}{\\partial \\beta} = \\frac{\\sum_i E_i e^{-\\beta E_i}}{Z}'}</BlockMath>
                        <BlockMath>{'C_V = \\frac{\\partial \\langle E \\rangle}{\\partial T} = k_B \\beta^2 \\langle (E - \\langle E \\rangle)^2 \\rangle'}</BlockMath>
                        <BlockMath>{'S = k_B (\\ln Z + \\beta \\langle E \\rangle)'}</BlockMath>
                        <BlockMath>{'F = \\langle E \\rangle - TS = -k_B T \\ln Z'}</BlockMath>
                      </div>
                    </Card>

                    <Card className="p-4">
                      <h4 className="font-medium mb-2">Fluctuation-Dissipation Relations</h4>
                      <div className="space-y-2">
                        <BlockMath>{'\\langle (\\Delta E)^2 \\rangle = \\langle E^2 \\rangle - \\langle E \\rangle^2 = k_B T^2 C_V'}</BlockMath>
                        <BlockMath>{'\\sigma_E = \\sqrt{\\langle (\\Delta E)^2 \\rangle}'}</BlockMath>
                      </div>
                    </Card>
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="applications">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Target className="w-5 h-5" />
                  System Applications
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div>
                    <h3 className="text-lg font-semibold mb-3">Classical Systems</h3>
                    <div className="space-y-3">
                      <Card className="p-3">
                        <h4 className="font-medium text-sm">Ideal Gas</h4>
                        <p className="text-xs text-muted-foreground">
                          Non-interacting particles in thermal equilibrium. Demonstrates equipartition theorem and classical limit.
                        </p>
                      </Card>
                      <Card className="p-3">
                        <h4 className="font-medium text-sm">Paramagnetic Systems</h4>
                        <p className="text-xs text-muted-foreground">
                          Magnetic dipoles in external field. Shows Curie law behavior and magnetic susceptibility.
                        </p>
                      </Card>
                    </div>
                  </div>

                  <div>
                    <h3 className="text-lg font-semibold mb-3">Quantum Systems</h3>
                    <div className="space-y-3">
                      <Card className="p-3">
                        <h4 className="font-medium text-sm">Harmonic Oscillator</h4>
                        <p className="text-xs text-muted-foreground">
                          Quantum oscillator ensemble. Exhibits zero-point energy and quantum heat capacity behavior.
                        </p>
                      </Card>
                      <Card className="p-3">
                        <h4 className="font-medium text-sm">Fermi/Bose Gases</h4>
                        <p className="text-xs text-muted-foreground">
                          Quantum statistical effects for fermions and bosons at finite temperature.
                        </p>
                       </Card>
                     </div>
                   </div>
                 </div>
               </CardContent>
               </Card>
             </TabsContent>
           </Tabs>
         </div>
    </PhysicsModuleLayout>
  );
}