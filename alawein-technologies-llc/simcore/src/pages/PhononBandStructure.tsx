import { useState, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Slider } from '@/components/ui/slider';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Switch } from '@/components/ui/switch';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Home, RotateCcw, Download, Info, BookOpen, Atom } from 'lucide-react';
import { InlineMath, BlockMath } from '@/components/ui/Math';

import { ScientificPlot } from '@/components/ScientificPlot';
import { toast } from '@/hooks/use-toast';
import Plot from 'react-plotly.js';
import { UNIFIED_SCIENTIFIC_PLOT_SYSTEM as PLOT, PHYSICS_AXIS_LABELS } from '@/lib/scientific-plot-system';
import { PhysicsModuleLayout } from '@/components/PhysicsModuleLayout';
import { PhysicsModuleHeader } from '@/components/PhysicsModuleHeader';
import { useSEO } from '@/hooks/use-seo';

// Monatomic chain physics calculation
const calculatePhononDispersionMono = (
  springConstant: number,
  mass: number,
  latticeConstant: number,
  temperature: number = 300
) => {
  const results = [];
  const a = latticeConstant;
  const K = springConstant;
  const M = mass;
  const kB = 1.38e-23; // Boltzmann constant
  
  // Generate q-path for 1D chain
  const nPoints = 300;
  const qMax = Math.PI / a;
  
  for (let i = 0; i < nPoints; i++) {
    const q = (i / (nPoints - 1)) * qMax;
    
    // Monatomic chain: ω(q) = 2√(K/M)|sin(qa/2)|
    const omega = 2 * Math.sqrt(K / M) * Math.abs(Math.sin(q * a / 2));
    const groupVelocity = Math.sqrt(K / M) * a * Math.cos(q * a / 2);
    const occupation = omega > 0 ? 1 / (Math.exp((1.055e-34 * omega * 1e12) / (kB * temperature)) - 1) : 0;
    
    results.push({
      q: q,
      qPath: q / qMax,
      frequency: omega / 1e12, // Convert to THz
      groupVelocity: groupVelocity,
      occupation: occupation,
      branch: 'acoustic'
    });
  }
  
  return results;
};

// Diatomic chain physics calculation with proper two-mass dynamics
const calculatePhononDispersionDi = (
  springConstant: number,
  mass1: number,
  mass2: number,
  latticeConstant: number,
  temperature: number = 300
) => {
  const results = [];
  const a = latticeConstant;
  const K = springConstant;
  const M1 = mass1;
  const M2 = mass2;
  const kB = 1.38e-23; // Boltzmann constant
  
  // Generate q-path for 1D chain
  const nPoints = 300;
  const qMax = Math.PI / a;
  
  for (let i = 0; i < nPoints; i++) {
    const q = (i / (nPoints - 1)) * qMax;
    
    // Diatomic chain dispersion relations
    const omega0Sq = K * (1/M1 + 1/M2);
    const deltaOmegaSq = K * K * (1/M1 + 1/M2) * (1/M1 + 1/M2) - 4 * K * K * Math.sin(q * a / 2) * Math.sin(q * a / 2) / (M1 * M2);
    
    // Acoustic branch (-)
    const omegaAcousticSq = 0.5 * (omega0Sq - Math.sqrt(deltaOmegaSq));
    const omegaAcoustic = Math.sqrt(Math.max(0, omegaAcousticSq));
    
    // Optical branch (+)
    const omegaOpticalSq = 0.5 * (omega0Sq + Math.sqrt(deltaOmegaSq));
    const omegaOptical = Math.sqrt(omegaOpticalSq);
    
    // Group velocities (numerical derivative)
    const dq = qMax / nPoints;
    const gradAcoustic = q < qMax - dq ? (omegaAcoustic - Math.sqrt(Math.max(0, 0.5 * (omega0Sq - Math.sqrt(K * K * (1/M1 + 1/M2) * (1/M1 + 1/M2) - 4 * K * K * Math.sin((q + dq) * a / 2) * Math.sin((q + dq) * a / 2) / (M1 * M2)))))) / dq : 0;
    const gradOptical = q < qMax - dq ? (omegaOptical - Math.sqrt(0.5 * (omega0Sq + Math.sqrt(K * K * (1/M1 + 1/M2) * (1/M1 + 1/M2) - 4 * K * K * Math.sin((q + dq) * a / 2) * Math.sin((q + dq) * a / 2) / (M1 * M2))))) / dq : 0;
    
    const occAcoustic = omegaAcoustic > 0 ? 1 / (Math.exp((1.055e-34 * omegaAcoustic * 1e12) / (kB * temperature)) - 1) : 0;
    const occOptical = omegaOptical > 0 ? 1 / (Math.exp((1.055e-34 * omegaOptical * 1e12) / (kB * temperature)) - 1) : 0;
    
    results.push({
      q: q,
      qPath: q / qMax,
      frequency: omegaAcoustic / 1e12, // Convert to THz
      groupVelocity: gradAcoustic,
      occupation: occAcoustic,
      branch: 'acoustic'
    });
    
    results.push({
      q: q,
      qPath: q / qMax,
      frequency: omegaOptical / 1e12, // Convert to THz
      groupVelocity: gradOptical,
      occupation: occOptical,
      branch: 'optical'
    });
  }
  
  return results;
};

// Calculate density of states
const calculateDOS = (phononData: any[]) => {
  const frequencies = phononData.map(d => d.frequency).sort((a, b) => a - b);
  const maxFreq = Math.max(...frequencies);
  const nBins = 200;
  const dos = [];
  
  for (let i = 0; i < nBins; i++) {
    const freq = (i / (nBins - 1)) * maxFreq;
    const sigma = maxFreq / 100;
    let density = 0;
    
    frequencies.forEach(f => {
      density += Math.exp(-0.5 * Math.pow((freq - f) / sigma, 2)) / (sigma * Math.sqrt(2 * Math.PI));
    });
    
    dos.push({ frequency: freq, dos: density });
  }
  
  return dos;
};

// Calculate thermal properties
const calculateThermalProperties = (phononData: any[], temperature: number) => {
  const kB = 1.38e-23;
  const hbar = 1.055e-34;
  
  let heatCapacity = 0;
  let thermalConductivity = 0;
  
  phononData.forEach(d => {
    if (d.frequency > 0) {
      const x = (hbar * d.frequency) / (kB * temperature);
      const n = 1 / (Math.exp(x) - 1);
      const dndt = (x * Math.exp(x)) / (Math.pow(Math.exp(x) - 1, 2) * temperature);
      
      heatCapacity += kB * x * x * Math.exp(x) / Math.pow(Math.exp(x) - 1, 2);
      thermalConductivity += d.groupVelocity * d.groupVelocity * heatCapacity / 3;
    }
  });
  
  return { heatCapacity, thermalConductivity };
};

// Phonon mode visualization component
const PhononModeSchematic = ({ latticeType, animate }: { latticeType: string, animate: boolean }) => {
  return (
    <div className="relative h-32 bg-background border border-border rounded-lg p-4 flex items-center justify-center">
      <div className="flex items-center space-x-4">
        {latticeType === 'monatomic' ? (
          // Monatomic chain
          <>
            <div className="flex space-x-2">
              {[...Array(6)].map((_, i) => (
                <div 
                  key={i}
                  className={`w-4 h-4 rounded-full bg-primary transition-transform duration-1000 ${animate ? 'animate-pulse' : ''}`}
                />
              ))}
            </div>
            <div className="text-sm text-muted-foreground">
              Acoustic mode: All atoms move in phase
            </div>
          </>
        ) : (
          // Diatomic chain
          <>
            <div className="flex space-x-2">
              {[...Array(3)].map((_, i) => (
                <div key={i} className="flex space-x-1">
                  <div className={`w-3 h-3 rounded-full bg-primary transform transition-transform duration-1000 ${
                    animate ? 'animate-pulse' : ''
                  }`} />
                  <div className={`w-5 h-5 rounded-full bg-secondary transform transition-transform duration-1000 ${
                    animate ? 'animate-bounce' : ''
                  }`} />
                </div>
              ))}
            </div>
            <div className="text-sm text-muted-foreground">
              <div>Acoustic: In-phase motion</div>
              <div>Optical: Out-of-phase motion</div>
            </div>
          </>
        )}
      </div>
    </div>
  );
};

const PhononBandStructure: React.FC = () => {
  const navigate = useNavigate();
  useSEO({ title: 'Phonon Band Structure – SimCore', description: 'Interactive monatomic and diatomic phonon dispersion with DOS and thermal properties.' });
  const schemaLD = useMemo(() => ({
    '@context': 'https://schema.org',
    '@type': 'TechArticle',
    headline: 'Phonon Band Structure',
    description: 'Interactive monatomic and diatomic phonon dispersion with DOS and thermal properties.',
    inLanguage: 'en',
    author: { '@type': 'Organization', name: 'SimCore' },
    publisher: { '@type': 'Organization', name: 'SimCore' },
    keywords: 'phonon band structure, dispersion, DOS, thermal properties',
    about: ['Phonons', 'Dispersion', 'Lattice dynamics'],
    mainEntityOfPage: typeof window !== 'undefined' ? window.location.href : ''
  }), []);
  const breadcrumbsLD = useMemo(() => ({
    '@context': 'https://schema.org',
    '@type': 'BreadcrumbList',
    itemListElement: [
      { '@type': 'ListItem', position: 1, name: 'Home', item: typeof window !== 'undefined' ? window.location.origin : '/' },
      { '@type': 'ListItem', position: 2, name: 'Modules', item: typeof window !== 'undefined' ? `${window.location.origin}/#modules-section` : '/#modules-section' },
      { '@type': 'ListItem', position: 3, name: 'Phonon Band Structure', item: typeof window !== 'undefined' ? window.location.href : '' }
    ]
  }), []);

  // Separate physics parameters for each system
  const [monoSpringConstant, setMonoSpringConstant] = useState(10.0);
  const [monoMass, setMonoMass] = useState(1.0);
  const [monoLatticeConstant, setMonoLatticeConstant] = useState(1.0);
  
  const [diSpringConstant, setDiSpringConstant] = useState(15.0);
  const [diMass1, setDiMass1] = useState(0.8);
  const [diMass2, setDiMass2] = useState(1.2);
  const [diLatticeConstant, setDiLatticeConstant] = useState(1.2);
  
  const [temperature, setTemperature] = useState(300);
  
  // UI state
  const [animate, setAnimate] = useState(false);
  const [activeTab, setActiveTab] = useState('simulation');
  
  // Calculate phonon data for both lattice types with separate parameters
  const monatomicData = useMemo(() => 
    calculatePhononDispersionMono(monoSpringConstant, monoMass, monoLatticeConstant, temperature),
    [monoSpringConstant, monoMass, monoLatticeConstant, temperature]
  );
  
  const diatomicData = useMemo(() => 
    calculatePhononDispersionDi(diSpringConstant, diMass1, diMass2, diLatticeConstant, temperature),
    [diSpringConstant, diMass1, diMass2, diLatticeConstant, temperature]
  );
  
  const monatomicDOS = useMemo(() => calculateDOS(monatomicData), [monatomicData]);
  const diatomicDOS = useMemo(() => calculateDOS(diatomicData), [diatomicData]);
  const monatomicThermal = useMemo(() => calculateThermalProperties(monatomicData, temperature), [monatomicData, temperature]);
  const diatomicThermal = useMemo(() => calculateThermalProperties(diatomicData, temperature), [diatomicData, temperature]);
  
  // Generate plot data for monatomic
  const monatomicPlotData = useMemo(() => {
    const acousticData = monatomicData.filter(d => d.branch === 'acoustic');
    
    return [{
      x: acousticData.map(d => d.qPath),
      y: acousticData.map(d => d.frequency),
      type: 'scatter' as const,
      mode: 'lines' as const,
      name: 'Acoustic',
      line: PLOT.lineStyles.bands.valence,
      hovertemplate: 'q: %{x:.3f}<br>ω: %{y:.3f} THz<extra></extra>'
    }];
  }, [monatomicData]);

  // Generate plot data for diatomic
  const diatomicPlotData = useMemo(() => {
    const acousticData = diatomicData.filter(d => d.branch === 'acoustic');
    const opticalData = diatomicData.filter(d => d.branch === 'optical');
    
    const traces: any[] = [{
      x: acousticData.map(d => d.qPath),
      y: acousticData.map(d => d.frequency),
      type: 'scatter' as const,
      mode: 'lines' as const,
      name: 'Acoustic',
      line: PLOT.lineStyles.bands.valence,
      hovertemplate: 'q: %{x:.3f}<br>ω: %{y:.3f} THz<extra></extra>'
    }];
    
    if (opticalData.length > 0) {
      traces.push({
        x: opticalData.map(d => d.qPath),
        y: opticalData.map(d => d.frequency),
        type: 'scatter' as const,
        mode: 'lines' as const,
        name: 'Optical',
        line: PLOT.lineStyles.bands.conduction,
        hovertemplate: 'q: %{x:.3f}<br>ω: %{y:.3f} THz<extra></extra>'
      });
    }
    
    return traces;
  }, [diatomicData]);
  
  const monatomicDOSPlotData = useMemo(() => [{
    x: monatomicDOS.map(d => d.dos),
    y: monatomicDOS.map(d => d.frequency),
    type: 'scatter' as const,
    mode: 'lines' as const,
    fill: 'tonextx',
    name: 'DOS',
    line: PLOT.lineStyles.dos.numerical,
    hovertemplate: 'DOS: %{x:.3f}<br>ω: %{y:.3f} THz<extra></extra>'
  }], [monatomicDOS]);

  const diatomicDOSPlotData = useMemo(() => [{
    x: diatomicDOS.map(d => d.dos),
    y: diatomicDOS.map(d => d.frequency),
    type: 'scatter' as const,
    mode: 'lines' as const,
    fill: 'tonextx',
    name: 'DOS',
    line: PLOT.lineStyles.dos.numerical,
    hovertemplate: 'DOS: %{x:.3f}<br>ω: %{y:.3f} THz<extra></extra>'
  }], [diatomicDOS]);
  
  const resetParameters = () => {
    setMonoSpringConstant(10.0);
    setMonoMass(1.0);
    setMonoLatticeConstant(1.0);
    setDiSpringConstant(15.0);
    setDiMass1(0.8);
    setDiMass2(1.2);
    setDiLatticeConstant(1.2);
    setTemperature(300);
    toast({ title: "Parameters reset to defaults" });
  };
  
  const exportData = () => {
    const data = {
      parameters: { 
        monatomic: { springConstant: monoSpringConstant, mass: monoMass, latticeConstant: monoLatticeConstant },
        diatomic: { springConstant: diSpringConstant, mass1: diMass1, mass2: diMass2, latticeConstant: diLatticeConstant },
        temperature 
      },
      monatomicData,
      diatomicData,
      monatomicDOS,
      diatomicDOS,
      thermalProperties: { monatomic: monatomicThermal, diatomic: diatomicThermal }
    };
    
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'phonon_data.json';
    a.click();
    URL.revokeObjectURL(url);
    toast({ title: "Data exported successfully" });
  };
  
  return (
    <PhysicsModuleLayout>
      <script type="application/ld+json" dangerouslySetInnerHTML={{ __html: JSON.stringify(schemaLD) }} />
      
      <PhysicsModuleHeader
        title="Phonon Band Structure"
        description="Lattice vibrations and thermal transport"
        category="Condensed Matter"
        difficulty="Intermediate"
        equation="\\omega(\\vec{q}) = \\sqrt{\\frac{K}{M}}|\\sin(qa/2)|"
        onReset={resetParameters}
        onExport={exportData}
      />
        
        <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full tabs-variant-field">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="simulation">Simulation</TabsTrigger>
            <TabsTrigger value="theory">Theory</TabsTrigger>
            <TabsTrigger value="analysis">Analysis</TabsTrigger>
          </TabsList>
          
          <TabsContent value="simulation" className="mt-6 flex-1">
            <div className="grid grid-cols-1 lg:grid-cols-5 gap-4 h-full">
              {/* Controls */}
              <div className="lg:col-span-1 space-y-4 h-full overflow-y-auto">
                <Card>
                  <CardHeader>
                    <CardTitle>Monatomic Parameters</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div>
                      <Label>Spring Constant (N/m): {monoSpringConstant}</Label>
                      <Slider
                        value={[monoSpringConstant]}
                        onValueChange={(value) => setMonoSpringConstant(value[0])}
                        max={50}
                        min={1}
                        step={1}
                        className="w-full"
                      />
                    </div>
                    
                    <div>
                      <Label>Mass (kg): {monoMass}</Label>
                      <Slider
                        value={[monoMass]}
                        onValueChange={(value) => setMonoMass(value[0])}
                        max={5}
                        min={0.1}
                        step={0.1}
                        className="w-full"
                      />
                    </div>
                    
                    <div>
                      <Label>Lattice Constant (m): {monoLatticeConstant}</Label>
                      <Slider
                        value={[monoLatticeConstant]}
                        onValueChange={(value) => setMonoLatticeConstant(value[0])}
                        max={3}
                        min={0.5}
                        step={0.1}
                        className="w-full"
                      />
                    </div>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader>
                    <CardTitle>Diatomic Parameters</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div>
                      <Label>Spring Constant (N/m): {diSpringConstant}</Label>
                      <Slider
                        value={[diSpringConstant]}
                        onValueChange={(value) => setDiSpringConstant(value[0])}
                        max={50}
                        min={1}
                        step={1}
                        className="w-full"
                      />
                    </div>
                    
                    <div>
                      <Label>Mass 1 (kg): {diMass1}</Label>
                      <Slider
                        value={[diMass1]}
                        onValueChange={(value) => setDiMass1(value[0])}
                        max={3}
                        min={0.1}
                        step={0.1}
                        className="w-full"
                      />
                    </div>

                    <div>
                      <Label>Mass 2 (kg): {diMass2}</Label>
                      <Slider
                        value={[diMass2]}
                        onValueChange={(value) => setDiMass2(value[0])}
                        max={3}
                        min={0.1}
                        step={0.1}
                        className="w-full"
                      />
                    </div>
                    
                    <div>
                      <Label>Lattice Constant (m): {diLatticeConstant}</Label>
                      <Slider
                        value={[diLatticeConstant]}
                        onValueChange={(value) => setDiLatticeConstant(value[0])}
                        max={3}
                        min={0.5}
                        step={0.1}
                        className="w-full"
                      />
                    </div>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader>
                    <CardTitle>Global Parameters</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div>
                      <Label>Temperature (K): {temperature}</Label>
                      <Slider
                        value={[temperature]}
                        onValueChange={(value) => setTemperature(value[0])}
                        max={1000}
                        min={10}
                        step={10}
                        className="w-full"
                      />
                    </div>
                    
                    <div className="flex items-center space-x-2">
                      <Switch
                        id="animate"
                        checked={animate}
                        onCheckedChange={setAnimate}
                      />
                      <Label htmlFor="animate">Animate</Label>
                    </div>
                  </CardContent>
                </Card>
              </div>
              
              {/* Visualizations */}
              <div className="lg:col-span-4 space-y-4 h-full overflow-y-auto">
                {/* Phonon mode schematics */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <Card>
                    <CardHeader>
                      <CardTitle className="flex items-center">
                        <Atom className="mr-2 h-5 w-5" />
                        Monatomic Chain
                      </CardTitle>
                      <CardDescription>Single atom per unit cell</CardDescription>
                    </CardHeader>
                    <CardContent>
                      <PhononModeSchematic latticeType="monatomic" animate={animate} />
                    </CardContent>
                  </Card>
                  
                  <Card>
                    <CardHeader>
                      <CardTitle className="flex items-center">
                        <Atom className="mr-2 h-5 w-5" />
                        Diatomic Chain
                      </CardTitle>
                      <CardDescription>Two atoms per unit cell</CardDescription>
                    </CardHeader>
                    <CardContent>
                      <PhononModeSchematic latticeType="diatomic" animate={animate} />
                    </CardContent>
                  </Card>
                </div>
                
                {/* Dispersion relations */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <Card>
                    <CardHeader>
                      <CardTitle>Monatomic Dispersion</CardTitle>
                      <CardDescription>ω(q) for single atom chain</CardDescription>
                    </CardHeader>
                    <CardContent>
                      <div className="h-[350px] w-full">
                        <Plot
                          data={monatomicPlotData}
                          layout={PLOT.createLayout(
                            'Monatomic Chain',
                            'q/q_max',
                            PHYSICS_AXIS_LABELS.frequency
                          )}
                          config={PLOT.config}
                          useResizeHandler={true}
                          className="w-full h-full"
                        />
                      </div>
                    </CardContent>
                  </Card>
                  
                  <Card>
                    <CardHeader>
                      <CardTitle>Diatomic Dispersion</CardTitle>
                      <CardDescription>ω(q) for two-atom chain</CardDescription>
                    </CardHeader>
                    <CardContent>
                      <div className="h-[350px] w-full">
                        <Plot
                          data={diatomicPlotData}
                          layout={PLOT.createLayout(
                            'Diatomic Chain',
                            'q/q_max',
                            PHYSICS_AXIS_LABELS.frequency
                          )}
                          config={PLOT.config}
                          useResizeHandler={true}
                          className="w-full h-full"
                        />
                      </div>
                    </CardContent>
                  </Card>
                </div>
                
                {/* DOS comparison */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <Card>
                    <CardHeader>
                      <CardTitle>Monatomic DOS</CardTitle>
                      <CardDescription>Density of states</CardDescription>
                    </CardHeader>
                    <CardContent>
                      <div className="h-[300px] w-full">
                        <Plot
                          data={monatomicDOSPlotData}
                          layout={PLOT.createLayout(
                            'Monatomic DOS',
                            'DOS (states/THz)',
                            PHYSICS_AXIS_LABELS.frequency
                          )}
                          config={PLOT.config}
                          useResizeHandler={true}
                          className="w-full h-full"
                        />
                      </div>
                    </CardContent>
                  </Card>
                  
                  <Card>
                    <CardHeader>
                      <CardTitle>Diatomic DOS</CardTitle>
                      <CardDescription>Density of states</CardDescription>
                    </CardHeader>
                    <CardContent>
                      <div className="h-[300px] w-full">
                        <Plot
                          data={diatomicDOSPlotData}
                          layout={PLOT.createLayout(
                            'Diatomic DOS',
                            'DOS (states/THz)',
                            PHYSICS_AXIS_LABELS.frequency
                          )}
                          config={PLOT.config}
                          useResizeHandler={true}
                          className="w-full h-full"
                        />
                      </div>
                    </CardContent>
                  </Card>
                </div>
                
                {/* Thermal properties comparison */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <Card>
                    <CardHeader>
                      <CardTitle>Monatomic Thermal Properties</CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-4">
                      <div className="grid grid-cols-2 gap-4 text-sm">
                        <div>
                          <div className="font-medium">Heat Capacity</div>
                          <div className="text-muted-foreground">
                            {monatomicThermal.heatCapacity.toExponential(2)} J/K
                          </div>
                        </div>
                        <div>
                          <div className="font-medium">Thermal Conductivity</div>
                          <div className="text-muted-foreground">
                            {monatomicThermal.thermalConductivity.toExponential(2)} W/mK
                          </div>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                  
                  <Card>
                    <CardHeader>
                      <CardTitle>Diatomic Thermal Properties</CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-4">
                      <div className="grid grid-cols-2 gap-4 text-sm">
                        <div>
                          <div className="font-medium">Heat Capacity</div>
                          <div className="text-muted-foreground">
                            {diatomicThermal.heatCapacity.toExponential(2)} J/K
                          </div>
                        </div>
                        <div>
                          <div className="font-medium">Thermal Conductivity</div>
                          <div className="text-muted-foreground">
                            {diatomicThermal.thermalConductivity.toExponential(2)} W/mK
                          </div>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                </div>
              </div>
            </div>
          </TabsContent>
          
          <TabsContent value="theory" className="mt-6">
            <Card>
              <CardHeader>
                <CardTitle>Phonon Theory</CardTitle>
                <CardDescription>Lattice dynamics and thermal transport</CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                <div>
                  <h3 className="text-lg font-semibold mb-3">Monatomic Chain</h3>
                  <p className="text-sm text-muted-foreground mb-3">
                    In a 1D monatomic chain, atoms are connected by springs with force constant K:
                  </p>
                  <BlockMath math="\omega(q) = 2\sqrt{\frac{K}{M}}\left|\sin\left(\frac{qa}{2}\right)\right|" />
                  <p className="text-sm text-muted-foreground mt-2">
                    This gives a single acoustic branch with linear dispersion near q = 0.
                  </p>
                </div>
                
                <div>
                  <h3 className="text-lg font-semibold mb-3">Diatomic Chain</h3>
                  <p className="text-sm text-muted-foreground mb-3">
                    Two different masses create acoustic and optical branches:
                  </p>
                  <BlockMath math="\omega^2(q) = \frac{K}{\mu}\left[1 \pm \sqrt{1 - \frac{4\mu^2\sin^2(qa/2)}{M_1 M_2}}\right]" />
                  <p className="text-sm text-muted-foreground mt-2">
                    where <InlineMath math="\mu = \frac{M_1 M_2}{M_1 + M_2}" /> is the reduced mass.
                  </p>
                </div>
                
                <div>
                  <h3 className="text-lg font-semibold mb-3">Thermal Properties</h3>
                  <p className="text-sm text-muted-foreground mb-3">
                    Heat capacity follows the Einstein/Debye model:
                  </p>
                  <BlockMath math="C_V = k_B \sum_{\mathbf{q},s} \left(\frac{\hbar\omega_{\mathbf{q}s}}{k_B T}\right)^2 \frac{e^{\hbar\omega_{\mathbf{q}s}/k_B T}}{(e^{\hbar\omega_{\mathbf{q}s}/k_B T} - 1)^2}" />
                </div>
              </CardContent>
            </Card>
          </TabsContent>
          
          <TabsContent value="analysis" className="mt-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <Card>
                <CardHeader>
                  <CardTitle>Calculated Properties</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div>
                      <div className="font-medium">Max Frequency</div>
                      <div className="text-muted-foreground">
                        {Math.max(...monatomicData.map(d => d.frequency), ...diatomicData.map(d => d.frequency)).toFixed(2)} THz
                      </div>
                    </div>
                     <div>
                       <div className="font-medium">Mono Sound Velocity</div>
                       <div className="text-muted-foreground">
                         {(Math.sqrt(monoSpringConstant / monoMass) * monoLatticeConstant).toFixed(2)} m/s
                       </div>
                     </div>
                     <div>
                       <div className="font-medium">Di Sound Velocity</div>
                       <div className="text-muted-foreground">
                         {(Math.sqrt(diSpringConstant / ((diMass1 + diMass2)/2)) * diLatticeConstant).toFixed(2)} m/s
                       </div>
                     </div>
                    <div>
                      <div className="font-medium">Modes</div>
                      <div className="text-muted-foreground">
                        Mono: 1 (A), Diatomic: 2 (A + O)
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
              
              <Card>
                <CardHeader>
                  <CardTitle>Parameter Effects</CardTitle>
                </CardHeader>
                <CardContent className="space-y-2 text-sm">
                  <div className="flex items-center justify-between">
                    <span>Spring Constant Effect:</span>
                    <Badge variant="secondary">Frequency scales as √K</Badge>
                  </div>
                  <div className="flex items-center justify-between">
                    <span>Mass Effect:</span>
                    <Badge variant="secondary">Frequency scales as 1/√M</Badge>
                  </div>
                  <div className="flex items-center justify-between">
                    <span>Temperature Effect:</span>
                    <Badge variant="secondary">Affects thermal properties</Badge>
                  </div>
                  <div className="flex items-center justify-between">
                    <span>Lattice Constant:</span>
                    <Badge variant="secondary">Sets velocity scale</Badge>
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>
        </Tabs>
    </PhysicsModuleLayout>
  );
};

export default PhononBandStructure;