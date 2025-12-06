import React, { useState, useMemo } from 'react';
import { Button } from '@/components/ui/button';
import { Slider } from '@/components/ui/slider';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Switch } from '@/components/ui/switch';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Download, Settings, Info, Eye, EyeOff, Zap, Atom } from 'lucide-react';
import { Card } from '@/components/ui/card';
import { PhysicsModuleLayout, PhysicsContentCard } from '@/components/PhysicsModuleLayout';
import { PhysicsModuleHeader } from '@/components/PhysicsModuleHeader';
import { ValleyBandStructurePlot } from '@/components/mos2/ValleyBandStructurePlot';
import { OpticalResponsePlot } from '@/components/mos2/OpticalResponsePlot';
import { CombinedBZPlots } from '@/components/mos2/CombinedBZPlots';
import { PhysicsTheory } from '@/components/PhysicsTheory';
import { InlineMath, BlockMath } from '@/components/ui/Math';
import { useSEO } from '@/hooks/use-seo';
// Enhanced MoS₂ Physics Engine with Literature-Accurate Models
const calculateMoS2ValleyPhysics = (
  kPoints: number[][],
  params: {
    delta: number;         // Band gap Δ (eV)
    lambda: number;        // SOC strength λ (eV)
    velocity: number;      // Fermi velocity ℏv (eV·Å)
    electricField: number; // Electric field (V/m)
    magneticField: number; // Magnetic field (T)
    interlayerCoupling: number;
    stackingOrder: string;
  }
) => {
  const results = [];
  const a = 3.19; // Lattice constant (Å)
  const hbar = 1; // Set ℏ = 1 in our units
  
  // Valley positions in reciprocal space (literature accurate)
  const K = [4 * Math.PI / (3 * a), 0];   // K valley at ~(1.32, 0)
  const Kp = [-4 * Math.PI / (3 * a), 0]; // K' valley at ~(-1.32, 0)
  
  console.log(`🔄 RECALCULATING with: Δ=${params.delta.toFixed(2)}eV, λ=${params.lambda.toFixed(3)}eV, ℏv=${params.velocity.toFixed(2)}eV·Å`);
  
  for (const [kx, ky] of kPoints) {
    // Distance from valley centers
    const qK = [kx - K[0], ky - K[1]];
    const qKp = [kx - Kp[0], ky - Kp[1]];
    const distK = Math.sqrt(qK[0]**2 + qK[1]**2);
    const distKp = Math.sqrt(qKp[0]**2 + qKp[1]**2);
    
    // Determine dominant valley (τ = ±1)
    const isKValley = distK < distKp;
    const tau = isKValley ? 1 : -1;
    const q = isKValley ? qK : qKp;
    const qMag = Math.sqrt(q[0]**2 + q[1]**2);
    
    // Extract physics parameters
    const Delta = params.delta;
    const lambda = params.lambda;
    const v = params.velocity;
    
    // Calculate energies for both spin channels (s = ±1)
    const kineticTerm = hbar * v * qMag; // |ℏv·q|
    const gapTerm = Delta / 2;
    
    const spinResults = [];
    for (const s of [1, -1]) {
      // Effective Hamiltonian: H_τs = (Δ/2)σz + ℏv(τkxσx + kyσy) + τsλ(1-σz)/2
      const socTerm = tau * s * lambda / 2;
      
      // Energy eigenvalues: E± = ±√[(ℏvq)² + (Δ/2)²] + τsλ/2
      const sqrtTerm = Math.sqrt(kineticTerm**2 + gapTerm**2);
      const E_conduction = gapTerm + sqrtTerm + socTerm;
      const E_valence = gapTerm - sqrtTerm + socTerm;
      
      spinResults.push({
        spin: s,
        energyConduction: E_conduction,
        energyValence: E_valence,
        socContribution: socTerm
      });
    }
    
    // Average energies for band structure plots
    const avgEC = (spinResults[0].energyConduction + spinResults[1].energyConduction) / 2;
    const avgEV = (spinResults[0].energyValence + spinResults[1].energyValence) / 2;
    
    // Berry curvature (TMD analytical formula from Xiao et al. PRL 2012)
    const denominator = Math.pow(Delta**2 + 4 * (hbar * v * qMag)**2, 3/2);
    const berryCurvature = denominator > 1e-12 ? 
      -tau * (2 * a**2 * (hbar * v)**2 * Delta) / denominator : 0;
    
    // Optical matrix elements with valley selection rules
    const theta = Math.atan2(q[1], q[0]);
    const opticalPrefactor = qMag > 1e-6 ? 
      (hbar * v * qMag) / Math.sqrt(kineticTerm**2 + gapTerm**2) : 0;
    
    // Circular polarization selection: σ+ couples to K, σ- couples to K'
    const opticalMatrixPlus = opticalPrefactor * (tau === 1 ? 1 : 0.05); // Strong K coupling
    const opticalMatrixMinus = opticalPrefactor * (tau === -1 ? 1 : 0.05); // Strong K' coupling
    
    // Spin texture (pseudospin direction)
    const spinTextureX = qMag > 1e-6 ? tau * q[0] / qMag : 0;
    const spinTextureY = qMag > 1e-6 ? q[1] / qMag : 0;
    
    // Valley magnetic moment (orbital + spin contributions)
    const orbitalMoment = qMag > 1e-6 ? 
      -tau * (hbar * v)**2 / (2 * (kineticTerm**2 + gapTerm**2)) : 0;
    const spinMoment = tau * lambda / 2;
    const totalMagneticMoment = orbitalMoment + spinMoment;
    
    // Group velocities
    const groupVelX = qMag > 1e-6 ? 
      hbar * v * q[0] / Math.sqrt(kineticTerm**2 + gapTerm**2) : 0;
    const groupVelY = qMag > 1e-6 ? 
      hbar * v * q[1] / Math.sqrt(kineticTerm**2 + gapTerm**2) : 0;
    
    results.push({
      kx, ky,
      valleyIndex: tau,
      energyConduction: avgEC,
      energyValence: avgEV,
      bandGap: avgEC - avgEV,
      berryCurvature,
      berryPhase: tau * Math.PI, // Valley-dependent Berry phase
      opticalMatrixPlus,
      opticalMatrixMinus,
      circularDichroism: opticalMatrixPlus - opticalMatrixMinus,
      magneticMoment: totalMagneticMoment,
      spinTextureX,
      spinTextureY,
      // Spin-resolved energies
      spinUpEnergy: spinResults[0].energyConduction,
      spinDownEnergy: spinResults[1].energyConduction,
      socSplitting: Math.abs(spinResults[0].socContribution - spinResults[1].socContribution),
      groupVelocityX: groupVelX,
      groupVelocityY: groupVelY,
      // Additional derived quantities
      effectiveMass: qMag > 1e-6 ? Delta / (2 * (hbar * v)**2) : Infinity,
      valleyHallVelocity: berryCurvature * params.electricField * 1e-6 // Convert to reasonable units
    });
  }
  
  // Debug output
  const maxBerry = Math.max(...results.map(p => Math.abs(p.berryCurvature)));
  console.log(`Max Berry curvature: ${maxBerry.toExponential(2)} Å²`);
  console.log(`Energy range: ${Math.min(...results.map(p => p.energyValence)).toFixed(2)} to ${Math.max(...results.map(p => p.energyConduction)).toFixed(2)} eV`);
  
  return results;
};

// Main component
export default function MoS2ValleyPhysics() {
  const [activeTab, setActiveTab] = useState('simulation');
  useSEO({ title: 'MoS₂ Valley Physics – SimCore', description: 'Valley-contrasting Berry curvature and optical selection in MoS₂.' });
  const schemaLD = useMemo(() => ({
    '@context': 'https://schema.org',
    '@type': 'TechArticle',
    headline: 'MoS₂ Valley Physics',
    description: 'Valley-contrasting Berry curvature and optical selection in MoS₂.',
    inLanguage: 'en',
    author: { '@type': 'Organization', name: 'SimCore' },
    publisher: { '@type': 'Organization', name: 'SimCore' },
    keywords: 'MoS2 valley physics, Berry curvature, optical selection rules, TMD',
    about: ['MoS2', 'Valleytronics', 'Berry curvature', 'Spin–orbit coupling'],
    mainEntityOfPage: typeof window !== 'undefined' ? window.location.href : ''
  }), []);
  const breadcrumbsLD = useMemo(() => ({
    '@context': 'https://schema.org',
    '@type': 'BreadcrumbList',
    itemListElement: [
      { '@type': 'ListItem', position: 1, name: 'Home', item: typeof window !== 'undefined' ? window.location.origin : '/' },
      { '@type': 'ListItem', position: 2, name: 'Modules', item: typeof window !== 'undefined' ? `${window.location.origin}/#modules-section` : '/#modules-section' },
      { '@type': 'ListItem', position: 3, name: 'MoS₂ Valley Physics', item: typeof window !== 'undefined' ? window.location.href : '' }
    ]
  }), []);

  // Enhanced physical parameters for TMD valley physics
  const [delta, setDelta] = useState([1.8]); // Band gap Δ (eV) - experimental MoS₂
  const [lambda, setLambda] = useState([0.15]); // SOC strength λ (eV) - realistic value
  const [velocity, setVelocity] = useState([3.0]); // Fermi velocity ℏv (eV·Å)
  const [electricField, setElectricField] = useState([0]); // E-field (V/m)
  const [magneticField, setMagneticField] = useState([0]); // B-field (T)
  const [interlayerCoupling, setInterlayerCoupling] = useState([0.11]); // t⊥ (eV)
  const [stackingOrder, setStackingOrder] = useState('AB'); // Stacking type
  
  // Visualization controls
  const [animate, setAnimate] = useState(false);
  
  // Enhanced valley physics parameters (use useMemo to ensure proper reactivity)
  const params = useMemo(() => ({
    delta: delta[0],
    lambda: lambda[0],
    velocity: velocity[0],
    electricField: electricField[0],
    magneticField: magneticField[0],
    interlayerCoupling: interlayerCoupling[0],
    stackingOrder: stackingOrder
  }), [delta, lambda, velocity, electricField, magneticField, interlayerCoupling, stackingOrder]);
  
  const kPoints = useMemo(() => {
    const points = [];
    const nPoints = 60;
    const range = 2 * Math.PI / 3.19; // BZ range
    
    for (let i = 0; i < nPoints; i++) {
      for (let j = 0; j < nPoints; j++) {
        const kx = (i - nPoints/2) * range / nPoints;
        const ky = (j - nPoints/2) * range / nPoints;
        points.push([kx, ky]);
      }
    }
    return points;
  }, []);
  
  const valleyData = useMemo(() => {
    console.log(`🔄 RECALCULATING with: Δ=${params.delta.toFixed(2)}eV, λ=${params.lambda.toFixed(3)}eV, ℏv=${params.velocity.toFixed(2)}eV·Å`);
    const data = calculateMoS2ValleyPhysics(kPoints, params);
    console.log(`✅ Generated ${data.length} valley data points`);
    return data;
  }, [kPoints, params]);

  // Module data for theory component
  const moduleData = {
    title: 'MoS₂ Valley Physics',
    description: 'Valley-contrasting Berry curvature and optical selection rules in transition metal dichalcogenides.',
    category: '2D Materials',
    difficulty: 'Advanced' as const,
    equation: '\\Omega(\\mathbf{k}) = \\nabla_{\\mathbf{k}} \\times \\langle u_{\\mathbf{k}}|i\\nabla_{\\mathbf{k}}|u_{\\mathbf{k}}\\rangle',
    theory: {
      overview: 'Valley-dependent physics in TMDs arising from broken inversion symmetry and strong spin-orbit coupling.',
      mathematics: [
        'k·p effective Hamiltonian',
        'Berry curvature calculation', 
        'Valley Hall conductivity',
        'Circular dichroism'
      ],
      references: [
        'Xiao et al., Phys. Rev. Lett. 108, 196802 (2012)',
        'Mak et al., Science 344, 1489 (2014)'
      ]
    }
  };

  const detailedTheory = {
    introduction: `Monolayer MoS₂ represents a paradigmatic example of valleytronics, where the valley degree of freedom in momentum space can be exploited for information processing. The combination of broken inversion symmetry and strong spin-orbit coupling leads to valley-contrasting physical properties, including Berry curvature, magnetic moments, and optical selection rules.`,
    
    fundamentals: [
      {
        title: 'Crystal Structure and Band Structure',
        content: 'MoS₂ has a hexagonal lattice with broken inversion symmetry due to the three-layer (S-Mo-S) structure. This leads to distinct K and K\' valleys with opposite valley indices.',
        equations: [
          '\\mathbf{K} = \\frac{4\\pi}{3a}(1, 0), \\quad \\mathbf{K}^\\prime = \\frac{4\\pi}{3a}(-1, 0)',
          'H_{\\tau s} = \\frac{\\Delta}{2}\\sigma_z + \\hbar v(\\tau k_x \\sigma_x + k_y \\sigma_y) + \\tau s \\lambda \\frac{1-\\sigma_z}{2}'
        ],
        derivation: 'The k·p model around valley centers captures the low-energy physics with valley index τ = ±1.'
      },
      {
        title: 'Berry Curvature and Valley Hall Effect',
        content: 'The Berry curvature acts as a magnetic field in momentum space, leading to valley-dependent transport phenomena.',
        equations: [
          '\\Omega_z(\\mathbf{k}) = -\\tau \\frac{2a^2t^2\\Delta}{(\\Delta^2 + 4a^2t^2|\\mathbf{q}|^2)^{3/2}}',
          '\\sigma^H = -\\frac{e^2}{h} \\sum_\\tau \\tau \\int \\Omega_z(\\mathbf{k}) f(\\mathbf{k}) d^2k'
        ],
        derivation: 'Valley-contrasting Berry curvature leads to the valley Hall effect when valleys are imbalanced.'
      },
      {
        title: 'Optical Valley Control',
        content: 'Circularly polarized light can selectively excite specific valleys due to optical selection rules.',
        equations: [
          '|\\langle \\psi_{c,\\mathbf{k}}|\\hat{p}_\\pm|\\psi_{v,\\mathbf{k}}\\rangle|^2 \\propto \\delta_{\\tau,\\pm}',
          'P_{\\text{circ}} = \\frac{I_{\\sigma^+} - I_{\\sigma^-}}{I_{\\sigma^+} + I_{\\sigma^-}}'
        ],
        derivation: 'The valley-dependent optical matrix elements enable valley-selective optical pumping.'
      }
    ],
    
    applications: [
      {
        title: 'Valleytronics Devices',
        description: 'Valley pseudospin can be used for information storage and processing in next-generation electronics.',
        examples: [
          'Valley-based field-effect transistors',
          'Optical valley filters and modulators',
          'Valley quantum dots for quantum computing'
        ]
      },
      {
        title: 'Optoelectronics',
        description: 'Valley-selective optical properties enable novel optoelectronic devices.',
        examples: [
          'Circular polarization detectors',
          'Valley LEDs with controlled helicity',
          'Ultrafast valley switches'
        ]
      }
    ],
    
    furtherReading: [
      {
        title: 'Coupled spin and valley physics in monolayers of MoS2 and other group-VI dichalcogenides',
        authors: 'Xiao, D., Liu, G.-B., Feng, W., Xu, X., Yao, W.',
        journal: 'Physical Review Letters',
        year: 2012,
        doi: '10.1103/PhysRevLett.108.196802'
      },
      {
        title: 'Control of valley polarization in monolayer MoS2 by optical helicity',
        authors: 'Mak, K. F., He, K., Shan, J., Heinz, T. F.',
        journal: 'Science',
        year: 2014,
        doi: '10.1126/science.1250140'
      }
    ]
  };

  const exportData = () => {
    const data = {
      parameters: params,
      valleyPhysics: valleyData,
      timestamp: new Date().toISOString(),
      metadata: {
        nPoints: kPoints.length
      }
    };
    
    const dataStr = JSON.stringify(data, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'mos2_valley_physics.json';
    link.click();
  };

  return (
    <PhysicsModuleLayout background="quantum">
      <script type="application/ld+json" dangerouslySetInnerHTML={{ __html: JSON.stringify(schemaLD) }} />
      
      <PhysicsModuleHeader
        title="MoS₂ Valley Physics"
        description="Valley-contrasting Berry curvature and optical selection rules in transition metal dichalcogenides"
        category="Materials & Crystals"
        difficulty="Advanced"
        equation="\Omega(\mathbf{k}) = \nabla_{\mathbf{k}} \times \langle u_{\mathbf{k}}|i\nabla_{\mathbf{k}}|u_{\mathbf{k}}\rangle"
        onExport={exportData}
      />
        <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6 tabs-variant-quantum">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="simulation" className="gap-2">
              <Settings className="h-4 w-4" />
              Simulation
            </TabsTrigger>
            <TabsTrigger value="theory" className="gap-2">
              <Info className="h-4 w-4" />
              Theory
            </TabsTrigger>
            <TabsTrigger value="analysis" className="gap-2">
              <Zap className="h-4 w-4" />
              Analysis
            </TabsTrigger>
          </TabsList>

          <TabsContent value="simulation" className="space-y-6">
            <div className="grid lg:grid-cols-4 gap-6">
              {/* Left Panel - Controls */}
              <div className="lg:col-span-1 space-y-6">
                {/* TMD Parameters */}
                <Card className="p-6">
                  <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                    <Atom className="h-5 w-5 text-primary" />
                    TMD Parameters
                  </h3>
                  <div className="space-y-6">
                    <div>
                      <Label className="text-sm font-medium flex items-center justify-between">
                        Band Gap Δ
                        <span className="text-muted-foreground">{delta[0].toFixed(2)} eV</span>
                      </Label>
                      <Slider
                        value={delta}
                        onValueChange={setDelta}
                        min={1.0}
                        max={3.0}
                        step={0.1}
                        className="mt-2"
                      />
                    </div>

                    <div>
                      <Label className="text-sm font-medium flex items-center justify-between">
                        SOC Strength λ
                        <span className="text-muted-foreground">{lambda[0].toFixed(3)} eV</span>
                      </Label>
                      <Slider
                        value={lambda}
                        onValueChange={setLambda}
                        min={0.05}
                        max={0.3}
                        step={0.01}
                        className="mt-2"
                      />
                    </div>

                    <div>
                      <Label className="text-sm font-medium flex items-center justify-between">
                        Fermi Velocity ℏv
                        <span className="text-muted-foreground">{velocity[0].toFixed(1)} eV·Å</span>
                      </Label>
                      <Slider
                        value={velocity}
                        onValueChange={setVelocity}
                        min={2.0}
                        max={5.0}
                        step={0.1}
                        className="mt-2"
                      />
                    </div>

                    <div>
                      <Label className="text-sm font-medium flex items-center justify-between">
                        Electric Field
                        <span className="text-muted-foreground">{electricField[0].toFixed(1)} V/m</span>
                      </Label>
                      <Slider
                        value={electricField}
                        onValueChange={setElectricField}
                        min={0}
                        max={1e8}
                        step={1e6}
                        className="mt-2"
                      />
                    </div>

                    <div>
                      <Label className="text-sm font-medium flex items-center justify-between">
                        Magnetic Field
                        <span className="text-muted-foreground">{magneticField[0].toFixed(2)} T</span>
                      </Label>
                      <Slider
                        value={magneticField}
                        onValueChange={setMagneticField}
                        min={0}
                        max={20}
                        step={0.5}
                        className="mt-2"
                      />
                    </div>

                    <div>
                      <Label className="text-sm font-medium">Stacking Order</Label>
                      <Select value={stackingOrder} onValueChange={setStackingOrder}>
                        <SelectTrigger className="w-full mt-2">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="AB">AB Stacking</SelectItem>
                          <SelectItem value="AA">AA Stacking</SelectItem>
                          <SelectItem value="twisted">Twisted</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                  </div>
                </Card>

                {/* Animation Controls */}
                <Card className="p-6">
                  <h3 className="text-lg font-semibold mb-4">Animation Mode</h3>
                  <div className="space-y-4">
                    <div className="flex items-center justify-between">
                      <Label htmlFor="animate">Enable Animation</Label>
                      <Switch
                        id="animate"
                        checked={animate}
                        onCheckedChange={setAnimate}
                      />
                    </div>
                  </div>
                </Card>
              </div>

              {/* Right Panel - Visualizations */}
              <div className="lg:col-span-3 space-y-6">
                {/* Top Row: Band Structure and Optical Response */}
                <div className="grid grid-cols-2 gap-6">
                  <Card className="p-6">
                    <div className="h-80">
                      <ValleyBandStructurePlot 
                        valleyData={valleyData}
                        showValence={true}
                        showConduction={true}
                        showSpinSplitting={true}
                      />
                    </div>
                  </Card>

                  <Card className="p-6">
                    <div className="h-80">
                      <OpticalResponsePlot 
                        valleyData={valleyData}
                        energyRange={[0, 3]}
                        showDichroism={true}
                      />
                    </div>
                  </Card>
                </div>

                {/* Bottom Row: Brillouin Zone Plots */}
                <Card className="p-6">
                  <div className="h-[600px]">
                    <CombinedBZPlots valleyData={valleyData} />
                  </div>
                </Card>
              </div>
            </div>
          </TabsContent>

          <TabsContent value="theory" className="space-y-6">
            <PhysicsTheory 
              module={moduleData}
              detailedTheory={detailedTheory}
            />
          </TabsContent>

          <TabsContent value="analysis" className="space-y-6">
            <div className="grid lg:grid-cols-2 gap-6">
              <Card className="p-6">
                <h3 className="text-lg font-semibold mb-4">Valley Hall Conductivity</h3>
                <div className="space-y-4">
                  <BlockMath math="\sigma_{xy}^H = \frac{e^2}{h} \sum_{\tau} \tau \int \Omega_z(\mathbf{k}) f(\mathbf{k}) d^2k" />
                  <p className="text-sm text-muted-foreground">
                    The valley Hall conductivity arises from the Berry curvature imbalance between K and K' valleys.
                  </p>
                </div>
              </Card>

              <Card className="p-6">
                <h3 className="text-lg font-semibold mb-4">Optical Selection Rules</h3>
                <div className="space-y-4">
                  <BlockMath math="|\langle \psi_{c,\mathbf{k}}|\hat{p}_\pm|\psi_{v,\mathbf{k}}\rangle|^2 \propto \delta_{\tau,\pm}" />
                  <p className="text-sm text-muted-foreground">
                    Circular polarization σ+ (σ-) selectively excites K (K') valley due to symmetry constraints.
                  </p>
                </div>
              </Card>
            </div>
          </TabsContent>
        </Tabs>
    </PhysicsModuleLayout>
  );
}