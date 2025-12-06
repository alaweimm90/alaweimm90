import React, { useState, useMemo, useEffect } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { ArrowLeft, Settings, BookOpen, CheckCircle, BarChart, Download, RefreshCw, Share2, Image } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { SwipeableTabSystem } from '@/components/SwipeableTabSystem';
import { TouchGestureHandler } from '@/components/TouchGestureHandler';
import { SimulationGrid } from '@/components/EnhancedResponsiveGrid';
import { PhysicsModuleLayout } from '@/components/PhysicsModuleLayout';
import { PhysicsModuleHeader } from '@/components/PhysicsModuleHeader';
import { useToast } from '@/hooks/use-toast';
import { useSEO } from '@/hooks/use-seo';
import { 
  GrapheneParameters, 
  DEFAULT_GRAPHENE_PARAMETERS,
  calculateExactBandStructure,
  calculateNNOnlyBandStructure,
  calculateExactDOS,
  calculateNNOnlyDOS,
  getExactBrillouinZoneData,
  validateExactDiracPoint
} from '@/lib/graphene-physics-exact';
import { GrapheneParameterControls } from '@/components/graphene/GrapheneParameterControls';
import { BandStructureComparison } from '@/components/graphene/BandStructureComparison';
import { DOSComparison } from '@/components/graphene/DOSComparison';
import { CorrectHoneycombLattice } from '@/components/graphene/CorrectHoneycombLattice';
import { BrillouinZonePlot } from '@/components/graphene/BrillouinZonePlot';
import DiracCone3D from '@/components/graphene/DiracCone3D';
import { GrapheneTheoryPanel } from '@/components/graphene/GrapheneTheoryPanel';
import { PhysicsTheory } from '@/components/PhysicsTheory';
import GrapheneDebugWrapper from '@/components/graphene/GrapheneDebugWrapper';
import html2canvas from 'html2canvas';


export default function GrapheneBandStructure() {
  const navigate = useNavigate();
  const location = useLocation();
  const { toast } = useToast();
  useSEO({ title: 'Graphene Band Structure – SimCore', description: 'Interactive graphene band structure, DOS, and Dirac cone simulation.' });
  const schemaLD = useMemo(() => ({
    '@context': 'https://schema.org',
    '@type': 'TechArticle',
    headline: 'Graphene Band Structure',
    description: 'Interactive graphene band structure, DOS, and Dirac cone simulation.',
    inLanguage: 'en',
    author: { '@type': 'Organization', name: 'SimCore' },
    publisher: { '@type': 'Organization', name: 'SimCore' },
    keywords: 'graphene band structure, DOS, Dirac cone, tight-binding, Brillouin zone',
    about: ['Graphene', 'Band structure', 'Tight-binding', 'Dirac point'],
    mainEntityOfPage: typeof window !== 'undefined' ? window.location.href : ''
  }), []);
  const breadcrumbsLD = useMemo(() => ({
    '@context': 'https://schema.org',
    '@type': 'BreadcrumbList',
    itemListElement: [
      { '@type': 'ListItem', position: 1, name: 'Home', item: typeof window !== 'undefined' ? window.location.origin : '/' },
      { '@type': 'ListItem', position: 2, name: 'Modules', item: typeof window !== 'undefined' ? `${window.location.origin}/#modules-section` : '/#modules-section' },
      { '@type': 'ListItem', position: 3, name: 'Graphene Band Structure', item: typeof window !== 'undefined' ? window.location.href : '' }
    ]
  }), []);
  
  const [activeTab, setActiveTab] = useState('simulation');
  const [parameters, setParameters] = useState<GrapheneParameters>(DEFAULT_GRAPHENE_PARAMETERS);
  const [showValence, setShowValence] = useState(true);
  const [showConduction, setShowConduction] = useState(true);
  const [dosSmoothing, setDosSmoothing] = useState(0.1);
  const [validationMode, setValidationMode] = useState<'exact' | 'experimental'>('exact');
  const [snapshots, setSnapshots] = useState<{ label: string; kPath: number[]; valence: number[]; conduction: number[]; color?: string }[]>([]);

  // Calculate band structures using EXACT physics
  const bandDataNN = useMemo(() => calculateNNOnlyBandStructure(parameters), [parameters]);
  const bandDataNNN = useMemo(() => calculateExactBandStructure(parameters), [parameters]);
  const dosDataNN = useMemo(() => calculateNNOnlyDOS(parameters, [-3, 3], 1000, 150), [parameters]);
  const dosDataNNN = useMemo(() => calculateExactDOS(parameters, [-3, 3], 1000, 150), [parameters]);
  const brillouinData = useMemo(() => getExactBrillouinZoneData(), []);

  // Calculate physical quantities
  const fermiVelocity = useMemo(() => {
    // v_F = 3 * t1 * a / (2 * ℏ) in m/s
    const a_meters = 1.42e-10; // Lattice constant in meters
    const hbar_js = 1.054571817e-34; // ℏ in J⋅s
    const t1_joules = parameters.t1 * 1.602176634e-19; // Convert eV to J
    return (3 * t1_joules * a_meters) / (2 * hbar_js); // Returns m/s
  }, [parameters.t1]);

  // Validate Dirac point
  const diracValidation = useMemo(() => validateExactDiracPoint(parameters), [parameters]);

  const resetParameters = () => {
    setParameters(DEFAULT_GRAPHENE_PARAMETERS);
    toast({
      title: "Parameters Reset",
      description: "All parameters have been reset to default values.",
    });
  };

  const exportData = () => {
    const data = {
      parameters,
      bandStructure: {
        NN: bandDataNN,
        NNN: bandDataNNN
      },
      dos: {
        NN: dosDataNN,
        NNN: dosDataNNN
      },
      validation: diracValidation,
      fermiVelocity
    };
    
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'graphene-band-structure.json';
    a.click();
    URL.revokeObjectURL(url);
    
    toast({
      title: "Data Exported",
      description: "Band structure data exported successfully.",
    });
  };

  const exportBandCSV = () => {
    const rows = ['k,NN_valence,NN_conduction,NNN_valence,NNN_conduction'];
    const len = Math.min(bandDataNN.kPath.length, bandDataNNN.kPath.length);
    for (let i = 0; i < len; i++) {
      rows.push([
        bandDataNN.kPath[i],
        bandDataNN.valence[i],
        bandDataNN.conduction[i],
        bandDataNNN.valence[i],
        bandDataNNN.conduction[i]
      ].join(','));
    }
    const blob = new Blob([rows.join('\n')], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'graphene-band-structure.csv';
    a.click();
    URL.revokeObjectURL(url);
    toast({ title: 'CSV Exported', description: 'Band structure CSV downloaded.' });
  };

  const saveSnapshot = () => {
    const label = `t=${parameters.t1.toFixed(1)} eV, t'=${parameters.t2.toFixed(2)} eV`;
    setSnapshots((prev) => ([
      ...prev,
      { label, kPath: bandDataNNN.kPath.slice(), valence: bandDataNNN.valence.slice(), conduction: bandDataNNN.conduction.slice() }
    ]));
    toast({ title: 'Snapshot saved', description: label });
  };

  const clearSnapshots = () => {
    setSnapshots([]);
    toast({ title: 'Snapshots cleared', description: 'All snapshots removed.' });
  };

  // Shareable preset link
  const copyShareLink = () => {
    try {
      const payload = { parameters, showValence, showConduction, dosSmoothing };
      const enc = encodeURIComponent(btoa(JSON.stringify(payload)));
      const url = `${window.location.origin}/modules/graphene-band-structure?p=${enc}`;
      navigator.clipboard.writeText(url);
      toast({ title: 'Share link copied', description: 'Preset encoded into URL.' });
    } catch (e) {
      toast({ title: 'Share failed', description: 'Could not create share link.' });
    }
  };

  // Load preset from URL if present
  useEffect(() => {
    const p = new URLSearchParams(location.search).get('p');
    if (!p) return;
    try {
      const obj = JSON.parse(atob(decodeURIComponent(p)));
      if (obj.parameters) setParameters(obj.parameters);
      if (typeof obj.showValence === 'boolean') setShowValence(obj.showValence);
      if (typeof obj.showConduction === 'boolean') setShowConduction(obj.showConduction);
      if (typeof obj.dosSmoothing === 'number') setDosSmoothing(obj.dosSmoothing);
      toast({ title: 'Preset loaded', description: 'Parameters applied from URL.' });
    } catch (e) {
      // ignore
    }
  }, [location.search, toast]);

  // Exports (PNG via html2canvas)
  const captureAndDownload = async (elementId: string, filename: string) => {
    const el = document.getElementById(elementId);
    if (!el) return;
    const canvas = await html2canvas(el as HTMLElement, { backgroundColor: null, scale: 2 });
    const dataUrl = canvas.toDataURL('image/png');
    const a = document.createElement('a');
    a.href = dataUrl;
    a.download = filename;
    a.click();
  };

  const exportPNGBand = () => captureAndDownload('band-plot-card', 'graphene-band-structure.png');
  const exportPNGDOS = () => captureAndDownload('dos-plot-card', 'graphene-dos.png');
  const exportPNGDirac = () => captureAndDownload('dirac3d-card', 'dirac-cone-3d.png');

  const exportDOSCSV = () => {
    const rows = ['E, DOS_NN, DOS_NNN'];
    const len = Math.min(dosDataNN.energies.length, dosDataNNN.energies.length);
    for (let i = 0; i < len; i++) {
      rows.push([
        dosDataNN.energies[i],
        dosDataNN.dos[i],
        dosDataNNN.dos[i],
      ].join(','));
    }
    const blob = new Blob([rows.join('\n')], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'graphene-dos.csv';
    a.click();
    URL.revokeObjectURL(url);
    toast({ title: 'CSV Exported', description: 'DOS CSV downloaded.' });
  };
  const fermiLevel = parameters.onsite;

  const moduleData = {
    module: {
      title: "Graphene Band Structure",
      description: "Tight-binding model of graphene including nearest-neighbor (NN) and next-nearest-neighbor (NNN) contributions",
      category: "Condensed Matter Physics",
      difficulty: "Advanced" as const,
      theory: {
        overview: "Graphene exhibits a unique electronic band structure with linear dispersion near Dirac points, arising from its honeycomb lattice structure.",
        mathematics: [
          "\\hat{H} = -t \\sum_{\\langle i,j \\rangle} (\\hat{a}_i^\\dagger \\hat{b}_j + h.c.) - t' \\sum_{\\langle\\langle i,j \\rangle\\rangle} (\\hat{a}_i^\\dagger \\hat{a}_j + \\hat{b}_i^\\dagger \\hat{b}_j + h.c.)",
          "E^\\pm(\\vec{k}) = -t' f_2(\\vec{k}) \\pm t |f_1(\\vec{k})|",
          "f_1(\\vec{k}) = \\sum_{\\mu=1}^{3} e^{i\\vec{k} \\cdot \\vec{\\delta}_{\\mu}}"
        ],
        references: [
          'Castro Neto, A. H. et al., Rev. Mod. Phys. 81, 109 (2009)',
          'Wallace, P. R., Phys. Rev. 71, 622 (1947)',
          'Reich, S., Maultzsch, J., Thomsen, C., Ordejón, P., Phys. Rev. B 66, 035412 (2002)'
        ]
      }
    },
    title: "Graphene Band Structure",
    description: "Tight-binding model of graphene including nearest-neighbor (NN) and next-nearest-neighbor (NNN) contributions",
    equations: [
      {
        title: "Hamiltonian",
        latex: "\\hat{H} = -t \\sum_{\\langle i,j \\rangle} (\\hat{a}_i^\\dagger \\hat{b}_j + h.c.) - t' \\sum_{\\langle\\langle i,j \\rangle\\rangle} (\\hat{a}_i^\\dagger \\hat{a}_j + \\hat{b}_i^\\dagger \\hat{b}_j + h.c.)"
      },
      {
        title: "Band Dispersion",
        latex: "E^\\pm(\\vec{k}) = -t' f_2(\\vec{k}) \\pm t |f_1(\\vec{k})|"
      },
      {
        title: "Structure Factor",
        latex: "f_1(\\vec{k}) = \\sum_{\\mu=1}^{3} e^{i\\vec{k} \\cdot \\vec{\\delta}_{\\mu}}"
      }
    ],
    keyPoints: [
      "Bipartite honeycomb lattice with A and B sublattices",
      "Linear dispersion near Dirac points (K and K')",
      "Nearest-neighbor hopping preserves chiral symmetry",
      "Next-nearest-neighbor hopping breaks chiral symmetry"
    ],
    references: [
      'Castro Neto, A. H. et al., Rev. Mod. Phys. 81, 109 (2009)',
      'Wallace, P. R., Phys. Rev. 71, 622 (1947)',
      'Reich, S., Maultzsch, J., Thomsen, C., Ordejón, P., Phys. Rev. B 66, 035412 (2002)'
    ]
  };

  return (
    <div className="min-h-screen bg-gradient-cosmic">
      <script type="application/ld+json" dangerouslySetInnerHTML={{ __html: JSON.stringify(schemaLD) }} />
      
      <div className="w-full mx-auto p-4 sm:p-6 container-responsive">
        <PhysicsModuleHeader
          title="Graphene Electronic Band Structure"
          description="Research-grade tight-binding calculations featuring nearest and next-nearest neighbor interactions in honeycomb lattice systems. Explore Dirac cone physics and massless fermion behavior."
          category="Band Structure"
          difficulty="Research"
          equation="E^{\pm}(\vec{k}) = -t' f_2(\vec{k}) \pm t |f_1(\vec{k})|"
          onExport={exportData}
          onReset={resetParameters}
          className="mb-6"
        />

        <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full tabs-variant-quantum">
          <TabsList className="w-full grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 mb-4 sm:mb-8 gap-1 sm:gap-0 min-h-[48px]">
            <TabsTrigger value="simulation" className="min-h-[44px] min-w-[44px] flex flex-col sm:flex-row items-center justify-center gap-1 sm:gap-2 text-xs sm:text-sm p-2 touch-target">
              <Settings className="w-4 h-4 flex-shrink-0" />
              <span className="hidden sm:inline font-medium">Simulation</span>
              <span className="inline sm:hidden text-[11px] font-medium">Sim</span>
            </TabsTrigger>
            <TabsTrigger value="theory" className="min-h-[44px] min-w-[44px] flex flex-col sm:flex-row items-center justify-center gap-1 sm:gap-2 text-xs sm:text-sm p-2 touch-target">
              <BookOpen className="w-4 h-4 flex-shrink-0" />
              <span className="hidden sm:inline font-medium">Theory</span>
              <span className="inline sm:hidden text-[11px] font-medium">Theory</span>
            </TabsTrigger>
            <TabsTrigger value="validation" className="min-h-[44px] min-w-[44px] flex flex-col sm:flex-row items-center justify-center gap-1 sm:gap-2 text-xs sm:text-sm p-2 touch-target">
              <CheckCircle className="w-4 h-4 flex-shrink-0" />
              <span className="hidden sm:inline font-medium">Validation</span>
              <span className="inline sm:hidden text-[11px] font-medium">Valid</span>
            </TabsTrigger>
            <TabsTrigger value="analysis" className="min-h-[44px] min-w-[44px] flex flex-col sm:flex-row items-center justify-center gap-1 sm:gap-2 text-xs sm:text-sm p-2 touch-target">
              <BarChart className="w-4 h-4 flex-shrink-0" />
              <span className="hidden sm:inline font-medium">Analysis</span>
              <span className="inline sm:hidden text-[11px] font-medium">Analysis</span>
            </TabsTrigger>
            <TabsTrigger value="debug" className="min-h-[44px] min-w-[44px] flex flex-col sm:flex-row items-center justify-center gap-1 sm:gap-2 text-xs sm:text-sm p-2 touch-target">
              <RefreshCw className="w-4 h-4 flex-shrink-0" />
              <span className="hidden sm:inline font-medium">Debug</span>
              <span className="inline sm:hidden text-[11px] font-medium">Debug</span>
            </TabsTrigger>
          </TabsList>

          <TabsContent value="simulation" className="space-y-6">
            
            {/* Responsive Layout: Mobile-first approach */}
            <div className="flex flex-col lg:flex-row gap-4 sm:gap-6">
              {/* Parameter Controls - Responsive sidebar */}
              <div className="w-full lg:w-80 xl:w-96 space-y-4 sm:space-y-6">
                <div className="p-4 sm:p-6 border rounded-lg bg-card/30">
                  <div className="flex items-center justify-between mb-4">
                    <h3 className="text-lg sm:text-xl font-semibold">Controls</h3>
                  </div>
                  <div className="space-y-4 sm:space-y-6">
                    <GrapheneParameterControls
                      parameters={parameters}
                      onParametersChange={setParameters}
                      showValence={showValence}
                      showConduction={showConduction}
                      onShowValenceChange={setShowValence}
                      onShowConductionChange={setShowConduction}
                      onReset={resetParameters}
                      onExport={exportData}
                      fermiVelocity={fermiVelocity}
                    />
                    
                    <div className="space-y-3">
                      <label className="text-sm font-medium">DOS Smoothing σ: {dosSmoothing.toFixed(2)}</label>
                      <input
                        type="range"
                        min="0.01"
                        max="0.5"
                        step="0.01"
                        value={dosSmoothing}
                        onChange={(e) => setDosSmoothing(parseFloat(e.target.value))}
                        className="w-full h-6 touch-target"
                      />
                    </div>
                    
                    <div className="space-y-4">
                      <div className="p-3 sm:p-4 border rounded-lg bg-card">
                        <h4 className="font-semibold mb-2 text-sm sm:text-base">Hamiltonian</h4>
                        <p className="text-xs sm:text-sm text-muted-foreground">
                          H = -t'f₂(k) ± t|f₁(k)|
                        </p>
                      </div>
                      <div className="p-3 sm:p-4 border rounded-lg bg-card">
                        <h4 className="font-semibold mb-2 text-sm sm:text-base">Energy Shift</h4>
                        <p className="text-xs sm:text-sm text-muted-foreground">
                          E±(k) + 3t' → Dirac at E=0
                        </p>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
              
            {/* Responsive Grid Layout - Mobile: 1 col, Tablet: 2 cols, Desktop: 2x2 */}
            <div className="flex-1 grid grid-cols-1 md:grid-cols-2 gap-4 sm:gap-6">
              
              {/* Band Structure */}
              <div id="band-plot-card" className="p-4 sm:p-6 border rounded-lg bg-card min-h-[300px] sm:min-h-[400px] md:min-h-[500px]">
                <div className="flex flex-col sm:flex-row sm:items-center justify-between mb-3 sm:mb-4 gap-2">
                  <h3 className="text-lg sm:text-xl font-semibold">Electronic Band Structure</h3>
                  <div className="flex items-center gap-2">
                    <Button variant="outline" size="sm" onClick={saveSnapshot}>Save snapshot</Button>
                    <Button variant="ghost" size="sm" onClick={clearSnapshots}>Clear</Button>
                    <Button variant="secondary" size="sm" onClick={exportBandCSV}>Export CSV</Button>
                    <Button variant="outline" size="sm" onClick={exportPNGBand}><Image className="w-4 h-4 mr-1" />PNG</Button>
                    <Button variant="ghost" size="sm" onClick={copyShareLink}><Share2 className="w-4 h-4 mr-1" />Share</Button>
                  </div>
                </div>
                <div className="h-[250px] sm:h-[320px] md:h-[420px]">
                  <BandStructureComparison
                    dataNN={bandDataNN}
                    dataNNN={bandDataNNN}
                    showValence={showValence}
                    showConduction={showConduction}
                    fermiLevel={0} // Normalized Fermi level at E/t = 0
                    snapshots={snapshots}
                  />
                </div>
              </div>

              {/* Density of States */}
              <div id="dos-plot-card" className="p-4 sm:p-6 border rounded-lg bg-card min-h-[300px] sm:min-h-[400px] md:min-h-[500px]">
                <div className="flex flex-col sm:flex-row sm:items-center justify-between mb-3 sm:mb-4 gap-2">
                  <h3 className="text-lg sm:text-xl font-semibold">Density of States</h3>
                  <div className="flex items-center gap-2">
                    <Button variant="secondary" size="sm" onClick={exportDOSCSV}>Export CSV</Button>
                    <Button variant="outline" size="sm" onClick={exportPNGDOS}><Image className="w-4 h-4 mr-1" />PNG</Button>
                    <Button variant="ghost" size="sm" onClick={copyShareLink}><Share2 className="w-4 h-4 mr-1" />Share</Button>
                  </div>
                </div>
                <div className="h-[250px] sm:h-[320px] md:h-[420px]">
                  <DOSComparison
                    dataNN={dosDataNN}
                    dataNNN={dosDataNNN}
                  />
                </div>
              </div>

              {/* Honeycomb Lattice */}
              <div className="p-4 sm:p-6 border rounded-lg bg-card min-h-[300px] sm:min-h-[400px] md:min-h-[500px]">
                <div className="flex flex-col sm:flex-row sm:items-center justify-between mb-3 sm:mb-4 gap-2">
                  <h3 className="text-lg sm:text-xl font-semibold">Honeycomb Lattice</h3>
                  <div className="text-xs sm:text-sm text-muted-foreground">A/B sublattices</div>
                </div>
                <div className="h-[250px] sm:h-[320px] md:h-[420px]">
                  <CorrectHoneycombLattice className="w-full h-full" />
                </div>
              </div>
              
              {/* Brillouin Zone */}
              <div className="p-4 sm:p-6 border rounded-lg bg-card min-h-[300px] sm:min-h-[400px] md:min-h-[500px]">
                <div className="flex flex-col sm:flex-row sm:items-center justify-between mb-3 sm:mb-4 gap-2">
                  <h3 className="text-lg sm:text-xl font-semibold">First Brillouin Zone</h3>
                  <div className="text-xs sm:text-sm text-muted-foreground">High-symmetry path</div>
                </div>
                <div className="h-[250px] sm:h-[320px] md:h-[420px]">
                  <BrillouinZonePlot data={brillouinData} />
                </div>
              </div>
              
            </div>
            </div>
            <div className="p-6 border rounded-lg bg-card/50">
              <h3 className="text-xl font-semibold mb-4">Physical Validation & Analysis</h3>
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                <div className="text-center p-4 border rounded-lg bg-card">
                  <div className="text-2xl font-bold physics-valence">
                    {diracValidation.isValid ? '✓' : '✗'}
                  </div>
                  <div className="text-sm text-muted-foreground">Dirac Point</div>
                  <div className="text-lg font-semibold">
                    {diracValidation.isValid ? 'Valid' : 'Invalid'}
                  </div>
                </div>
                <div className="text-center p-4 border rounded-lg bg-card">
                  <div className="text-2xl font-bold physics-neutral">
                    {(diracValidation.gapAtK / parameters.t1).toFixed(6)}
                  </div>
                  <div className="text-sm text-muted-foreground">Gap at K (E/t)</div>
                </div>
                <div className="text-center p-4 border rounded-lg bg-card">
                  <div className="text-2xl font-bold physics-dos">
                    {(fermiVelocity / 1000000).toFixed(2)}
                  </div>
                  <div className="text-sm text-muted-foreground">vF (×10⁶ m/s)</div>
                </div>
                <div className="text-center p-4 border rounded-lg bg-card">
                  <div className="text-2xl font-bold physics-conduction">
                    {(parameters.t2 / parameters.t1).toFixed(3)}
                  </div>
                  <div className="text-sm text-muted-foreground">t'/t ratio</div>
                </div>
              </div>
            </div>

          </TabsContent>

          <TabsContent value="theory">
            <PhysicsTheory {...moduleData} />
          </TabsContent>

          <TabsContent value="validation">
            <GrapheneTheoryPanel 
              parameters={parameters}
              fermiVelocity={fermiVelocity}
              validationMode={validationMode === 'exact'}
              onValidationToggle={() => setValidationMode(validationMode === 'exact' ? 'experimental' : 'exact')}
            />
          </TabsContent>

          <TabsContent value="analysis" className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div className="p-6 border rounded-lg bg-card">
                <h3 className="text-xl font-semibold mb-4">Fermi Velocity</h3>
                <div className="space-y-2">
                  <p className="text-sm text-muted-foreground">
                    Linear dispersion coefficient near Dirac points
                  </p>
                  <p className="text-2xl font-bold">
                    {(fermiVelocity / 1000).toFixed(2)} × 10³ m/s
                  </p>
                  <p className="text-sm">
                    ≈ {(fermiVelocity / 299792458).toFixed(3)}c
                  </p>
                </div>
              </div>

              <div className="p-6 border rounded-lg bg-card">
                <h3 className="text-xl font-semibold mb-4">Fermi Level</h3>
                <div className="space-y-2">
                  <p className="text-sm text-muted-foreground">
                    Chemical potential (onsite energy)
                  </p>
                  <p className="text-2xl font-bold">
                    {fermiLevel.toFixed(3)} eV
                  </p>
                </div>
              </div>

              <div className="p-6 border rounded-lg bg-card">
                <h3 className="text-xl font-semibold mb-4">Energy Gap at K</h3>
                <div className="space-y-2">
                  <p className="text-sm text-muted-foreground">
                    Band gap at Dirac point (should be ≈ 0)
                  </p>
                  <p className="text-2xl font-bold">
                    {(diracValidation.gapAtK * 1000).toFixed(2)} meV
                  </p>
                  <div className="flex items-center gap-2">
                    {diracValidation.isValid ? (
                      <CheckCircle className="w-4 h-4 text-green-500" />
                    ) : (
                      <div className="w-4 h-4 rounded-full bg-red-500" />
                    )}
                    <span className="text-sm">
                      {diracValidation.isValid ? 'Valid Dirac Point' : 'Invalid Dirac Point'}
                    </span>
                  </div>
                </div>
              </div>
            </div>
          </TabsContent>

          <TabsContent value="debug" className="space-y-6">
            <div className="p-6 border rounded-lg bg-card">
              <h3 className="text-xl font-semibold mb-4">Debug Console Output</h3>
              <p className="text-sm text-muted-foreground mb-4">
                Check the browser console for detailed debugging information. The plots below use diagnostic rendering to isolate rendering issues.
              </p>
              <GrapheneDebugWrapper
                dataNN={bandDataNN}
                dataNNN={bandDataNNN}
                dosNN={dosDataNN}
                dosNNN={dosDataNNN}
                enableDebug={true}
              />
            </div>

            <div className="p-6 border rounded-lg bg-card">
              <h3 className="text-xl font-semibold mb-4">Data Validation Summary</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                <div className="p-4 border rounded-lg bg-card/50">
                  <h4 className="font-semibold physics-valence">NN Band Data</h4>
                  <div className="text-sm space-y-1">
                    <p>k-points: {bandDataNN.kPath.length}</p>
                    <p>k-range: {(Math.max(...bandDataNN.kPath) - Math.min(...bandDataNN.kPath)).toFixed(3)}</p>
                    <p>E-range: [{Math.min(...bandDataNN.valence, ...bandDataNN.conduction).toFixed(2)}, {Math.max(...bandDataNN.valence, ...bandDataNN.conduction).toFixed(2)}]</p>
                  </div>
                </div>
                
                <div className="p-4 border rounded-lg bg-card/50">
                  <h4 className="font-semibold physics-neutral">NNN Band Data</h4>
                  <div className="text-sm space-y-1">
                    <p>k-points: {bandDataNNN.kPath.length}</p>
                    <p>k-range: {(Math.max(...bandDataNNN.kPath) - Math.min(...bandDataNNN.kPath)).toFixed(3)}</p>
                    <p>E-range: [{Math.min(...bandDataNNN.valence, ...bandDataNNN.conduction).toFixed(2)}, {Math.max(...bandDataNNN.valence, ...bandDataNNN.conduction).toFixed(2)}]</p>
                  </div>
                </div>

                <div className="p-4 border rounded-lg bg-card/50">
                  <h4 className="font-semibold physics-dos">NN DOS Data</h4>
                  <div className="text-sm space-y-1">
                    <p>Energies: {dosDataNN.energies.length}</p>
                    <p>DOS points: {dosDataNN.dos.length}</p>
                    <p>Max DOS: {Math.max(...dosDataNN.dos).toFixed(3)}</p>
                  </div>
                </div>

                <div className="p-4 border rounded-lg bg-card/50">
                  <h4 className="font-semibold physics-conduction">NNN DOS Data</h4>
                  <div className="text-sm space-y-1">
                    <p>Energies: {dosDataNNN.energies.length}</p>
                    <p>DOS points: {dosDataNNN.dos.length}</p>
                    <p>Max DOS: {Math.max(...dosDataNNN.dos).toFixed(3)}</p>
                  </div>
                </div>
              </div>
            </div>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
}