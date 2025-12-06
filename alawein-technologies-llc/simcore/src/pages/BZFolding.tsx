import React, { useState, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Slider } from '@/components/ui/slider';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Label } from '@/components/ui/label';
import { ArrowLeft, Download, Settings, Info, Home } from 'lucide-react';
import { PhysicsModuleLayout, PhysicsContentCard } from '@/components/PhysicsModuleLayout';
import { PhysicsModuleHeader } from '@/components/PhysicsModuleHeader';
import { InlineMath, BlockMath } from '@/components/ui/Math';

import { toast } from '@/hooks/use-toast';
import Plot from 'react-plotly.js';
import { UNIFIED_SCIENTIFIC_PLOT_SYSTEM as PLOT } from '@/lib/scientific-plot-system';
import { useSEO } from '@/hooks/use-seo';

// 1D Band Folding - Starting point
const calculate1DBandFolding = (foldingFactor: number, bandwidth: number) => {
  const results = [];
  const nPoints = 200;
  
  for (let i = 0; i < nPoints; i++) {
    const k = (i / (nPoints - 1)) * Math.PI; // First BZ: -π to π
    
    // Original band
    const E_original = -bandwidth * Math.cos(k);
    
    // Folded bands
    for (let n = 0; n < foldingFactor; n++) {
      const k_folded = k / foldingFactor + (n * 2 * Math.PI) / foldingFactor;
      const E_folded = -bandwidth * Math.cos(k_folded);
      
      results.push({
        k: k,
        k_folded: k / foldingFactor,
        energy: E_folded,
        band: n,
        type: '1D'
      });
    }
  }
  
  return results;
};

// 2D Square Lattice Folding
const calculate2DBandFolding = (supercellSize: number, bandwidth: number) => {
  const results = [];
  const nPoints = 50;
  
  for (let i = 0; i < nPoints; i++) {
    for (let j = 0; j < nPoints; j++) {
      const kx = (i / (nPoints - 1)) * Math.PI;
      const ky = (j / (nPoints - 1)) * Math.PI;
      
      // Original 2D tight-binding band
      const E_original = -2 * bandwidth * (Math.cos(kx) + Math.cos(ky));
      
      // Folded k-points
      const kx_folded = kx / supercellSize;
      const ky_folded = ky / supercellSize;
      
      results.push({
        kx: kx,
        ky: ky,
        kx_folded: kx_folded,
        ky_folded: ky_folded,
        energy: E_original,
        type: '2D'
      });
    }
  }
  
  return results;
};

// Twisted Bilayer Graphene (TBG) - Magic Angle
const calculateTBGFolding = (twistAngle: number, interlayerCoupling: number) => {
  const results = [];
  const theta = twistAngle * Math.PI / 180;
  const t = 2.7; // eV, intralayer hopping
  const tPerp = interlayerCoupling; // eV, interlayer coupling
  
  // Magic angle calculation
  const magicAngle = Math.asin(Math.sqrt(3) * tPerp / (8 * t)) * 180 / Math.PI;
  const isMagicAngle = Math.abs(twistAngle - magicAngle) < 0.1;
  
  // Moiré lattice constant
  const a = 2.46; // Å, graphene lattice constant
  const aMoire = a / (2 * Math.sin(theta / 2));
  
  // Generate moiré BZ
  const nPoints = 100;
  const kMoire = 2 * Math.PI / aMoire;
  
  for (let i = 0; i < nPoints; i++) {
    for (let j = 0; j < nPoints; j++) {
      const kx = (i / (nPoints - 1) - 0.5) * kMoire;
      const ky = (j / (nPoints - 1) - 0.5) * kMoire;
      
      // Simplified TBG Hamiltonian near K point
      const K = 4 * Math.PI / (3 * a);
      const q = Math.sqrt(kx * kx + ky * ky);
      
      // Intralayer dispersion
      const v_F = 1e6; // m/s, Fermi velocity
      const E_graphene = 6.582e-34 * v_F * q / 1.602e-19; // Convert to eV
      
      // Interlayer coupling effect
      const couplingPhase = Math.cos(3 * Math.atan2(ky, kx));
      const interlayerTerm = tPerp * couplingPhase;
      
      // Mini-band structure
      const E_plus = E_graphene + interlayerTerm;
      const E_minus = E_graphene - interlayerTerm;
      
      results.push({
        kx: kx,
        ky: ky,
        energy_plus: E_plus,
        energy_minus: E_minus,
        magicAngle: magicAngle,
        isMagicAngle: isMagicAngle,
        moirePeriod: aMoire,
        type: 'TBG'
      });
    }
  }
  
  return results;
};


const BZFolding: React.FC = () => {
  const navigate = useNavigate();
  useSEO({ title: 'Brillouin Zone Folding – SimCore', description: 'From 1D chains to twisted bilayer graphene.' });
  const schemaLD = {
    '@context': 'https://schema.org',
    '@type': 'TechArticle',
    headline: 'Brillouin Zone Folding',
    description: 'From 1D chains to twisted bilayer graphene',
    inLanguage: 'en',
    author: { '@type': 'Organization', name: 'SimCore' },
    publisher: { '@type': 'Organization', name: 'SimCore' },
    keywords: 'BZ folding, band structure, supercell, twisted bilayer graphene',
    about: ['Brillouin zone', 'Band folding', 'Supercells', 'Twisted bilayer graphene'],
    mainEntityOfPage: typeof window !== 'undefined' ? window.location.href : ''
  } as const;
  
  // 1D parameters
  const [foldingFactor, setFoldingFactor] = useState(2);
  const [bandwidth1D, setBandwidth1D] = useState(2.0);
  
  // 2D parameters  
  const [supercellSize, setSupercellSize] = useState(2);
  const [bandwidth2D, setBandwidth2D] = useState(4.0);
  
  // TBG parameters
  const [twistAngle, setTwistAngle] = useState(1.05);
  const [interlayerCoupling, setInterlayerCoupling] = useState(0.11);
  
  // UI state
  const [activeTab, setActiveTab] = useState('simulation');
  
  // Calculate data for all systems
  const data1D = useMemo(() => calculate1DBandFolding(foldingFactor, bandwidth1D), [foldingFactor, bandwidth1D]);
  const data2D = useMemo(() => calculate2DBandFolding(supercellSize, bandwidth2D), [supercellSize, bandwidth2D]);
  const dataTBG = useMemo(() => calculateTBGFolding(twistAngle, interlayerCoupling), [twistAngle, interlayerCoupling]);
  
  // Generate plot data for each system
  const plotData1D = useMemo(() => {
    const bands = Array.from(new Set(data1D.map((d: any) => d.band)));
    return bands.map((band: any) => {
      const bandData = data1D.filter((d: any) => d.band === band);
      return {
        x: bandData.map((d: any) => d.k_folded),
        y: bandData.map((d: any) => d.energy),
        type: 'scatter',
        mode: 'lines',
        name: `Band ${band}`,
        line: { width: 3, color: band === 0 ? '#3b82f6' : '#ef4444' }
      };
    });
  }, [data1D]);

  const plotData2D = useMemo(() => {
    return [{
      x: data2D.map((d: any) => d.kx_folded),
      y: data2D.map((d: any) => d.ky_folded),
      z: data2D.map((d: any) => d.energy),
      type: 'scatter3d',
      mode: 'markers',
      marker: { size: 2, color: data2D.map((d: any) => d.energy), colorscale: 'Viridis' }
    }];
  }, [data2D]);

  const plotDataTBG = useMemo(() => {
    return [
      {
        x: dataTBG.map((d: any) => d.kx),
        y: dataTBG.map((d: any) => d.ky),
        z: dataTBG.map((d: any) => d.energy_plus),
        type: 'scatter3d',
        mode: 'markers',
        name: 'Upper band',
        marker: { size: 1, color: '#8b5cf6' }
      },
      {
        x: dataTBG.map((d: any) => d.kx),
        y: dataTBG.map((d: any) => d.ky),
        z: dataTBG.map((d: any) => d.energy_minus),
        type: 'scatter3d',
        mode: 'markers',
        name: 'Lower band',
        marker: { size: 1, color: '#f59e0b' }
      }
    ];
  }, [dataTBG]);
  
  const resetParameters = () => {
    setFoldingFactor(2);
    setBandwidth1D(2.0);
    setSupercellSize(2);
    setBandwidth2D(4.0);
    setTwistAngle(1.05);
    setInterlayerCoupling(0.11);
    toast({ title: "Parameters reset to defaults" });
  };
  
  const exportData = () => {
    const data = {
      parameters: {
        foldingFactor, bandwidth1D, supercellSize, bandwidth2D, 
        twistAngle, interlayerCoupling
      },
      data1D: data1D.slice(0, 1000),
      data2D: data2D.slice(0, 1000),
      dataTBG: dataTBG.slice(0, 1000)
    };
    
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'bz_folding_data.json';
    a.click();
    URL.revokeObjectURL(url);
    toast({ title: "Data exported successfully" });
  };
  
  // Get magic angle info for TBG
  const magicAngleInfo = useMemo(() => {
    if (dataTBG.length > 0) {
      const sample = dataTBG[0] as any;
      return {
        magicAngle: sample.magicAngle,
        isMagicAngle: sample.isMagicAngle,
        moirePeriod: sample.moirePeriod
      };
    }
    return null;
  }, [dataTBG]);
  
  return (
    <PhysicsModuleLayout>
      <script type="application/ld+json" dangerouslySetInnerHTML={{ __html: JSON.stringify(schemaLD) }} />
      <PhysicsModuleHeader
        title="Brillouin Zone Folding"
        description="From 1D chains to twisted bilayer graphene"
        category="Condensed Matter"
        difficulty="Advanced"
        equation="\\vec{k} \\rightarrow \\vec{k} + \\vec{G}"
        onReset={resetParameters}
        onExport={exportData}
      />
        
        <div className="flex-1 overflow-hidden">
          <Tabs value={activeTab} onValueChange={setActiveTab} className="h-full flex flex-col">
            <TabsList className="grid w-full grid-cols-3">
              <TabsTrigger value="simulation">Simulation</TabsTrigger>
              <TabsTrigger value="theory">Theory</TabsTrigger>
              <TabsTrigger value="analysis">Analysis</TabsTrigger>
            </TabsList>
            
            <TabsContent value="simulation" className="flex-1 overflow-hidden">
              <div className="h-full grid grid-cols-1 lg:grid-cols-4 gap-6 p-6">
                {/* Controls */}
                <div className="lg:col-span-1 space-y-4 overflow-y-auto">
                  {/* 1D Controls */}
                  <Card>
                    <CardHeader>
                      <CardTitle>1D Chain</CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-4">
                      <div>
                        <Label>Folding Factor: {foldingFactor}</Label>
                        <Slider
                          value={[foldingFactor]}
                          onValueChange={(value) => setFoldingFactor(value[0])}
                          max={8}
                          min={2}
                          step={1}
                          className="w-full"
                        />
                      </div>
                      <div>
                        <Label>Bandwidth (eV): {bandwidth1D}</Label>
                        <Slider
                          value={[bandwidth1D]}
                          onValueChange={(value) => setBandwidth1D(value[0])}
                          max={5}
                          min={0.5}
                          step={0.1}
                          className="w-full"
                        />
                      </div>
                    </CardContent>
                  </Card>

                  {/* 2D Controls */}
                  <Card>
                    <CardHeader>
                      <CardTitle>2D Square Lattice</CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-4">
                      <div>
                        <Label>Supercell Size: {supercellSize}×{supercellSize}</Label>
                        <Slider
                          value={[supercellSize]}
                          onValueChange={(value) => setSupercellSize(value[0])}
                          max={5}
                          min={2}
                          step={1}
                          className="w-full"
                        />
                      </div>
                      <div>
                        <Label>Bandwidth (eV): {bandwidth2D}</Label>
                        <Slider
                          value={[bandwidth2D]}
                          onValueChange={(value) => setBandwidth2D(value[0])}
                          max={8}
                          min={1}
                          step={0.1}
                          className="w-full"
                        />
                      </div>
                    </CardContent>
                  </Card>

                  {/* TBG Controls */}
                  <Card>
                    <CardHeader>
                      <CardTitle>Twisted Bilayer Graphene</CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-4">
                      <div>
                        <Label>Twist Angle (°): {twistAngle}</Label>
                        <Slider
                          value={[twistAngle]}
                          onValueChange={(value) => setTwistAngle(value[0])}
                          max={5}
                          min={0.5}
                          step={0.05}
                          className="w-full"
                        />
                      </div>
                      <div>
                        <Label>Interlayer Coupling (eV): {interlayerCoupling}</Label>
                        <Slider
                          value={[interlayerCoupling]}
                          onValueChange={(value) => setInterlayerCoupling(value[0])}
                          max={0.3}
                          min={0.05}
                          step={0.01}
                          className="w-full"
                        />
                      </div>
                      
                      {magicAngleInfo && (
                        <div className="p-3 bg-muted rounded-lg text-sm">
                          <div className="font-medium mb-2">Magic Angle Info</div>
                          <div>θ_magic = {magicAngleInfo.magicAngle.toFixed(2)}°</div>
                          <div>Moiré period = {magicAngleInfo.moirePeriod.toFixed(1)} Å</div>
                          {magicAngleInfo.isMagicAngle && (
                            <Badge className="mt-2" variant="default">At Magic Angle!</Badge>
                          )}
                        </div>
                      )}
                    </CardContent>
                  </Card>
                </div>
              
                 {/* Visualizations */}
                <div className="lg:col-span-3 space-y-4 overflow-y-auto">
                  {/* 1D Chain Plot */}
                  <Card className="h-[30vh]">
                    <CardHeader>
                      <CardTitle>1D Chain - Folded Bands</CardTitle>
                    </CardHeader>
                    <CardContent className="h-[calc(100%-4rem)]">
                      <Plot
                        data={plotData1D}
                        layout={{
                          ...PLOT.createLayout('', 'k (folded)', 'Energy (eV)', { showLegend: true }),
                          margin: { l: 60, r: 40, t: 20, b: 60 },
                          height: undefined,
                          legend: { x: 0, y: 1 }
                        }}
                        config={PLOT.config}
                        className="w-full h-full"
                        useResizeHandler
                      />
                    </CardContent>
                  </Card>

                  {/* 2D Lattice Plot */}
                  <Card className="h-[30vh]">
                    <CardHeader>
                      <CardTitle>2D Square Lattice - Energy Surface</CardTitle>
                    </CardHeader>
                    <CardContent className="h-[calc(100%-4rem)]">
                      <Plot
                        data={plotData2D}
                        layout={{
                          ...PLOT.layout,
                          title: '',
                          scene: {
                            xaxis: { title: 'kₓ (folded)' },
                            yaxis: { title: 'kᵧ (folded)' },
                            zaxis: { title: 'Energy (eV)' },
                            camera: {
                              eye: { x: 1.5, y: 1.5, z: 1.5 }
                            }
                          },
                          margin: { l: 40, r: 40, t: 20, b: 40 },
                          height: undefined,
                          showlegend: false,
                          font: { size: 11 }
                        }}
                        config={PLOT.config}
                        className="w-full h-full"
                        useResizeHandler
                      />
                    </CardContent>
                  </Card>

                  {/* TBG Plot */}
                  <Card className="h-[30vh]">
                    <CardHeader>
                      <CardTitle>Twisted Bilayer Graphene - Mini-bands</CardTitle>
                    </CardHeader>
                    <CardContent className="h-[calc(100%-4rem)]">
                      <Plot
                        data={plotDataTBG}
                        layout={{
                          ...PLOT.layout,
                          title: '',
                          scene: {
                            xaxis: { title: 'kₓ' },
                            yaxis: { title: 'kᵧ' },
                            zaxis: { title: 'Energy (eV)' },
                            camera: {
                              eye: { x: 1.5, y: 1.5, z: 1.5 }
                            }
                          },
                          margin: { l: 40, r: 40, t: 20, b: 40 },
                          height: undefined,
                          showlegend: true,
                          legend: { x: 0, y: 1 },
                          font: { size: 11 }
                        }}
                        config={PLOT.config}
                        className="w-full h-full"
                        useResizeHandler
                      />
                    </CardContent>
                  </Card>
                </div>
              </div>
            </TabsContent>
          
          <TabsContent value="theory" className="mt-6">
            <Card>
              <CardHeader>
                <CardTitle>BZ Folding Theory</CardTitle>
                <CardDescription>Understanding superlattice band structures</CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                <div>
                  <h3 className="text-lg font-semibold mb-3">1D Chain Folding</h3>
                  <p className="text-sm text-muted-foreground mb-3">
                    Creating a supercell with N unit cells folds the BZ by factor N:
                  </p>
                  <BlockMath math="k_{folded} = k_{original} + \frac{2\pi n}{Na}" />
                  <p className="text-sm text-muted-foreground mt-2">
                    This creates N bands in the reduced BZ.
                  </p>
                </div>
                
                <div>
                  <h3 className="text-lg font-semibold mb-3">2D Square Lattice</h3>
                  <p className="text-sm text-muted-foreground mb-3">
                    Supercell folding in 2D creates band multiplication:
                  </p>
                  <BlockMath math="E(\mathbf{k}) = -2t[\cos(k_x a) + \cos(k_y a)]" />
                  <p className="text-sm text-muted-foreground mt-2">
                    The <InlineMath math="N \times N" /> supercell creates <InlineMath math="N^2" /> folded bands.
                  </p>
                </div>
                
                <div>
                  <h3 className="text-lg font-semibold mb-3">Twisted Bilayer Graphene</h3>
                  <p className="text-sm text-muted-foreground mb-3">
                    The magic angle where flat bands appear:
                  </p>
                  <BlockMath math="\theta_m = \sin^{-1}\left(\frac{\sqrt{3}t_\perp}{8t}\right)" />
                  <p className="text-sm text-muted-foreground mt-2">
                    At θ ≈ 1.05°, interlayer coupling creates nearly flat mini-bands.
                  </p>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
          
          <TabsContent value="analysis" className="mt-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <Card>
                <CardHeader>
                  <CardTitle>System Properties</CardTitle>
                </CardHeader>
                 <CardContent className="space-y-4">
                  <div>
                    <h4 className="font-medium mb-2">1D Chain</h4>
                    <div className="grid grid-cols-2 gap-4 text-sm">
                      <div>
                        <div className="font-medium">Folding Factor</div>
                        <div className="text-muted-foreground">{foldingFactor}</div>
                      </div>
                      <div>
                        <div className="font-medium">Bands Created</div>
                        <div className="text-muted-foreground">{foldingFactor}</div>
                      </div>
                      <div>
                        <div className="font-medium">BZ Size</div>
                        <div className="text-muted-foreground">π/{foldingFactor}</div>
                      </div>
                      <div>
                        <div className="font-medium">Bandwidth</div>
                        <div className="text-muted-foreground">{bandwidth1D} eV</div>
                      </div>
                    </div>
                  </div>
                  
                  <div>
                    <h4 className="font-medium mb-2">2D Square Lattice</h4>
                    <div className="grid grid-cols-2 gap-4 text-sm">
                      <div>
                        <div className="font-medium">Supercell</div>
                        <div className="text-muted-foreground">{supercellSize}×{supercellSize}</div>
                      </div>
                      <div>
                        <div className="font-medium">Bands Created</div>
                        <div className="text-muted-foreground">{supercellSize * supercellSize}</div>
                      </div>
                      <div>
                        <div className="font-medium">BZ Reduction</div>
                        <div className="text-muted-foreground">1/{supercellSize}²</div>
                      </div>
                      <div>
                        <div className="font-medium">Bandwidth</div>
                        <div className="text-muted-foreground">{bandwidth2D} eV</div>
                      </div>
                    </div>
                  </div>
                  
                  <div>
                    <h4 className="font-medium mb-2">Twisted Bilayer Graphene</h4>
                    {magicAngleInfo && (
                      <div className="grid grid-cols-2 gap-4 text-sm">
                        <div>
                          <div className="font-medium">Twist Angle</div>
                          <div className="text-muted-foreground">{twistAngle}°</div>
                        </div>
                        <div>
                          <div className="font-medium">Magic Angle</div>
                          <div className="text-muted-foreground">{magicAngleInfo.magicAngle.toFixed(2)}°</div>
                        </div>
                        <div>
                          <div className="font-medium">Moiré Period</div>
                          <div className="text-muted-foreground">{magicAngleInfo.moirePeriod.toFixed(1)} Å</div>
                        </div>
                        <div>
                          <div className="font-medium">Coupling</div>
                          <div className="text-muted-foreground">{interlayerCoupling} eV</div>
                        </div>
                      </div>
                    )}
                  </div>
                </CardContent>
              </Card>
              
              <Card>
                <CardHeader>
                  <CardTitle>Physical Effects</CardTitle>
                </CardHeader>
                 <CardContent className="space-y-4 text-sm">
                  <div>
                    <h4 className="font-medium mb-2">1D Chain Effects</h4>
                    <div className="space-y-2">
                      <div className="flex items-center justify-between">
                        <span>Band Multiplication:</span>
                        <Badge variant="secondary">×{foldingFactor}</Badge>
                      </div>
                      <div className="flex items-center justify-between">
                        <span>k-space Compression:</span>
                        <Badge variant="secondary">1/{foldingFactor}</Badge>
                      </div>
                    </div>
                  </div>
                  
                  <div>
                    <h4 className="font-medium mb-2">2D Lattice Effects</h4>
                    <div className="space-y-2">
                      <div className="flex items-center justify-between">
                        <span>Band Multiplication:</span>
                        <Badge variant="secondary">×{supercellSize}²</Badge>
                      </div>
                      <div className="flex items-center justify-between">
                        <span>Supercell Effect:</span>
                        <Badge variant="secondary">Enhanced DOS</Badge>
                      </div>
                    </div>
                  </div>
                  
                  <div>
                    <h4 className="font-medium mb-2">TBG Effects</h4>
                    <div className="space-y-2">
                      <div className="flex items-center justify-between">
                        <span>Moiré Pattern:</span>
                        <Badge variant="secondary">Mini-BZ</Badge>
                      </div>
                      <div className="flex items-center justify-between">
                        <span>Magic Angle:</span>
                        <Badge variant={magicAngleInfo?.isMagicAngle ? "default" : "secondary"}>
                          {magicAngleInfo?.isMagicAngle ? 'Active' : 'Inactive'}
                        </Badge>
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>
          </Tabs>
        </div>
    </PhysicsModuleLayout>
  );
};

export default BZFolding;