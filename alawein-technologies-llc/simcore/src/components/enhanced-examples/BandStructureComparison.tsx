/**
 * Enhanced Band Structure Comparison
 * Compare band structures across different materials
 */

import React, { useState, useMemo } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Slider } from '@/components/ui/slider';
import { Switch } from '@/components/ui/switch';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { ScientificPlot } from '@/components/ScientificPlot';
import { MLAnalysis } from '@/components/MLAnalysis';
import { 
  Layers, 
  GitCompare, 
  Zap, 
  BarChart3,
  Settings,
  Download,
  Eye,
  EyeOff
} from 'lucide-react';

interface MaterialProperties {
  name: string;
  latticeConstant: number;
  bandGap: number;
  electronMass: number;
  holeMass: number;
  dielectricConstant: number;
  color: string;
}

const materials: MaterialProperties[] = [
  {
    name: 'Silicon',
    latticeConstant: 5.43,
    bandGap: 1.12,
    electronMass: 0.26,
    holeMass: 0.81,
    dielectricConstant: 11.7,
    color: '#3b82f6'
  },
  {
    name: 'Germanium',
    latticeConstant: 5.66,
    bandGap: 0.66,
    electronMass: 0.22,
    holeMass: 0.34,
    dielectricConstant: 16.0,
    color: '#ef4444'
  },
  {
    name: 'GaAs',
    latticeConstant: 5.65,
    bandGap: 1.42,
    electronMass: 0.067,
    holeMass: 0.45,
    dielectricConstant: 12.9,
    color: '#10b981'
  },
  {
    name: 'InP',
    latticeConstant: 5.87,
    bandGap: 1.35,
    electronMass: 0.08,
    holeMass: 0.6,
    dielectricConstant: 12.4,
    color: '#f59e0b'
  },
  {
    name: 'GaN',
    latticeConstant: 3.19,
    bandGap: 3.4,
    electronMass: 0.2,
    holeMass: 1.4,
    dielectricConstant: 9.5,
    color: '#8b5cf6'
  }
];

interface BandStructureData {
  kPoints: number[];
  energyBands: number[][];
  dos: number[];
  energyAxis: number[];
}

export function BandStructureComparison() {
  const [selectedMaterials, setSelectedMaterials] = useState<string[]>(['Silicon', 'GaAs']);
  const [showValenceBands, setShowValenceBands] = useState(true);
  const [showConductionBands, setShowConductionBands] = useState(true);
  const [showDOS, setShowDOS] = useState(false);
  const [temperatureK, setTemperatureK] = useState(300);
  const [strainPercent, setStrainPercent] = useState(0);
  const [analysisType, setAnalysisType] = useState<'bandgap' | 'effective_mass' | 'mobility'>('bandgap');

  // Calculate band structure for a material
  const calculateBandStructure = useMemo(() => {
    return (material: MaterialProperties): BandStructureData => {
      const nKPoints = 100;
      const nBands = 8;
      const nEnergyPoints = 200;
      
      // Create k-point path (Γ-X-M-Γ)
      const kPoints = [];
      for (let i = 0; i < nKPoints; i++) {
        if (i < nKPoints / 3) {
          // Γ to X
          kPoints.push((i / (nKPoints / 3)) * Math.PI / material.latticeConstant);
        } else if (i < 2 * nKPoints / 3) {
          // X to M
          const t = (i - nKPoints / 3) / (nKPoints / 3);
          kPoints.push(Math.PI / material.latticeConstant * (1 + t * (Math.sqrt(2) - 1)));
        } else {
          // M to Γ
          const t = (i - 2 * nKPoints / 3) / (nKPoints / 3);
          kPoints.push(Math.PI / material.latticeConstant * Math.sqrt(2) * (1 - t));
        }
      }

      // Calculate energy bands using tight-binding approximation
      const energyBands: number[][] = [];
      for (let band = 0; band < nBands; band++) {
        const energies = [];
        for (let ki = 0; ki < nKPoints; ki++) {
          const k = kPoints[ki];
          
          // Simple tight-binding model with strain effects
          const strainFactor = 1 + strainPercent / 100;
          const t = 2.7 * strainFactor; // Hopping parameter
          
          let energy;
          if (band < nBands / 2) {
            // Valence bands
            energy = -material.bandGap / 2 - t * Math.cos(k * material.latticeConstant) - band * 0.5;
          } else {
            // Conduction bands
            energy = material.bandGap / 2 + t * Math.cos(k * material.latticeConstant) + (band - nBands / 2) * 0.3;
          }
          
          // Add thermal broadening
          const thermalEnergy = 8.617e-5 * temperatureK; // kT in eV
          energy += (Math.random() - 0.5) * thermalEnergy * 0.1;
          
          energies.push(energy);
        }
        energyBands.push(energies);
      }

      // Calculate density of states
      const energyAxis = [];
      const dos = [];
      const eMin = -material.bandGap - 2;
      const eMax = material.bandGap + 2;
      
      for (let i = 0; i < nEnergyPoints; i++) {
        const energy = eMin + (eMax - eMin) * i / (nEnergyPoints - 1);
        energyAxis.push(energy);
        
        // Simple DOS calculation
        let dosValue = 0;
        const sigma = 0.1; // Broadening
        
        for (const band of energyBands) {
          for (const bandEnergy of band) {
            dosValue += Math.exp(-((energy - bandEnergy) ** 2) / (2 * sigma ** 2));
          }
        }
        
        dos.push(dosValue / (sigma * Math.sqrt(2 * Math.PI)));
      }

      return {
        kPoints,
        energyBands,
        dos,
        energyAxis
      };
    };
  }, [temperatureK, strainPercent]);

  // Get band structure data for selected materials
  const bandStructureData = useMemo(() => {
    return selectedMaterials.map(materialName => {
      const material = materials.find(m => m.name === materialName);
      if (!material) return null;
      
      return {
        material,
        data: calculateBandStructure(material)
      };
    }).filter(Boolean);
  }, [selectedMaterials, calculateBandStructure]);

  // Prepare plot data for band structures
  const bandStructurePlotData = useMemo(() => {
    const datasets = [];
    
    for (const { material, data } of bandStructureData) {
      if (!material || !data) continue;
      
      // Add each band as a separate dataset
      data.energyBands.forEach((band, bandIndex) => {
        const isValence = bandIndex < data.energyBands.length / 2;
        const isConduction = bandIndex >= data.energyBands.length / 2;
        
        if ((isValence && showValenceBands) || (isConduction && showConductionBands)) {
          datasets.push({
            x: data.kPoints,
            y: band,
            label: `${material.name} Band ${bandIndex + 1}`,
            color: material.color,
            opacity: isValence ? 0.8 : 0.6,
            lineStyle: isConduction ? 'dashed' : 'solid'
          });
        }
      });
    }
    
    return { datasets };
  }, [bandStructureData, showValenceBands, showConductionBands]);

  // Prepare DOS plot data
  const dosPlotData = useMemo(() => {
    const datasets = [];
    
    for (const { material, data } of bandStructureData) {
      if (!material || !data) continue;
      
      datasets.push({
        x: data.energyAxis,
        y: data.dos,
        label: `${material.name} DOS`,
        color: material.color
      });
    }
    
    return { datasets };
  }, [bandStructureData]);

  // Analysis data for ML
  const analysisData = useMemo(() => {
    return bandStructureData.map(({ material, data }) => {
      if (!material || !data) return [];
      
      return [
        material.bandGap,
        material.electronMass,
        material.holeMass,
        material.dielectricConstant,
        temperatureK,
        strainPercent
      ];
    }).filter(row => row.length > 0);
  }, [bandStructureData, temperatureK, strainPercent]);

  const toggleMaterial = (materialName: string) => {
    setSelectedMaterials(prev => {
      if (prev.includes(materialName)) {
        return prev.filter(name => name !== materialName);
      } else {
        return [...prev, materialName];
      }
    });
  };

  return (
    <div className="w-full space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <GitCompare className="h-5 w-5" />
            Enhanced Band Structure Comparison
          </CardTitle>
          <CardDescription>
            Compare electronic band structures across different semiconductor materials
          </CardDescription>
        </CardHeader>
        <CardContent>
          {/* Material Selection */}
          <div className="space-y-4">
            <div>
              <Label className="text-base font-medium">Select Materials to Compare</Label>
              <div className="flex flex-wrap gap-2 mt-2">
                {materials.map((material) => (
                  <Button
                    key={material.name}
                    variant={selectedMaterials.includes(material.name) ? "default" : "outline"}
                    onClick={() => toggleMaterial(material.name)}
                    className="flex items-center gap-2"
                    style={{
                      backgroundColor: selectedMaterials.includes(material.name) ? material.color : undefined,
                      borderColor: material.color
                    }}
                  >
                    <div 
                      className="w-3 h-3 rounded-full"
                      style={{ backgroundColor: material.color }}
                    />
                    {material.name}
                  </Button>
                ))}
              </div>
            </div>

            {/* Visualization Options */}
            <div className="grid gap-4 md:grid-cols-3 p-4 bg-muted rounded-lg">
              <div className="flex items-center justify-between">
                <Label>Valence Bands</Label>
                <Switch
                  checked={showValenceBands}
                  onCheckedChange={setShowValenceBands}
                />
              </div>
              
              <div className="flex items-center justify-between">
                <Label>Conduction Bands</Label>
                <Switch
                  checked={showConductionBands}
                  onCheckedChange={setShowConductionBands}
                />
              </div>
              
              <div className="flex items-center justify-between">
                <Label>Density of States</Label>
                <Switch
                  checked={showDOS}
                  onCheckedChange={setShowDOS}
                />
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Main Visualization */}
      <div className="grid gap-6 lg:grid-cols-3">
        <div className="lg:col-span-2">
          <Card>
            <CardHeader>
              <CardTitle>Band Structure Comparison</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="h-96">
                <ScientificPlot
                  title="Electronic Band Structure"
                  data={bandStructurePlotData}
                  plotType="2d"
                  xLabel="k-vector (1/Å)"
                  yLabel="Energy (eV)"
                  showGrid={true}
                  showLegend={true}
                />
              </div>
            </CardContent>
          </Card>
        </div>

        {showDOS && (
          <div>
            <Card>
              <CardHeader>
                <CardTitle>Density of States</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="h-96">
                  <ScientificPlot
                    title="DOS Comparison"
                    data={dosPlotData}
                    plotType="2d"
                    xLabel="Energy (eV)"
                    yLabel="DOS (states/eV)"
                    showGrid={true}
                    showLegend={true}
                  />
                </div>
              </CardContent>
            </Card>
          </div>
        )}
      </div>

      {/* Analysis Tabs */}
      <Tabs defaultValue="parameters" className="w-full">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="parameters">Parameters</TabsTrigger>
          <TabsTrigger value="properties">Material Properties</TabsTrigger>
          <TabsTrigger value="analysis">ML Analysis</TabsTrigger>
          <TabsTrigger value="export">Export</TabsTrigger>
        </TabsList>

        <TabsContent value="parameters" className="space-y-4">
          <div className="grid gap-4 md:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Environmental Conditions</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <Label>Temperature: {temperatureK} K</Label>
                  <Slider
                    value={[temperatureK]}
                    onValueChange={([value]) => setTemperatureK(value)}
                    min={4}
                    max={800}
                    step={10}
                  />
                </div>
                
                <div className="space-y-2">
                  <Label>Strain: {strainPercent.toFixed(1)}%</Label>
                  <Slider
                    value={[strainPercent]}
                    onValueChange={([value]) => setStrainPercent(value)}
                    min={-5}
                    max={5}
                    step={0.1}
                  />
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Analysis Type</CardTitle>
              </CardHeader>
              <CardContent>
                <Select value={analysisType} onValueChange={(value: any) => setAnalysisType(value)}>
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="bandgap">Band Gap Analysis</SelectItem>
                    <SelectItem value="effective_mass">Effective Mass</SelectItem>
                    <SelectItem value="mobility">Carrier Mobility</SelectItem>
                  </SelectContent>
                </Select>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="properties" className="space-y-4">
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
            {materials.filter(m => selectedMaterials.includes(m.name)).map((material) => (
              <Card key={material.name}>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <div 
                      className="w-4 h-4 rounded-full"
                      style={{ backgroundColor: material.color }}
                    />
                    {material.name}
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-2">
                  <div className="flex justify-between">
                    <span className="text-sm text-muted-foreground">Lattice Constant:</span>
                    <span className="text-sm font-medium">{material.latticeConstant} Å</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-muted-foreground">Band Gap:</span>
                    <span className="text-sm font-medium">{material.bandGap} eV</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-muted-foreground">Electron Mass:</span>
                    <span className="text-sm font-medium">{material.electronMass} m₀</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-muted-foreground">Hole Mass:</span>
                    <span className="text-sm font-medium">{material.holeMass} m₀</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-muted-foreground">Dielectric Const:</span>
                    <span className="text-sm font-medium">{material.dielectricConstant}</span>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>

        <TabsContent value="analysis">
          <MLAnalysis
            data={analysisData}
            parameters={{
              temperature: temperatureK,
              strain: strainPercent,
              numMaterials: selectedMaterials.length
            }}
            simulationType="band_structure_comparison"
          />
        </TabsContent>

        <TabsContent value="export">
          <Card>
            <CardHeader>
              <CardTitle>Export Options</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="text-sm text-muted-foreground">
                  Export comparison data and visualizations
                </div>
                <div className="flex gap-2">
                  <Button variant="outline" className="flex items-center gap-2">
                    <Download className="h-4 w-4" />
                    Export Band Data
                  </Button>
                  <Button variant="outline" className="flex items-center gap-2">
                    <Download className="h-4 w-4" />
                    Export DOS Data
                  </Button>
                  <Button variant="outline" className="flex items-center gap-2">
                    <Download className="h-4 w-4" />
                    Export Comparison Report
                  </Button>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}