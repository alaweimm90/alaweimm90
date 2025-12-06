import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Slider } from '@/components/ui/slider';
import { Switch } from '@/components/ui/switch';
import { Label } from '@/components/ui/label';
import { Input } from '@/components/ui/input';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { 
  Atom, 
  Grid3x3, 
  Eye, 
  Download, 
  RotateCcw, 
  Plus, 
  Minus,
  Layers,
  Box,
  GitBranch,
  Compass
} from 'lucide-react';
import { useCrystalStore, crystalStructures } from '@/lib/crystal-store';

export function StructurePanel() {
  const {
    selectedStructure,
    setSelectedStructure,
    latticeConstant,
    setLatticeConstant,
    supercell,
    setSupercell,
    showPrimitiveCell,
    resetView
  } = useCrystalStore();

  const structure = crystalStructures[selectedStructure];

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Atom className="h-5 w-5" />
          Crystal Structure
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
        <div>
          <Label>Structure Type</Label>
          <Select value={selectedStructure} onValueChange={setSelectedStructure}>
            <SelectTrigger>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {Object.entries(crystalStructures).map(([key, struct]) => (
                <SelectItem key={key} value={key}>
                  {struct.name}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          {structure && (
            <div className="mt-2 space-y-1">
              <div className="flex gap-2">
                <Badge variant="secondary">{structure.system}</Badge>
                <Badge variant="outline">{structure.latticeType}</Badge>
              </div>
              <p className="text-sm text-muted-foreground">
                Space Group: {structure.spaceGroup} | Point Group: {structure.pointGroup}
              </p>
            </div>
          )}
        </div>

        <div>
          <Label className="flex items-center justify-between">
            Lattice Constant
            <span className="text-sm text-muted-foreground">{latticeConstant.toFixed(2)} Å</span>
          </Label>
          <Slider
            value={[latticeConstant]}
            onValueChange={([value]) => setLatticeConstant(value)}
            min={0.5}
            max={5.0}
            step={0.1}
            className="mt-2"
          />
        </div>

        <div>
          <Label>Supercell Dimensions</Label>
          <div className="grid grid-cols-3 gap-2 mt-2">
            {supercell.map((dim, i) => (
              <div key={i} className="space-y-1">
                <Label className="text-xs">{'abc'[i]}</Label>
                <Input
                  type="number"
                  value={dim}
                  onChange={(e) => {
                    const newSupercell = [...supercell] as [number, number, number];
                    newSupercell[i] = Math.max(1, parseInt(e.target.value) || 1);
                    setSupercell(newSupercell);
                  }}
                  min={1}
                  max={5}
                  className="text-center"
                />
              </div>
            ))}
          </div>
        </div>

        {structure && (
          <div>
            <Label>Lattice Parameters</Label>
            <div className="grid grid-cols-2 gap-2 mt-2 text-sm">
              <div>a = {(structure.latticeParameters.a * latticeConstant).toFixed(2)} Å</div>
              <div>α = {structure.latticeParameters.alpha}°</div>
              <div>b = {(structure.latticeParameters.b * latticeConstant).toFixed(2)} Å</div>
              <div>β = {structure.latticeParameters.beta}°</div>
              <div>c = {(structure.latticeParameters.c * latticeConstant).toFixed(2)} Å</div>
              <div>γ = {structure.latticeParameters.gamma}°</div>
            </div>
          </div>
        )}

        <Button onClick={resetView} variant="outline" className="w-full">
          <RotateCcw className="h-4 w-4 mr-2" />
          Reset View
        </Button>
      </CardContent>
    </Card>
  );
}

export function VisualizationPanel() {
  const {
    showAtoms,
    showBonds,
    showUnitCell,
    showAxes,
    atomScale,
    bondCutoff,
    toggleShowAtoms,
    toggleShowBonds,
    toggleShowUnitCell,
    toggleShowAxes,
    setAtomScale,
    setBondCutoff
  } = useCrystalStore();

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Eye className="h-5 w-5" />
          Visualization
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <Label htmlFor="show-atoms">Show Atoms</Label>
            <Switch id="show-atoms" checked={showAtoms} onCheckedChange={toggleShowAtoms} />
          </div>
          
          <div className="flex items-center justify-between">
            <Label htmlFor="show-bonds">Show Bonds</Label>
            <Switch id="show-bonds" checked={showBonds} onCheckedChange={toggleShowBonds} />
          </div>
          
          <div className="flex items-center justify-between">
            <Label htmlFor="show-unit-cell">Show Unit Cell</Label>
            <Switch id="show-unit-cell" checked={showUnitCell} onCheckedChange={toggleShowUnitCell} />
          </div>
          
          <div className="flex items-center justify-between">
            <Label htmlFor="show-axes">Show Axes</Label>
            <Switch id="show-axes" checked={showAxes} onCheckedChange={toggleShowAxes} />
          </div>
        </div>

        <div>
          <Label className="flex items-center justify-between">
            Atom Scale
            <span className="text-sm text-muted-foreground">{atomScale.toFixed(1)}×</span>
          </Label>
          <Slider
            value={[atomScale]}
            onValueChange={([value]) => setAtomScale(value)}
            min={0.1}
            max={3.0}
            step={0.1}
            className="mt-2"
          />
        </div>

        <div>
          <Label className="flex items-center justify-between">
            Bond Cutoff
            <span className="text-sm text-muted-foreground">{bondCutoff.toFixed(1)} Å</span>
          </Label>
          <Slider
            value={[bondCutoff]}
            onValueChange={([value]) => setBondCutoff(value)}
            min={0.5}
            max={5.0}
            step={0.1}
            className="mt-2"
          />
        </div>
      </CardContent>
    </Card>
  );
}

export function MillerPlanesPanel() {
  const { millerPlanes, addMillerPlane, removeMillerPlane, toggleMillerPlane } = useCrystalStore();
  const [h, setH] = useState(1);
  const [k, setK] = useState(0);
  const [l, setL] = useState(0);

  const handleAddPlane = () => {
    addMillerPlane([h, k, l]);
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Layers className="h-5 w-5" />
          Miller Planes
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div>
          <Label>Add Miller Plane (hkl)</Label>
          <div className="grid grid-cols-4 gap-2 mt-2">
            <Input
              type="number"
              value={h}
              onChange={(e) => setH(parseInt(e.target.value) || 0)}
              placeholder="h"
            />
            <Input
              type="number"
              value={k}
              onChange={(e) => setK(parseInt(e.target.value) || 0)}
              placeholder="k"
            />
            <Input
              type="number"
              value={l}
              onChange={(e) => setL(parseInt(e.target.value) || 0)}
              placeholder="l"
            />
            <Button onClick={handleAddPlane} size="sm">
              <Plus className="h-4 w-4" />
            </Button>
          </div>
        </div>

        <div className="space-y-2">
          {millerPlanes.map((plane, index) => (
            <div key={index} className="flex items-center justify-between p-2 border rounded">
              <div className="flex items-center gap-2">
                <div 
                  className="w-4 h-4 rounded" 
                  style={{ backgroundColor: plane.color }}
                />
                <span className="font-mono">
                  ({plane.indices[0]}{plane.indices[1]}{plane.indices[2]})
                </span>
              </div>
              <div className="flex items-center gap-1">
                <Switch
                  checked={plane.visible}
                  onCheckedChange={() => toggleMillerPlane(index)}
                />
                <Button
                  onClick={() => removeMillerPlane(index)}
                  size="sm"
                  variant="ghost"
                >
                  <Minus className="h-3 w-3" />
                </Button>
              </div>
            </div>
          ))}
        </div>

        {millerPlanes.length === 0 && (
          <p className="text-sm text-muted-foreground text-center py-4">
            No Miller planes defined
          </p>
        )}
      </CardContent>
    </Card>
  );
}

export function AdvancedPanel() {
  const {
    showSymmetryOperations,
    showReciprocalLattice,
    showBrillouinZone,
    showWignerSeitzCell,
    toggleSymmetryOperations,
    toggleReciprocalLattice,
    toggleBrillouinZone,
    toggleWignerSeitzCell,
    exportFormat,
    exportStructure
  } = useCrystalStore();

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <GitBranch className="h-5 w-5" />
          Advanced Features
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <Label htmlFor="symmetry-ops">Symmetry Operations</Label>
            <Switch 
              id="symmetry-ops" 
              checked={showSymmetryOperations} 
              onCheckedChange={toggleSymmetryOperations} 
            />
          </div>
          
          <div className="flex items-center justify-between">
            <Label htmlFor="reciprocal">Reciprocal Lattice</Label>
            <Switch 
              id="reciprocal" 
              checked={showReciprocalLattice} 
              onCheckedChange={toggleReciprocalLattice} 
            />
          </div>
          
          <div className="flex items-center justify-between">
            <Label htmlFor="brillouin">Brillouin Zone</Label>
            <Switch 
              id="brillouin" 
              checked={showBrillouinZone} 
              onCheckedChange={toggleBrillouinZone} 
            />
          </div>
          
          <div className="flex items-center justify-between">
            <Label htmlFor="wigner-seitz">Wigner-Seitz Cell</Label>
            <Switch 
              id="wigner-seitz" 
              checked={showWignerSeitzCell} 
              onCheckedChange={toggleWignerSeitzCell} 
            />
          </div>
        </div>

        <div>
          <Label>Export Structure</Label>
          <div className="flex gap-2 mt-2">
            <Button onClick={exportStructure} className="flex-1">
              <Download className="h-4 w-4 mr-2" />
              Export CIF
            </Button>
            <Button onClick={exportStructure} variant="outline" className="flex-1">
              <Download className="h-4 w-4 mr-2" />
              Export XYZ
            </Button>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

export function ControlPanels() {
  return (
    <div className="space-y-6">
      <Tabs defaultValue="structure" className="w-full">
        <TabsList className="grid grid-cols-4 w-full">
          <TabsTrigger value="structure">Structure</TabsTrigger>
          <TabsTrigger value="visual">Visual</TabsTrigger>
          <TabsTrigger value="planes">Planes</TabsTrigger>
          <TabsTrigger value="advanced">Advanced</TabsTrigger>
        </TabsList>
        
        <TabsContent value="structure" className="space-y-4">
          <StructurePanel />
        </TabsContent>
        
        <TabsContent value="visual" className="space-y-4">
          <VisualizationPanel />
        </TabsContent>
        
        <TabsContent value="planes" className="space-y-4">
          <MillerPlanesPanel />
        </TabsContent>
        
        <TabsContent value="advanced" className="space-y-4">
          <AdvancedPanel />
        </TabsContent>
      </Tabs>
    </div>
  );
}