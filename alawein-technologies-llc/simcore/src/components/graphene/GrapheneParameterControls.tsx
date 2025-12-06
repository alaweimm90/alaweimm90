import React from 'react';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Slider } from '@/components/ui/slider';
import { Label } from '@/components/ui/label';
import { Switch } from '@/components/ui/switch';
import { Badge } from '@/components/ui/badge';
import { RotateCcw, Download, Settings } from 'lucide-react';
import { InlineMath } from '@/components/ui/Math';
import { GrapheneParameters } from '@/lib/graphene-physics-exact';

interface GrapheneParameterControlsProps {
  parameters: GrapheneParameters;
  onParametersChange: (params: GrapheneParameters) => void;
  showValence: boolean;
  showConduction: boolean;
  onShowValenceChange: (show: boolean) => void;
  onShowConductionChange: (show: boolean) => void;
  onReset: () => void;
  onExport: () => void;
  fermiVelocity: number;
}

export function GrapheneParameterControls({
  parameters,
  onParametersChange,
  showValence,
  showConduction,
  onShowValenceChange,
  onShowConductionChange,
  onReset,
  onExport,
  fermiVelocity
}: GrapheneParameterControlsProps) {
  
  const updateParameter = (key: keyof GrapheneParameters, value: number) => {
    if (key === 'strain') return;
    onParametersChange({ ...parameters, [key]: value });
  };

  const updateStrain = (component: keyof GrapheneParameters['strain'], value: number) => {
    onParametersChange({
      ...parameters,
      strain: { ...parameters.strain, [component]: value }
    });
  };

  return (
    <Card className="p-6 space-y-6">
      <div className="flex items-center justify-between">
        <h3 className="text-xl font-semibold font-serif flex items-center gap-2">
          <Settings className="w-5 h-5 text-primary" />
          Parameters
        </h3>
        <div className="flex gap-2">
          <Button variant="ghost" size="sm" onClick={onReset}>
            <RotateCcw className="w-4 h-4" />
          </Button>
          <Button variant="outline" size="sm" onClick={onExport}>
            <Download className="w-4 h-4" />
          </Button>
        </div>
      </div>

      {/* Physical Parameters Documentation */}
      <div className="space-y-4">
        <h4 className="font-medium text-lg">⚛️ Physical Parameters</h4>
        
        <div className="text-xs space-y-2 p-3 bg-muted/30 rounded-lg">
          <div className="flex justify-between">
            <span>C-C bond length:</span>
            <span className="font-mono">1.42 Å</span>
          </div>
          <div className="flex justify-between">
            <span>Lattice constant <InlineMath math="a" />:</span>
            <span className="font-mono">2.46 Å</span>
          </div>
          <div className="flex justify-between">
            <span>Current t:</span>
            <span className="font-mono">{parameters.t1.toFixed(2)} eV</span>
          </div>
          <div className="flex justify-between">
            <span>Current t':</span>
            <span className="font-mono">{parameters.t2.toFixed(3)} eV</span>
          </div>
          <div className="flex justify-between">
            <span>Ratio t'/t:</span>
            <span className="font-mono">{(parameters.t2/parameters.t1).toFixed(3)}</span>
          </div>
          <div className="flex justify-between">
            <span>k-path sampling:</span>
            <span className="font-mono">n = 100</span>
          </div>
          <div className="flex justify-between">
            <span>Total k-points:</span>
            <span className="font-mono">300</span>
          </div>
        </div>

        <div>
          <Label className="text-sm font-medium flex items-center gap-2 mb-2">
            NN Hopping <InlineMath math="t" /> (eV)
          </Label>
          <Slider
            value={[parameters.t1]}
            onValueChange={([value]) => updateParameter('t1', value)}
            max={3.0}
            min={2.5}
            step={0.1}
            className="mt-2"
          />
          <div className="text-xs text-muted-foreground mt-1">
            {parameters.t1.toFixed(1)} eV
          </div>
        </div>
        
        <div>
          <Label className="text-sm font-medium flex items-center gap-2 mb-2">
            NNN Hopping <InlineMath math="t'" /> (eV)
          </Label>
          <Slider
            value={[parameters.t2]}
            onValueChange={([value]) => updateParameter('t2', value)}
            max={0.3}
            min={0.0}
            step={0.01}
            className="mt-2"
          />
          <div className="text-xs text-muted-foreground mt-1">
            {parameters.t2.toFixed(2)} eV = {(parameters.t2/parameters.t1*10).toFixed(0)}% × t
          </div>
        </div>


        <div>
          <Label className="text-sm font-medium flex items-center gap-2 mb-2">
            Onsite Energy (eV)
          </Label>
          <Slider
            value={[parameters.onsite]}
            onValueChange={([value]) => updateParameter('onsite', value)}
            max={1.0}
            min={-1.0}
            step={0.01}
            className="mt-2"
          />
          <div className="text-xs text-muted-foreground mt-1">
            {parameters.onsite.toFixed(3)} eV
          </div>
        </div>
      </div>

      {/* Strain Parameters */}
      <div className="space-y-4">
        <h4 className="font-medium text-lg">Strain Tensor (%)</h4>
        
        <div>
          <Label className="text-sm font-medium flex items-center gap-2 mb-2">
            <InlineMath math="\varepsilon_{xx}" />
          </Label>
          <Slider
            value={[parameters.strain.exx * 100]}
            onValueChange={([value]) => updateStrain('exx', value / 100)}
            max={10}
            min={-10}
            step={0.1}
            className="mt-2"
          />
          <div className="text-xs text-muted-foreground mt-1">
            {(parameters.strain.exx * 100).toFixed(1)}%
          </div>
        </div>

        <div>
          <Label className="text-sm font-medium flex items-center gap-2 mb-2">
            <InlineMath math="\varepsilon_{yy}" />
          </Label>
          <Slider
            value={[parameters.strain.eyy * 100]}
            onValueChange={([value]) => updateStrain('eyy', value / 100)}
            max={10}
            min={-10}
            step={0.1}
            className="mt-2"
          />
          <div className="text-xs text-muted-foreground mt-1">
            {(parameters.strain.eyy * 100).toFixed(1)}%
          </div>
        </div>

        <div>
          <Label className="text-sm font-medium flex items-center gap-2 mb-2">
            <InlineMath math="\varepsilon_{xy}" />
          </Label>
          <Slider
            value={[parameters.strain.exy * 100]}
            onValueChange={([value]) => updateStrain('exy', value / 100)}
            max={10}
            min={-10}
            step={0.1}
            className="mt-2"
          />
          <div className="text-xs text-muted-foreground mt-1">
            {(parameters.strain.exy * 100).toFixed(1)}%
          </div>
        </div>
      </div>

      {/* Display Controls */}
      <div className="space-y-4">
        <h4 className="font-medium text-lg">Display Options</h4>
        
        <div className="flex items-center justify-between">
          <Label htmlFor="show-valence" className="text-sm font-medium">
            Show Valence Band
          </Label>
          <Switch
            id="show-valence"
            checked={showValence}
            onCheckedChange={onShowValenceChange}
          />
        </div>

        <div className="flex items-center justify-between">
          <Label htmlFor="show-conduction" className="text-sm font-medium">
            Show Conduction Band
          </Label>
          <Switch
            id="show-conduction"
            checked={showConduction}
            onCheckedChange={onShowConductionChange}
          />
        </div>
      </div>

      {/* High-Symmetry Path */}
      <div className="space-y-4">
        <h4 className="font-medium text-lg">🔄 k-space Path</h4>
        
        <div className="text-xs space-y-2 p-3 bg-muted/30 rounded-lg">
          <div className="text-center font-semibold">Γ → K → M → Γ</div>
          <div className="grid grid-cols-2 gap-2">
            <div>Γ: x = 0</div>
            <div>K: x = 100</div>
            <div>M: x = 200</div>
            <div>Γ: x = 300</div>
          </div>
        </div>
      </div>

      {/* Energy Alignment */}
      <div className="space-y-4">
        <h4 className="font-medium text-lg">🔼 Energy Alignment</h4>
        
        <div className="text-xs space-y-2 p-3 bg-muted/30 rounded-lg">
          <div className="flex justify-between">
            <span>NN-only shift:</span>
            <span className="font-mono">0 eV</span>
          </div>
          <div className="flex justify-between">
            <span>NN+NNN shift:</span>
            <span className="font-mono">+{(3 * parameters.t2).toFixed(2)} eV</span>
          </div>
          <div className="flex justify-between">
            <span>Dirac point at:</span>
            <span className="font-mono">E = 0</span>
          </div>
        </div>
      </div>

      {/* Derived Quantities */}
      <div className="space-y-4">
        <h4 className="font-medium text-lg">📊 Derived Quantities</h4>
        
        <div className="space-y-2">
          <div className="flex justify-between items-center">
            <span className="text-sm">Fermi Velocity:</span>
            <Badge variant="outline" className="font-mono">
              {(fermiVelocity / 1e6).toFixed(2)} × 10⁶ m/s
            </Badge>
          </div>
          
          <div className="flex justify-between items-center">
            <span className="text-sm">DOS Energy Range:</span>
            <Badge variant="outline" className="font-mono">
              -3 to +3 eV
            </Badge>
          </div>
        </div>
      </div>
    </Card>
  );
}