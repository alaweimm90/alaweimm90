import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Slider } from '@/components/ui/slider';
import { Switch } from '@/components/ui/switch';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import { 
  Play, 
  Pause, 
  RotateCcw, 
  Download, 
  Trash2, 
  Settings,
  Zap,
  Eye,
  Atom
} from 'lucide-react';
import { useLLGStore, vec3 } from '@/lib/llg-store';

export function LLGControlPanel() {
  const {
    // State
    magnetization,
    appliedField,
    gilbert_damping,
    anisotropy,
    initialTheta,
    initialPhi,
    isRunning,
    time,
    timeData,
    
    // Visualization options
    showEffectiveField,
    showTorque,
    showTrajectory,
    show2DPlots,
    
    // Actions
    startSimulation,
    pauseSimulation,
    resetSimulation,
    setAppliedField,
    setGilbertDamping,
    setAnisotropy,
    setInitialTheta,
    setInitialPhi,
    setRandomInitialPosition,
    toggleEffectiveField,
    toggleTorque,
    toggleTrajectory,
    toggle2DPlots,
    exportTrajectory,
    clearHistory
  } = useLLGStore();

  const fieldDirections = {
    'x': [1, 0, 0] as [number, number, number],
    'y': [0, 1, 0] as [number, number, number],
    'z': [0, 0, 1] as [number, number, number],
    'xy': [1, 1, 0] as [number, number, number],
    'xyz': [1, 1, 1] as [number, number, number]
  };

  const [selectedFieldDir, setSelectedFieldDir] = React.useState('z');
  const [selectedEasyAxis, setSelectedEasyAxis] = React.useState('z');

  React.useEffect(() => {
    setAppliedField(
      appliedField.magnitude, 
      vec3.normalize(fieldDirections[selectedFieldDir as keyof typeof fieldDirections])
    );
  }, [selectedFieldDir]);

  React.useEffect(() => {
    setAnisotropy(
      anisotropy.strength,
      vec3.normalize(fieldDirections[selectedEasyAxis as keyof typeof fieldDirections])
    );
  }, [selectedEasyAxis]);

  return (
    <div className="h-full overflow-y-auto">
      <div className="space-y-6 p-4">
        {/* Simulation Control */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Settings className="w-5 h-5" />
              Simulation Control
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex gap-2">
              <Button
                onClick={isRunning ? pauseSimulation : startSimulation}
                className="flex-1"
                variant={isRunning ? "destructive" : "default"}
              >
                {isRunning ? (
                  <>
                    <Pause className="h-4 w-4 mr-2" />
                    Pause
                  </>
                ) : (
                  <>
                    <Play className="h-4 w-4 mr-2" />
                    Start
                  </>
                )}
              </Button>
              
              <Button onClick={resetSimulation} variant="outline">
                <RotateCcw className="h-4 w-4" />
              </Button>
            </div>

            <div className="grid grid-cols-2 gap-4 text-sm">
              <div>
                <Label className="text-muted-foreground">Time</Label>
                <div className="font-mono">{(time * 1e9).toFixed(2)} ns</div>
              </div>
              <div>
                <Label className="text-muted-foreground">Data Points</Label>
                <div className="font-mono">{timeData.length}</div>
              </div>
            </div>

            <div className="flex gap-2">
              <Button onClick={exportTrajectory} variant="outline" className="flex-1">
                <Download className="h-4 w-4 mr-2" />
                Export CSV
              </Button>
              <Button onClick={clearHistory} variant="outline">
                <Trash2 className="h-4 w-4" />
              </Button>
            </div>

            {/* Current State */}
            <Separator />
            <div className="space-y-3">
              <Label className="text-sm font-medium">Current State</Label>
              <div className="grid grid-cols-3 gap-4 text-sm font-mono">
                <div>
                  <Label className="text-muted-foreground">mx</Label>
                  <div className="text-red-500">{magnetization[0].toFixed(4)}</div>
                </div>
                <div>
                  <Label className="text-muted-foreground">my</Label>
                  <div className="text-green-500">{magnetization[1].toFixed(4)}</div>
                </div>
                <div>
                  <Label className="text-muted-foreground">mz</Label>
                  <div className="text-blue-500">{magnetization[2].toFixed(4)}</div>
                </div>
              </div>
              
              <div className="flex justify-between text-sm">
                <Label className="text-muted-foreground">|m|</Label>
                <Badge variant={Math.abs(vec3.magnitude(magnetization) - 1) < 1e-6 ? "default" : "destructive"}>
                  {vec3.magnitude(magnetization).toFixed(6)}
                </Badge>
              </div>
            </div>
          </CardContent>
        </Card>
        
        {/* Initial State */}
        <Card>
          <CardHeader>
            <CardTitle>Initial Magnetization State</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <Label>Current Position</Label>
              <div className="grid grid-cols-3 gap-2 text-sm font-mono">
                <div className="text-center">
                  <div className="text-muted-foreground">mx</div>
                  <div className="text-red-500">{magnetization[0].toFixed(3)}</div>
                </div>
                <div className="text-center">
                  <div className="text-muted-foreground">my</div>
                  <div className="text-green-500">{magnetization[1].toFixed(3)}</div>
                </div>
                <div className="text-center">
                  <div className="text-muted-foreground">mz</div>
                  <div className="text-blue-500">{magnetization[2].toFixed(3)}</div>
                </div>
              </div>
            </div>
            
            <Separator />
            
            <div className="space-y-2">
              <Label>Polar Angle θ: {(initialTheta * 180 / Math.PI).toFixed(1)}°</Label>
              <Slider
                value={[initialTheta]}
                onValueChange={([value]) => setInitialTheta(value)}
                min={0}
                max={Math.PI}
                step={0.01}
                className="w-full"
              />
              <div className="text-xs text-muted-foreground">
                θ = 0°: North pole (+z) | θ = 90°: Equator | θ = 180°: South pole (-z)
              </div>
            </div>
            
            <div className="space-y-2">
              <Label>Azimuthal Angle φ: {(initialPhi * 180 / Math.PI).toFixed(1)}°</Label>
              <Slider
                value={[initialPhi]}
                onValueChange={([value]) => setInitialPhi(value)}
                min={0}
                max={2 * Math.PI}
                step={0.01}
                className="w-full"
              />
              <div className="text-xs text-muted-foreground">
                φ = 0°: +x axis | φ = 90°: +y axis | φ = 180°: -x axis | φ = 270°: -y axis
              </div>
            </div>
            
            <Button 
              onClick={setRandomInitialPosition}
              variant="outline"
              className="w-full"
            >
              🎲 Random Position
            </Button>
          </CardContent>
        </Card>
        
        {/* Physics Parameters */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Atom className="w-5 h-5" />
              Physics Parameters
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            {/* Applied Field */}
            <div className="space-y-3">
              <Label className="flex items-center gap-2">
                <Zap className="h-4 w-4" />
                Applied Field
              </Label>
              <div>
                <Label className="flex items-center justify-between">
                  Magnitude (Tesla)
                  <span className="text-sm text-muted-foreground">{appliedField.magnitude.toFixed(3)} T</span>
                </Label>
                <Slider
                  value={[appliedField.magnitude]}
                  onValueChange={([value]) => setAppliedField(value, appliedField.direction)}
                  min={0}
                  max={2}
                  step={0.001}
                  className="mt-2"
                />
              </div>

              <div>
                <Label>Direction</Label>
                <Select value={selectedFieldDir} onValueChange={setSelectedFieldDir}>
                  <SelectTrigger className="mt-2">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="x">+X axis</SelectItem>
                    <SelectItem value="y">+Y axis</SelectItem>
                    <SelectItem value="z">+Z axis</SelectItem>
                    <SelectItem value="xy">XY diagonal</SelectItem>
                    <SelectItem value="xyz">XYZ diagonal</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>

            <Separator />

            {/* Gilbert Damping */}
            <div>
              <Label className="flex items-center justify-between">
                Damping Parameter α
                <span className="text-sm text-muted-foreground">{gilbert_damping.toFixed(3)}</span>
              </Label>
              <Slider
                value={[gilbert_damping]}
                onValueChange={([value]) => setGilbertDamping(value)}
                min={0}
                max={1}
                step={0.001}
                className="mt-2"
              />
              <div className="text-xs text-muted-foreground mt-2">
                α = 0: No damping (perpetual precession)<br/>
                α &gt; 0: Energy dissipation → equilibrium
              </div>
            </div>

            <Separator />

            {/* Anisotropy */}
            <div className="space-y-4">
              <Label>Magnetic Anisotropy</Label>
              <div>
                <Label className="flex items-center justify-between">
                  Anisotropy Strength K
                  <span className="text-sm text-muted-foreground">{anisotropy.strength.toFixed(4)} T</span>
                </Label>
                <Slider
                  value={[anisotropy.strength]}
                  onValueChange={([value]) => setAnisotropy(value, anisotropy.easyAxis)}
                  min={0}
                  max={0.1}
                  step={0.001}
                  className="mt-2"
                />
              </div>

              <div>
                <Label>Easy Axis</Label>
                <Select value={selectedEasyAxis} onValueChange={setSelectedEasyAxis}>
                  <SelectTrigger className="mt-2">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="x">X axis</SelectItem>
                    <SelectItem value="y">Y axis</SelectItem>
                    <SelectItem value="z">Z axis</SelectItem>
                    <SelectItem value="xy">XY diagonal</SelectItem>
                    <SelectItem value="xyz">XYZ diagonal</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>
          </CardContent>
        </Card>
        
        {/* Visualization Options */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Eye className="w-5 h-5" />
              Visualization Options
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <Label htmlFor="effective-field">Effective Field</Label>
                <Switch 
                  id="effective-field" 
                  checked={showEffectiveField} 
                  onCheckedChange={toggleEffectiveField} 
                />
              </div>

              <div className="flex items-center justify-between">
                <Label htmlFor="torque">Torque Vector</Label>
                <Switch 
                  id="torque" 
                  checked={showTorque} 
                  onCheckedChange={toggleTorque} 
                />
              </div>

              <div className="flex items-center justify-between">
                <Label htmlFor="trajectory">Trajectory</Label>
                <Switch 
                  id="trajectory" 
                  checked={showTrajectory} 
                  onCheckedChange={toggleTrajectory} 
                />
              </div>

              <div className="flex items-center justify-between">
                <Label htmlFor="time-plots">Time Evolution Plots</Label>
                <Switch 
                  id="time-plots" 
                  checked={show2DPlots} 
                  onCheckedChange={toggle2DPlots} 
                />
              </div>
            </div>

            <Separator />

            <div className="space-y-2 text-sm">
              <Label>Vector Legend</Label>
              <div className="space-y-1">
                <div className="flex items-center gap-2">
                  <div className="w-4 h-2 bg-red-500 rounded"></div>
                  <span>Magnetization (m)</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-4 h-2 bg-green-500 rounded"></div>
                  <span>Effective Field (H_eff)</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-4 h-2 bg-amber-500 rounded"></div>
                  <span>Torque (τ = m × H_eff)</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-4 h-2 bg-purple-500 rounded"></div>
                  <span>Trajectory</span>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}