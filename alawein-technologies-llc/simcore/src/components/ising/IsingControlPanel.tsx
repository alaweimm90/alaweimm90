import React from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Slider } from '@/components/ui/slider';
import { Label } from '@/components/ui/label';
import { Switch } from '@/components/ui/switch';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Separator } from '@/components/ui/separator';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { 
  Play, Pause, RotateCcw, Settings, Thermometer, Magnet, 
  Activity, TrendingUp, Download, Shuffle, Target 
} from 'lucide-react';
import { useIsingStore } from './IsingSimulationStore';
import { InlineMath } from '@/components/ui/Math';

interface IsingControlPanelProps {
  onRunPhaseTransition: () => void;
  onExportData: () => void;
  onSetConfiguration: (config: 'all_up' | 'all_down' | 'random' | 'checkerboard') => void;
  currentMagnetization?: number;
  currentEnergy?: number;
  isAnalyzing?: boolean;
}

export const IsingControlPanel: React.FC<IsingControlPanelProps> = ({
  onRunPhaseTransition,
  onExportData,
  onSetConfiguration,
  currentMagnetization = 0,
  currentEnergy = 0,
  isAnalyzing = false
}) => {
  const {
    size, setSize,
    temperature, setTemperature,
    magneticField, setMagneticField,
    algorithm, setAlgorithm,
    isRunning, toggleSimulation,
    resetSimulation,
    animationSpeed, setAnimationSpeed,
    showDomains, toggleDomains,
    showCorrelation, toggleCorrelation,
    equilibrationSteps, setEquilibrationSteps,
    measurementSteps, setMeasurementSteps,
    skipInitialSteps, toggleSkipInitialSteps,
    step,
    magnetizationHistory,
    energyHistory,
    heatCapacityHistory,
    susceptibilityHistory
  } = useIsingStore();

  const criticalTemperature = 2.269; // Onsager's exact result
  const nearCritical = Math.abs(temperature - criticalTemperature) < 0.1;

  return (
    <div className="space-y-6">
      {/* Simulation Controls */}
      <Card>
        <CardHeader>
          <div className="flex items-center gap-2">
            <Settings className="w-5 h-5 text-primary" />
            <CardTitle>Simulation Control</CardTitle>
          </div>
          <CardDescription>
            Monte Carlo simulation parameters and algorithm selection
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* Algorithm Selection */}
          <div>
            <Label className="text-sm font-medium">Monte Carlo Algorithm</Label>
            <Select value={algorithm} onValueChange={setAlgorithm}>
              <SelectTrigger className="mt-1">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="metropolis">Metropolis</SelectItem>
                <SelectItem value="wolff">Wolff Cluster</SelectItem>
                <SelectItem value="heat_bath">Heat Bath</SelectItem>
              </SelectContent>
            </Select>
            <div className="text-xs text-muted-foreground mt-1">
              {algorithm === 'wolff' && 'Efficient near critical temperature - flips entire clusters'}
              {algorithm === 'metropolis' && 'Standard single-spin flip algorithm'}
              {algorithm === 'heat_bath' && 'Samples from equilibrium distribution directly'}
            </div>
          </div>

          {/* Temperature Control */}
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <Label className="text-sm font-medium flex items-center gap-2">
                <Thermometer className="w-4 h-4" />
                Temperature
              </Label>
              <Badge variant={nearCritical ? "destructive" : "secondary"} className="text-xs">
                T = {temperature.toFixed(3)}
              </Badge>
            </div>
            <Slider
              value={[temperature]}
              onValueChange={([value]) => setTemperature(value)}
              min={0.1}
              max={5.0}
              step={0.01}
              className="w-full"
            />
            <div className="flex justify-between text-xs text-muted-foreground">
              <span>Ordered</span>
              <span className="font-medium">
                <InlineMath math="T_c \approx 2.269" />
              </span>
              <span>Disordered</span>
            </div>
            {nearCritical && (
              <Alert>
                <AlertDescription className="text-xs">
                  🔥 Near critical temperature! Expect critical slowing down and large fluctuations.
                </AlertDescription>
              </Alert>
            )}
          </div>

          {/* Magnetic Field */}
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <Label className="text-sm font-medium flex items-center gap-2">
                <Magnet className="w-4 h-4" />
                External Field
              </Label>
              <Badge variant="outline" className="text-xs">
                H = {magneticField.toFixed(3)}
              </Badge>
            </div>
            <Slider
              value={[magneticField]}
              onValueChange={([value]) => setMagneticField(value)}
              min={-2}
              max={2}
              step={0.01}
              className="w-full"
            />
            <div className="text-xs text-muted-foreground">
              External magnetic field breaks symmetry
            </div>
          </div>

          {/* System Size */}
          <div className="space-y-2">
            <Label className="text-sm font-medium">
              System Size: {size}×{size} = {size * size} spins
            </Label>
            <Slider
              value={[size]}
              onValueChange={([value]) => setSize(value)}
              min={16}
              max={128}
              step={8}
              className="w-full"
            />
            <div className="text-xs text-muted-foreground">
              Larger systems reduce finite-size effects
            </div>
          </div>

          {/* Animation Speed */}
          <div className="space-y-2">
            <Label className="text-sm font-medium">
              Update Interval: {animationSpeed}ms
            </Label>
            <Slider
              value={[animationSpeed]}
              onValueChange={([value]) => setAnimationSpeed(value)}
              min={10}
              max={1000}
              step={10}
              className="w-full"
            />
          </div>
        </CardContent>
      </Card>

      {/* Measurement Settings */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Target className="w-5 h-5 text-primary" />
            Measurement Settings
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex items-center justify-between">
            <Label className="text-sm">Skip Initial Steps</Label>
            <Switch checked={skipInitialSteps} onCheckedChange={toggleSkipInitialSteps} />
          </div>
          
          <div className="space-y-2">
            <Label className="text-sm">Equilibration Steps: {equilibrationSteps}</Label>
            <Slider
              value={[equilibrationSteps]}
              onValueChange={([value]) => setEquilibrationSteps(value)}
              min={100}
              max={5000}
              step={100}
              className="w-full"
            />
          </div>
          
          <div className="space-y-2">
            <Label className="text-sm">Measurement Steps: {measurementSteps}</Label>
            <Slider
              value={[measurementSteps]}
              onValueChange={([value]) => setMeasurementSteps(value)}
              min={100}
              max={5000}
              step={100}
              className="w-full"
            />
          </div>
        </CardContent>
      </Card>

      {/* Visualization Options */}
      <Card>
        <CardHeader>
          <CardTitle>Visualization</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex items-center justify-between">
            <Label className="text-sm">Show Domain Structure</Label>
            <Switch checked={showDomains} onCheckedChange={toggleDomains} />
          </div>
          
          <div className="flex items-center justify-between">
            <Label className="text-sm">Show Correlation Data</Label>
            <Switch checked={showCorrelation} onCheckedChange={toggleCorrelation} />
          </div>
        </CardContent>
      </Card>

      {/* Action Buttons */}
      <Card>
        <CardContent className="p-4 space-y-3">
          <div className="grid grid-cols-2 gap-2">
            <Button
              onClick={toggleSimulation}
              variant={isRunning ? "destructive" : "default"}
              disabled={isAnalyzing}
            >
              {isRunning ? (
                <>
                  <Pause className="w-4 h-4 mr-2" />
                  Pause
                </>
              ) : (
                <>
                  <Play className="w-4 h-4 mr-2" />
                  Run
                </>
              )}
            </Button>
            
            <Button onClick={resetSimulation} variant="outline">
              <RotateCcw className="w-4 h-4 mr-2" />
              Reset
            </Button>
          </div>

          {/* Initial Configuration */}
          <div className="grid grid-cols-2 gap-2">
            <Button
              onClick={() => onSetConfiguration('all_up')}
              variant="outline"
              size="sm"
              disabled={isRunning}
            >
              All ↑
            </Button>
            <Button
              onClick={() => onSetConfiguration('all_down')}
              variant="outline"
              size="sm"
              disabled={isRunning}
            >
              All ↓
            </Button>
            <Button
              onClick={() => onSetConfiguration('checkerboard')}
              variant="outline"
              size="sm"
              disabled={isRunning}
            >
              Checker
            </Button>
            <Button
              onClick={() => onSetConfiguration('random')}
              variant="outline"
              size="sm"
              disabled={isRunning}
            >
              <Shuffle className="w-3 h-3 mr-1" />
              Random
            </Button>
          </div>

          <Button 
            onClick={onRunPhaseTransition} 
            variant="outline" 
            className="w-full"
            disabled={isRunning || isAnalyzing}
          >
            {isAnalyzing ? (
              <>
                <Activity className="w-4 h-4 mr-2 animate-spin" />
                Analyzing...
              </>
            ) : (
              <>
                <TrendingUp className="w-4 h-4 mr-2" />
                Phase Transition Sweep
              </>
            )}
          </Button>

          <Button 
            onClick={onExportData} 
            variant="outline" 
            className="w-full"
          >
            <Download className="w-4 h-4 mr-2" />
            Export CSV Data
          </Button>
        </CardContent>
      </Card>

      {/* Current Status */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Activity className="w-5 h-5 text-primary" />
            Current State
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-3 text-sm">
          <div className="flex justify-between">
            <span>Monte Carlo Steps:</span>
            <Badge variant="outline">{step}</Badge>
          </div>
          
          <div className="flex justify-between">
            <span>Magnetization:</span>
            <Badge variant="outline">
              {currentMagnetization.toFixed(4)}
            </Badge>
          </div>
          
          <div className="flex justify-between">
            <span>Energy per spin:</span>
            <Badge variant="outline">
              {currentEnergy.toFixed(3)}
            </Badge>
          </div>

          {heatCapacityHistory.length > 0 && (
            <>
              <div className="flex justify-between">
                <span>Heat Capacity:</span>
                <Badge variant="outline">
                  {heatCapacityHistory[heatCapacityHistory.length - 1]?.toFixed(3)}
                </Badge>
              </div>
              
              <div className="flex justify-between">
                <span>Susceptibility:</span>
                <Badge variant="outline">
                  {susceptibilityHistory[susceptibilityHistory.length - 1]?.toFixed(3)}
                </Badge>
              </div>
            </>
          )}

          {magnetizationHistory.length > 100 && (
            <div className="space-y-1">
              <span className="text-xs text-muted-foreground">Equilibration Progress:</span>
              <Progress 
                value={Math.min(100, (step / equilibrationSteps) * 100)} 
                className="h-2"
              />
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
};

export default IsingControlPanel;