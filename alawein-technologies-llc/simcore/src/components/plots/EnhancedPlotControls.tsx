import React from 'react';
import { Card } from '@/components/ui/card';
import { Label } from '@/components/ui/label';
import { Switch } from '@/components/ui/switch';
import { Eye, EyeOff } from 'lucide-react';

// Minimal trace shape for controls
type TraceLike = { name: string } & Record<string, any>;

interface EnhancedPlotControlsProps {
  traces: TraceLike[];
  traceVisibility: Record<string, boolean>;
  toggleTraceVisibility: (name: string) => void;
  plotState: { autoRange: boolean; showGrid: boolean; showLegend: boolean };
  setPlotState: React.Dispatch<React.SetStateAction<{ autoRange: boolean; showGrid: boolean; showLegend: boolean }>>;
  validationInfo?: { isValid: boolean; message: string; details?: string[] };
}

export const EnhancedPlotControls: React.FC<EnhancedPlotControlsProps> = ({
  traces,
  traceVisibility,
  toggleTraceVisibility,
  plotState,
  setPlotState,
  validationInfo,
}) => {
  return (
    <Card className="p-4 bg-muted/20 border-border/30">
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {/* Trace Visibility Controls */}
        <div className="space-y-3">
          <Label className="text-sm font-medium">Trace Visibility</Label>
          <div className="space-y-2">
            {traces.map((trace) => (
              <div key={trace.name} className="flex items-center space-x-2">
                <Switch
                  id={`trace-${trace.name}`}
                  checked={!!traceVisibility[trace.name]}
                  onCheckedChange={() => toggleTraceVisibility(trace.name)}
                  className="data-[state=checked]:bg-primary"
                />
                <Label
                  htmlFor={`trace-${trace.name}`}
                  className="text-xs cursor-pointer flex items-center gap-2"
                >
                  {traceVisibility[trace.name] ? (
                    <Eye className="w-3 h-3" />
                  ) : (
                    <EyeOff className="w-3 h-3" />
                  )}
                  {trace.name}
                </Label>
              </div>
            ))}
          </div>
        </div>

        {/* Display Options */}
        <div className="space-y-3">
          <Label className="text-sm font-medium">Display Options</Label>
          <div className="space-y-2">
            <div className="flex items-center space-x-2">
              <Switch
                id="show-grid"
                checked={plotState.showGrid}
                onCheckedChange={(checked) =>
                  setPlotState((prev) => ({ ...prev, showGrid: checked }))
                }
              />
              <Label htmlFor="show-grid" className="text-xs cursor-pointer">
                Show Grid
              </Label>
            </div>
            <div className="flex items-center space-x-2">
              <Switch
                id="show-legend"
                checked={plotState.showLegend}
                onCheckedChange={(checked) =>
                  setPlotState((prev) => ({ ...prev, showLegend: checked }))
                }
              />
              <Label htmlFor="show-legend" className="text-xs cursor-pointer">
                Show Legend
              </Label>
            </div>
          </div>
        </div>

        {/* Plot Statistics */}
        <div className="space-y-3">
          <Label className="text-sm font-medium">Plot Statistics</Label>
          <div className="space-y-1 text-xs text-muted-foreground">
            <div>Total Traces: {traces.length}</div>
            <div>
              Visible Traces: {Object.values(traceVisibility).filter(Boolean).length}
            </div>
            <div>
              Data Points: {
                traces.reduce(
                  (sum, t) => sum + (Array.isArray((t as any).x) ? (t as any).x.length : 0),
                  0
                )
              }
            </div>
            {validationInfo && (
              <div className="mt-2 p-2 bg-card/50 rounded border">
                <div
                  className={`font-medium ${validationInfo.isValid ? 'text-green-600' : 'text-red-600'}`}
                >
                  {validationInfo.message}
                </div>
                {validationInfo.details &&
                  validationInfo.details.map((detail, index) => (
                    <div key={index} className="text-xs mt-1">
                      {detail}
                    </div>
                  ))}
              </div>
            )}
          </div>
        </div>
      </div>
    </Card>
  );
};

export default EnhancedPlotControls;
