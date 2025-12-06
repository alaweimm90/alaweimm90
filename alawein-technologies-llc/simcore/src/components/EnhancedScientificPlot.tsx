/**
 * Enhanced Scientific Plotting Component
 * 
 * Advanced plotting system with improved performance, theming, and scientific features
 * Built on the unified scientific plotting system with additional enhancements.
 * 
 * @author Dr. Meshal Alawein - UC Berkeley
 * @version 2.0.0 - Phase 1 Enhancements
 */

import React, { memo, useMemo, useState, useCallback } from 'react';
import Plot from 'react-plotly.js';
import { Card } from '@/components/ui/card';
import { 
  UNIFIED_SCIENTIFIC_PLOT_SYSTEM,
  PHYSICS_ENERGY_RANGES,
  PHYSICS_AXIS_LABELS
} from '@/lib/scientific-plot-system';
import { EnhancedPlotControls } from './plots/EnhancedPlotControls';
import { EnhancedPlotHeader } from './plots/EnhancedPlotHeader';
import { EnhancedPlotFooter } from './plots/EnhancedPlotFooter';

interface PlotTrace {
  x: number[];
  y: number[];
  name: string;
  type: 'line' | 'scatter' | 'markers';
  style?: {
    color?: string;
    width?: number;
    dash?: string;
    size?: number;
  };
  visible?: boolean;
}

interface EnhancedScientificPlotProps {
  title: string;
  xLabel: string;
  yLabel: string;
  traces: PlotTrace[];
  plotType: 'band-structure' | 'dos' | 'time-series' | 'phase-diagram' | 'custom';
  energyRange?: { min: number; max: number };
  showControls?: boolean;
  showLegend?: boolean;
  showGrid?: boolean;
  showExportOptions?: boolean;
  height?: number;
  className?: string;
  onDataExport?: (data: any) => void;
  onPlotUpdate?: (update: any) => void;
  validationInfo?: {
    isValid: boolean;
    message: string;
    details?: string[];
  };
}

export const EnhancedScientificPlot: React.FC<EnhancedScientificPlotProps> = memo(({
  title,
  xLabel,
  yLabel,
  traces,
  plotType,
  energyRange,
  showControls = true,
  showLegend = true,
  showGrid = false,
  showExportOptions = true,
  height = 450,
  className = '',
  onDataExport,
  onPlotUpdate,
  validationInfo
}) => {
  const [traceVisibility, setTraceVisibility] = useState<Record<string, boolean>>(
    traces.reduce((acc, trace, index) => ({
      ...acc,
      [trace.name]: trace.visible !== false
    }), {})
  );
  const [showAdvancedControls, setShowAdvancedControls] = useState(false);
  const [plotState, setPlotState] = useState({
    autoRange: true,
    showGrid: showGrid,
    showLegend: showLegend
  });

  // Memoize plot data with visibility filtering
  const plotData = useMemo(() => {
    return traces
      .filter(trace => traceVisibility[trace.name])
      .map(trace => {
        return UNIFIED_SCIENTIFIC_PLOT_SYSTEM.createTrace(
          trace.x,
          trace.y,
          trace.name,
          trace.type,
          trace.style
        );
      });
  }, [traces, traceVisibility]);

  // Memoize layout configuration
  const layout = useMemo(() => {
    const baseLayout = UNIFIED_SCIENTIFIC_PLOT_SYSTEM.createLayout(
      title,
      xLabel,
      yLabel,
      {
        energyRange: energyRange || (plotType === 'band-structure' ? PHYSICS_ENERGY_RANGES.graphene : undefined),
        showLegend: plotState.showLegend,
        height
      }
    );

    // Add grid lines if enabled
    if (plotState.showGrid) {
      return {
        ...baseLayout,
        xaxis: {
          ...baseLayout.xaxis,
          showgrid: true,
          gridcolor: `${UNIFIED_SCIENTIFIC_PLOT_SYSTEM.colors.muted}20`,
          gridwidth: 1
        },
        yaxis: {
          ...baseLayout.yaxis,
          showgrid: true,
          gridcolor: `${UNIFIED_SCIENTIFIC_PLOT_SYSTEM.colors.muted}20`,
          gridwidth: 1
        }
      };
    }

    return baseLayout;
  }, [title, xLabel, yLabel, energyRange, plotType, height, plotState]);

  // Handle trace visibility toggle
  const toggleTraceVisibility = useCallback((traceName: string) => {
    setTraceVisibility(prev => ({
      ...prev,
      [traceName]: !prev[traceName]
    }));
  }, []);

  // Handle data export
  const handleExport = useCallback((format: 'svg' | 'png' | 'json') => {
    if (format === 'json' && onDataExport) {
      const exportData = {
        title,
        xLabel,
        yLabel,
        traces: traces.map(trace => ({
          name: trace.name,
          x: trace.x,
          y: trace.y,
          visible: traceVisibility[trace.name]
        })),
        metadata: {
          plotType,
          timestamp: new Date().toISOString(),
          energyRange
        }
      };
      onDataExport(exportData);
    }
    // SVG/PNG export handled by Plotly natively
  }, [title, xLabel, yLabel, traces, traceVisibility, plotType, energyRange, onDataExport]);

  // Get plot-specific styling based on type
  const getPlotTypeConfig = useMemo(() => {
    const configs = {
      'band-structure': {
        badge: 'Band Structure',
        badgeVariant: 'default' as const,
        energyRange: PHYSICS_ENERGY_RANGES.graphene,
        xAxisLabel: PHYSICS_AXIS_LABELS.momentum,
        yAxisLabel: PHYSICS_AXIS_LABELS.energy
      },
      'dos': {
        badge: 'Density of States',
        badgeVariant: 'secondary' as const,
        energyRange: PHYSICS_ENERGY_RANGES.general,
        xAxisLabel: PHYSICS_AXIS_LABELS.dos,
        yAxisLabel: PHYSICS_AXIS_LABELS.energy
      },
      'time-series': {
        badge: 'Time Evolution',
        badgeVariant: 'outline' as const,
        energyRange: undefined,
        xAxisLabel: PHYSICS_AXIS_LABELS.time,
        yAxisLabel: 'Observable'
      },
      'phase-diagram': {
        badge: 'Phase Diagram',
        badgeVariant: 'destructive' as const,
        energyRange: undefined,
        xAxisLabel: PHYSICS_AXIS_LABELS.temperature,
        yAxisLabel: PHYSICS_AXIS_LABELS.order
      },
      'custom': {
        badge: 'Custom Plot',
        badgeVariant: 'outline' as const,
        energyRange: undefined,
        xAxisLabel: xLabel,
        yAxisLabel: yLabel
      }
    };
    return configs[plotType];
  }, [plotType, xLabel, yLabel]);

  if (!plotData.length) {
    return (
      <Card className={`p-6 ${className}`}>
        <div className="flex items-center justify-center h-64 text-muted-foreground">
          <div className="text-center space-y-2">
            <div className="text-lg font-medium">No Data Available</div>
            <p className="text-sm">Add traces to display the plot</p>
          </div>
        </div>
      </Card>
    );
  }

  return (
    <Card className={`p-4 space-y-4 ${className} bg-card/50 backdrop-blur-sm border-border/50`}>
      {/* Header with controls */}
      <EnhancedPlotHeader
        title={title}
        badgeLabel={getPlotTypeConfig.badge}
        badgeVariant={getPlotTypeConfig.badgeVariant}
        validationInfo={validationInfo}
        showControls={showControls}
        showExportOptions={showExportOptions}
        onExport={() => handleExport('json')}
        onToggleControls={() => setShowAdvancedControls(!showAdvancedControls)}
      />

      {/* Advanced Controls Panel */}
      {showAdvancedControls && (
        <EnhancedPlotControls
          traces={traces}
          traceVisibility={traceVisibility}
          toggleTraceVisibility={toggleTraceVisibility}
          plotState={plotState}
          setPlotState={setPlotState}
          validationInfo={validationInfo}
        />
      )}

      {/* Main Plot */}
      <div className="relative" style={{ height }}>
        <Plot
          data={plotData}
          layout={layout}
          config={{
            ...UNIFIED_SCIENTIFIC_PLOT_SYSTEM.config,
            toImageButtonOptions: {
              ...UNIFIED_SCIENTIFIC_PLOT_SYSTEM.config.toImageButtonOptions,
              filename: `${title.replace(/\s+/g, '_').toLowerCase()}_${Date.now()}`
            }
          }}
          useResizeHandler
          className="scientific-plot-enhanced w-full h-full rounded border border-border/20"
          onUpdate={onPlotUpdate}
        />
      </div>

      {/* Footer with plot info */}
      <EnhancedPlotFooter traceCount={plotData.length} />
    </Card>
  );
});

// Display name for debugging
EnhancedScientificPlot.displayName = 'EnhancedScientificPlot';

// Export convenience functions for common plot types
export const createBandStructurePlot = (
  kPath: number[],
  valenceBand: number[],
  conductionBand: number[],
  title: string = 'Electronic Band Structure'
): EnhancedScientificPlotProps => ({
  title,
  xLabel: PHYSICS_AXIS_LABELS.momentum,
  yLabel: PHYSICS_AXIS_LABELS.energy,
  plotType: 'band-structure',
  traces: [
    {
      x: kPath,
      y: valenceBand,
      name: 'Valence Band',
      type: 'line',
      style: UNIFIED_SCIENTIFIC_PLOT_SYSTEM.lineStyles.bands.valence
    },
    {
      x: kPath,
      y: conductionBand,
      name: 'Conduction Band',
      type: 'line',
      style: UNIFIED_SCIENTIFIC_PLOT_SYSTEM.lineStyles.bands.conduction
    }
  ],
  energyRange: PHYSICS_ENERGY_RANGES.graphene,
  showControls: true,
  showLegend: true
});

export const createDOSPlot = (
  energies: number[],
  dos: number[],
  title: string = 'Density of States'
): EnhancedScientificPlotProps => ({
  title,
  xLabel: PHYSICS_AXIS_LABELS.dos,
  yLabel: PHYSICS_AXIS_LABELS.energy,
  plotType: 'dos',
  traces: [
    {
      x: dos,
      y: energies,
      name: 'DOS',
      type: 'line',
      style: UNIFIED_SCIENTIFIC_PLOT_SYSTEM.lineStyles.dos.numerical
    }
  ],
  energyRange: PHYSICS_ENERGY_RANGES.general,
  showControls: true,
  showLegend: false
});