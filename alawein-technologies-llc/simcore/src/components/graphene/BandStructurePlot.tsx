import React, { memo, useMemo } from 'react';
import Plot from 'react-plotly.js';
import { BandStructureResult } from '@/lib/graphene-tb-physics';
import { 
  UNIFIED_SCIENTIFIC_PLOT_SYSTEM,
  PHYSICS_ENERGY_RANGES,
  PHYSICS_AXIS_LABELS
} from '@/lib/scientific-plot-system';

interface BandStructurePlotProps {
  data: BandStructureResult;
  showValence: boolean;
  showConduction: boolean;
  fermiLevel?: number;
}

export const BandStructurePlot = memo(function BandStructurePlot({ 
  data, 
  showValence, 
  showConduction, 
  fermiLevel = 0 
}: BandStructurePlotProps) {
  // Component implementation starts here
  // Memoize plot data to prevent unnecessary recalculations
  const plotData = useMemo(() => {
    if (!data.kPath.length) return [];
    
    const traces = [];

    if (showValence) {
      traces.push(UNIFIED_SCIENTIFIC_PLOT_SYSTEM.createBandTrace(
        data.kPath,
        data.valence,
        'valence',
        'Valence Band'
      ));
    }

    if (showConduction) {
      traces.push(UNIFIED_SCIENTIFIC_PLOT_SYSTEM.createBandTrace(
        data.kPath,
        data.conduction,
        'conduction',
        'Conduction Band'
      ));
    }

    // Add Fermi level reference line
    if (fermiLevel !== undefined && data.kPath.length > 0) {
      traces.push(UNIFIED_SCIENTIFIC_PLOT_SYSTEM.createFermiTrace(
        [data.kPath[0], data.kPath[data.kPath.length - 1]],
        fermiLevel,
        'Fermi Level'
      ));
    }

    return traces;
  }, [data.kPath, data.valence, data.conduction, showValence, showConduction, fermiLevel]);

  // Memoize layout to prevent unnecessary recalculations  
  const layout = useMemo(() => {
    const baseLayout = UNIFIED_SCIENTIFIC_PLOT_SYSTEM.createLayout(
      'Electronic Band Structure',
      PHYSICS_AXIS_LABELS.momentum,
      PHYSICS_AXIS_LABELS.energy,
      { 
        energyRange: PHYSICS_ENERGY_RANGES.graphene,
        showLegend: true,
        height: 450
      }
    );

    // Add k-point labels for high-symmetry points
    if (data.labelPositions && data.labels) {
      return {
        ...baseLayout,
        xaxis: {
          ...baseLayout.xaxis,
          tickvals: data.labelPositions,
          ticktext: data.labels,
          tickmode: 'array' as const
        }
      };
    }

    return baseLayout;
  }, [data.labelPositions, data.labels]);

  // Loading state with theme-aware styling
  if (!data.kPath.length) {
    return (
      <div className="flex items-center justify-center h-96 text-muted-foreground bg-card/30 rounded-lg border border-border/50">
        <div className="text-center space-y-2">
          <div className="animate-pulse">
            <div className="w-8 h-8 bg-primary/20 rounded-full mx-auto mb-2"></div>
          </div>
          <p className="text-sm">Calculating band structure...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="w-full h-full bg-card/30 rounded-lg border border-border/20">
      <Plot
        data={plotData}
        layout={layout}
        config={UNIFIED_SCIENTIFIC_PLOT_SYSTEM.config}
        className="w-full h-full"
        useResizeHandler
      />
    </div>
  );
});