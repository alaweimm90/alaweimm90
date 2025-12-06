import React from 'react';
import Plot from 'react-plotly.js';
import { BandStructureResult } from '@/lib/graphene-physics-exact';
import { UNIFIED_SCIENTIFIC_PLOT_SYSTEM as PLOT, PHYSICS_AXIS_LABELS, PHYSICS_ENERGY_RANGES } from '@/lib/scientific-plot-system';

interface BandStructureComparisonProps {
  dataNN: BandStructureResult;
  dataNNN: BandStructureResult;
  showValence: boolean;
  showConduction: boolean;
  fermiLevel?: number;
  snapshots?: { label: string; kPath: number[]; valence: number[]; conduction: number[]; color?: string }[];
}

export function BandStructureComparison({ 
  dataNN, 
  dataNNN,
  showValence, 
  showConduction, 
  fermiLevel = 0,
  snapshots = []
}: BandStructureComparisonProps) {
  if (!dataNN.kPath.length || !dataNNN.kPath.length) {
    return (
      <div className="flex items-center justify-center h-96 text-muted-foreground">
        Loading band structure comparison...
      </div>
    );
  }

  const plotData: any[] = [];

  // Build traces using unified scientific styles
  // NN bands (solid)
  if (showValence) {
    plotData.push(PLOT.createBandTrace(
      dataNN.kPath,
      dataNN.valence,
      'valence',
      'NN Valence'
    ));
  }

  if (showConduction) {
    plotData.push(PLOT.createBandTrace(
      dataNN.kPath,
      dataNN.conduction,
      'conduction',
      'NN Conduction'
    ));
  }

  // NN+NNN bands (dotted)
  if (showValence) {
    plotData.push({
      ...PLOT.createBandTrace(dataNNN.kPath, dataNNN.valence, 'valence', 'NN+NNN Valence'),
      line: { ...PLOT.lineStyles.bands.valence, dash: 'dot' }
    });
  }

  if (showConduction) {
    plotData.push({
      ...PLOT.createBandTrace(dataNNN.kPath, dataNNN.conduction, 'conduction', 'NN+NNN Conduction'),
      line: { ...PLOT.lineStyles.bands.conduction, dash: 'dot' }
    });
  }

  // Fermi level line
  if (fermiLevel !== undefined) {
    plotData.push(PLOT.createFermiTrace(
      [dataNN.kPath[0], dataNN.kPath[dataNN.kPath.length - 1]],
      fermiLevel,
      'Fermi Level'
    ));
  }

  // Optional snapshot overlays
  if (snapshots && snapshots.length) {
    snapshots.forEach((s) => {
      plotData.push({
        x: s.kPath,
        y: s.valence,
        type: 'scatter' as const,
        mode: 'lines' as const,
        name: `Snapshot ${s.label} Valence`,
        line: { color: s.color || '#A3A3A3', width: 2, dash: 'dashdot' },
        showlegend: true
      });
      plotData.push({
        x: s.kPath,
        y: s.conduction,
        type: 'scatter' as const,
        mode: 'lines' as const,
        name: `Snapshot ${s.label} Conduction`,
        line: { color: s.color || '#A3A3A3', width: 2, dash: 'dashdot' },
        showlegend: true
      });
    });
  }

  // Vertical K and K' markers
  const kIndex = dataNN.labels.findIndex((l: string) => l === 'K');
  const kpIndex = dataNN.labels.findIndex((l: string) => l === "K'");
  const yMin = -3, yMax = 3;
  const shapes: any[] = [];
  if (kIndex >= 0) {
    const x = dataNN.labelPositions[kIndex];
    shapes.push({ type: 'line', x0: x, x1: x, y0: yMin, y1: yMax, line: { color: PLOT.colors.physics.diracPoint, width: 1, dash: 'dash' } });
  }
  if (kpIndex >= 0) {
    const x = dataNN.labelPositions[kpIndex];
    shapes.push({ type: 'line', x0: x, x1: x, y0: yMin, y1: yMax, line: { color: PLOT.colors.physics.diracPoint, width: 1, dash: 'dash' } });
  }

  const layout = {
    ...PLOT.createLayout(
      'Electronic Band Structure',
      PHYSICS_AXIS_LABELS.momentum,
      PHYSICS_AXIS_LABELS.energy,
      { energyRange: PHYSICS_ENERGY_RANGES.graphene, showLegend: true, height: 450 }
    ),
    xaxis: {
      ...PLOT.layout.xaxis,
      tickvals: dataNN.labelPositions,
      ticktext: dataNN.labels,
      tickmode: 'array' as const
    },
    shapes
  } as any;

  return (
    <div className="w-full h-full">
      <Plot
        data={plotData}
        layout={layout}
        config={PLOT.config}
        style={{ width: '100%', height: '100%' }}
      />
    </div>
  );
}