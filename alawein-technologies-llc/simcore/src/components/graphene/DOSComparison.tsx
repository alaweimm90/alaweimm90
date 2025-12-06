import React from 'react';
import Plot from 'react-plotly.js';
import { DOSResult } from '@/lib/graphene-physics-exact';
import { UNIFIED_SCIENTIFIC_PLOT_SYSTEM as PLOT, PHYSICS_AXIS_LABELS } from '@/lib/scientific-plot-system';

interface DOSComparisonProps {
  dataNN: DOSResult;
  dataNNN: DOSResult;
}

export function DOSComparison({ dataNN, dataNNN }: DOSComparisonProps) {
  if (!dataNN.energies.length || !dataNNN.energies.length) {
    return (
      <div className="flex items-center justify-center h-96 text-muted-foreground">
        Loading DOS comparison...
      </div>
    );
  }

  const plotData = [
    {
      x: dataNN.dos,
      y: dataNN.energies,
      type: 'scatter' as const,
      mode: 'lines' as const,
      name: 'NN only',
      line: PLOT.lineStyles.dos.numerical,
      hovertemplate: 'E/t: %{y:.2f}<br>DOS: %{x:.3f}<extra></extra>'
    },
    {
      x: dataNNN.dos,
      y: dataNNN.energies,
      type: 'scatter' as const,
      mode: 'lines' as const,
      name: 'NN+NNN',
      line: PLOT.lineStyles.dos.analytical,
      hovertemplate: 'E/t: %{y:.2f}<br>DOS: %{x:.3f}<extra></extra>'
    }
  ];

  const layout = PLOT.createLayout(
    'Density of States',
    PHYSICS_AXIS_LABELS.dos,
    PHYSICS_AXIS_LABELS.energy,
    { energyRange: { min: -3, max: 3 }, showLegend: true, height: 400 }
  );

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
