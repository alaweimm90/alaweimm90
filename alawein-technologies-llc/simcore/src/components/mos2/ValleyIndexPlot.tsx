import React from 'react';
import Plot from 'react-plotly.js';
import { UNIFIED_SCIENTIFIC_PLOT_SYSTEM as PLOT, PHYSICS_AXIS_LABELS } from '@/lib/scientific-plot-system';

interface ValleyData {
  kx: number;
  ky: number;
  valleyIndex: number;
  berryCurvature: number;
}

interface ValleyIndexPlotProps {
  valleyData: ValleyData[];
}

export const ValleyIndexPlot: React.FC<ValleyIndexPlotProps> = ({ valleyData }) => {
  if (!valleyData.length) {
    return (
      <div className="flex items-center justify-center h-96 text-muted-foreground">
        Loading valley index...
      </div>
    );
  }

  // Separate K and K' valley points
  const kValleyPoints = valleyData.filter(p => p.valleyIndex > 0);
  const kPrimeValleyPoints = valleyData.filter(p => p.valleyIndex < 0);

  const plotData = [];

  // K valley points (τ = +1, green)
  plotData.push({
    x: kValleyPoints.map(p => p.kx),
    y: kValleyPoints.map(p => p.ky),
    type: 'scatter' as const,
    mode: 'markers' as const,
    name: 'K valley (τ = +1)',
    marker: {
      size: 3,
      color: '#10b981',
      opacity: 0.8
    },
    hovertemplate: 'K valley<br>kₓ: %{x:.3f} Å⁻¹<br>kᵧ: %{y:.3f} Å⁻¹<extra></extra>'
  });

  // K' valley points (τ = -1, orange)
  plotData.push({
    x: kPrimeValleyPoints.map(p => p.kx),
    y: kPrimeValleyPoints.map(p => p.ky),
    type: 'scatter' as const,
    mode: 'markers' as const,
    name: 'K\' valley (τ = -1)',
    marker: {
      size: 3,
      color: '#f97316',
      opacity: 0.8
    },
    hovertemplate: 'K\' valley<br>kₓ: %{x:.3f} Å⁻¹<br>kᵧ: %{y:.3f} Å⁻¹<extra></extra>'
  });

const layout = PLOT.createLayout(
    'Valley Index Distribution',
    PHYSICS_AXIS_LABELS.kx,
    PHYSICS_AXIS_LABELS.ky
  );

  layout.xaxis = {
    ...layout.xaxis,
    range: [-2, 2],
    title: { text: 'kₓ (Å⁻¹)', font: { size: 14 } }
  };

  layout.yaxis = {
    ...layout.yaxis,
    range: [-2, 2],
    scaleanchor: 'x',
    scaleratio: 1,
    title: { text: 'kᵧ (Å⁻¹)', font: { size: 14 } }
  };

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
};