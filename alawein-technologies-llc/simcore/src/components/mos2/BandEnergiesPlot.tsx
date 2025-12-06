import React from 'react';
import Plot from 'react-plotly.js';
import { UNIFIED_SCIENTIFIC_PLOT_SYSTEM as PLOT, PHYSICS_AXIS_LABELS } from '@/lib/scientific-plot-system';

interface ValleyData {
  kx: number;
  ky: number;
  energyConduction: number;
  energyValence: number;
}

interface BandEnergiesPlotProps {
  valleyData: ValleyData[];
}

export const BandEnergiesPlot: React.FC<BandEnergiesPlotProps> = ({ valleyData }) => {
  if (!valleyData.length) {
    return (
      <div className="flex items-center justify-center h-96 text-muted-foreground">
        Loading band energies...
      </div>
    );
  }

  // Create energy color mapping
  const energies = valleyData.map(p => p.energyConduction);
  const minEnergy = Math.min(...energies);
  const maxEnergy = Math.max(...energies);

  const plotData = [];

  // Conduction band energies
  plotData.push({
    x: valleyData.map(p => p.kx),
    y: valleyData.map(p => p.ky),
    z: energies,
    type: 'scatter' as const,
    mode: 'markers' as const,
    name: 'Conduction Band Energy',
    marker: {
      size: 4,
      color: energies,
      colorscale: 'Viridis',
      showscale: true,
      colorbar: {
        title: 'Energy (eV)',
        titleside: 'right'
      },
      cmin: minEnergy,
      cmax: maxEnergy
    },
    hovertemplate: 'Band Energy<br>kₓ: %{x:.3f} Å⁻¹<br>kᵧ: %{y:.3f} Å⁻¹<br>E: %{marker.color:.3f} eV<extra></extra>'
  });

const layout = PLOT.createLayout(
    'Conduction Band Energies',
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
        className="w-full h-full"
        useResizeHandler
      />
    </div>
  );
};