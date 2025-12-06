import React from 'react';
import Plot from 'react-plotly.js';
import { UNIFIED_SCIENTIFIC_PLOT_SYSTEM as PLOT, PHYSICS_AXIS_LABELS } from '@/lib/scientific-plot-system';

interface ValleyData {
  kx: number;
  ky: number;
  spinTextureX: number;
  spinTextureY: number;
  valleyIndex: number;
}

interface SpinTexturePlotProps {
  valleyData: ValleyData[];
}

export const SpinTexturePlot: React.FC<SpinTexturePlotProps> = ({ valleyData }) => {
  if (!valleyData.length) {
    return (
      <div className="flex items-center justify-center h-96 text-muted-foreground">
        Loading spin texture...
      </div>
    );
  }

  // Calculate spin angle for color mapping
  const plotData = [];

  // Subsample for clarity (every 4th point)
  const subsampledData = valleyData.filter((_, i) => i % 4 === 0);

  // Spin texture arrows
  for (const point of subsampledData) {
    const spinAngle = Math.atan2(point.spinTextureY, point.spinTextureX);
    const hue = ((spinAngle + Math.PI) / (2 * Math.PI)) * 360; // Map to 0-360°
    
    plotData.push({
      x: [point.kx, point.kx + point.spinTextureX * 0.1],
      y: [point.ky, point.ky + point.spinTextureY * 0.1],
      type: 'scatter' as const,
      mode: 'lines' as const,
      line: {
        color: `hsl(${hue}, 80%, 60%)`,
        width: 2
      },
      showlegend: false,
      hoverinfo: 'skip' as const
    });
  }

  // Add background points colored by spin angle
  const backgroundPoints = valleyData.map(point => {
    const spinAngle = Math.atan2(point.spinTextureY, point.spinTextureX);
    const hue = ((spinAngle + Math.PI) / (2 * Math.PI)) * 360;
    return {
      x: point.kx,
      y: point.ky,
      color: hue
    };
  });

  plotData.unshift({
    x: backgroundPoints.map(p => p.x),
    y: backgroundPoints.map(p => p.y),
    type: 'scatter' as const,
    mode: 'markers' as const,
    name: 'Spin Direction',
    marker: {
      size: 4,
      color: backgroundPoints.map(p => p.color),
      colorscale: 'HSV',
      showscale: true,
      colorbar: {
        title: 'Spin Angle',
        tickmode: 'array',
        tickvals: [0, 90, 180, 270, 360],
        ticktext: ['0°', '90°', '180°', '270°', '360°']
      }
    },
    hovertemplate: 'Spin texture<br>kₓ: %{x:.3f} Å⁻¹<br>kᵧ: %{y:.3f} Å⁻¹<extra></extra>'
  });

const layout = PLOT.createLayout(
    'Spin Texture Field',
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