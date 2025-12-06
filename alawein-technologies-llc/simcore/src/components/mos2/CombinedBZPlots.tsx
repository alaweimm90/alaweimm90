import React from 'react';
import Plot from 'react-plotly.js';
import { UNIFIED_SCIENTIFIC_PLOT_SYSTEM as PLOT } from '@/lib/scientific-plot-system';

interface ValleyData {
  kx: number;
  ky: number;
  valleyIndex: number;
  berryCurvature: number;
  spinTextureX: number;
  spinTextureY: number;
  energyConduction: number;
  energyValence: number;
}

interface CombinedBZPlotsProps {
  valleyData: ValleyData[];
}

export const CombinedBZPlots: React.FC<CombinedBZPlotsProps> = ({ valleyData }) => {
  if (!valleyData.length) {
    return (
      <div className="flex items-center justify-center h-96 text-muted-foreground">
        Loading BZ visualizations...
      </div>
    );
  }

  // Prepare data for all four subplot types
  const plotData = [];

  // 1. Valley Index (subplot 1) - Top Left
  const kValleyPoints = valleyData.filter(p => p.valleyIndex > 0);
  const kPrimeValleyPoints = valleyData.filter(p => p.valleyIndex < 0);

  plotData.push({
    x: kValleyPoints.map(p => p.kx),
    y: kValleyPoints.map(p => p.ky),
    type: 'scatter' as const,
    mode: 'markers' as const,
    name: 'K valley (τ = +1)',
    marker: { size: 3, color: '#10b981', opacity: 0.8 },
    xaxis: 'x1',
    yaxis: 'y1',
    hovertemplate: 'K valley<br>kₓ: %{x:.3f}<br>kᵧ: %{y:.3f}<extra></extra>'
  });

  plotData.push({
    x: kPrimeValleyPoints.map(p => p.kx),
    y: kPrimeValleyPoints.map(p => p.ky),
    type: 'scatter' as const,
    mode: 'markers' as const,
    name: 'K\' valley (τ = -1)',
    marker: { size: 3, color: '#f97316', opacity: 0.8 },
    xaxis: 'x1',
    yaxis: 'y1',
    hovertemplate: 'K\' valley<br>kₓ: %{x:.3f}<br>kᵧ: %{y:.3f}<extra></extra>'
  });

  // 2. Berry Curvature (subplot 2) - Top Right
  const berryValues = valleyData.map(p => p.berryCurvature);
  const maxBerry = Math.max(...berryValues.map(Math.abs));

  plotData.push({
    x: valleyData.map(p => p.kx),
    y: valleyData.map(p => p.ky),
    type: 'scatter' as const,
    mode: 'markers' as const,
    name: 'Berry Curvature Ω(k)',
    marker: {
      size: 4,
      color: berryValues,
      colorscale: [
        [0, '#ef4444'], [0.5, '#ffffff'], [1, '#3b82f6']
      ],
      showscale: true,
      colorbar: {
        title: 'Ω(k) (Å²)',
        x: 0.48,
        len: 0.4
      },
      cmin: -maxBerry,
      cmax: maxBerry
    },
    xaxis: 'x2',
    yaxis: 'y2',
    hovertemplate: 'Berry Curvature<br>kₓ: %{x:.3f}<br>kᵧ: %{y:.3f}<br>Ω: %{marker.color:.2e}<extra></extra>'
  });

  // 3. Spin Texture (subplot 3) - Bottom Left
  const spinAngles = valleyData.map(p => {
    const angle = Math.atan2(p.spinTextureY, p.spinTextureX);
    return ((angle + Math.PI) / (2 * Math.PI)) * 360;
  });

  plotData.push({
    x: valleyData.map(p => p.kx),
    y: valleyData.map(p => p.ky),
    type: 'scatter' as const,
    mode: 'markers' as const,
    name: 'Spin Texture',
    marker: {
      size: 4,
      color: spinAngles,
      colorscale: 'HSV',
      showscale: true,
      colorbar: {
        title: 'Spin Angle (°)',
        x: 0.98,
        len: 0.4,
        y: 0.25
      }
    },
    xaxis: 'x3',
    yaxis: 'y3',
    hovertemplate: 'Spin Texture<br>kₓ: %{x:.3f}<br>kᵧ: %{y:.3f}<br>Angle: %{marker.color:.0f}°<extra></extra>'
  });

  // 4. Band Energies (subplot 4) - Bottom Right
  const energies = valleyData.map(p => p.energyConduction);
  const minEnergy = Math.min(...energies);
  const maxEnergy = Math.max(...energies);

  plotData.push({
    x: valleyData.map(p => p.kx),
    y: valleyData.map(p => p.ky),
    type: 'scatter' as const,
    mode: 'markers' as const,
    name: 'Band Energies',
    marker: {
      size: 4,
      color: energies,
      colorscale: 'Viridis',
      showscale: true,
      colorbar: {
        title: 'Energy (eV)',
        x: 0.98,
        len: 0.4,
        y: 0.75
      },
      cmin: minEnergy,
      cmax: maxEnergy
    },
    xaxis: 'x4',
    yaxis: 'y4',
    hovertemplate: 'Band Energy<br>kₓ: %{x:.3f}<br>kᵧ: %{y:.3f}<br>E: %{marker.color:.3f} eV<extra></extra>'
  });

  // Add hexagonal BZ boundaries for each subplot
  const a = 3.19; // Lattice constant
  const bzSize = 4 * Math.PI / (3 * a); // Full BZ size
  
  // Hexagon vertices for BZ boundary
  const hexagonX = [];
  const hexagonY = [];
  for (let i = 0; i <= 6; i++) {
    const angle = (i * Math.PI) / 3;
    hexagonX.push(bzSize * Math.cos(angle));
    hexagonY.push(bzSize * Math.sin(angle));
  }

  // Add BZ boundary to each subplot
  for (let subplotIndex = 1; subplotIndex <= 4; subplotIndex++) {
    plotData.push({
      x: hexagonX,
      y: hexagonY,
      type: 'scatter' as const,
      mode: 'lines' as const,
      name: 'BZ Boundary',
      line: { color: '#374151', width: 2 },
      showlegend: subplotIndex === 1,
      xaxis: `x${subplotIndex}` as any,
      yaxis: `y${subplotIndex}` as any,
      hoverinfo: 'skip'
    });
  }

  // Layout with 4 subplots in 2x2 grid
  const layout = {
    title: {
      text: 'Brillouin Zone Visualizations',
      font: { size: 16, family: 'Inter, sans-serif' }
    },
    showlegend: true,
    legend: {
      orientation: 'h' as const,
      y: -0.1,
      x: 0.5,
      xanchor: 'center' as const
    },
    // Subplot 1: Valley Index (top-left)
    xaxis1: {
      domain: [0, 0.45],
      anchor: 'y1' as const,
      title: { text: 'kₓ (Å⁻¹)', font: { size: 12 } },
      range: [-2, 2],
      showgrid: false,
      zeroline: false,
      tickfont: { size: 10 }
    },
    yaxis1: {
      domain: [0.55, 1],
      anchor: 'x1' as const,
      title: { text: 'kᵧ (Å⁻¹)', font: { size: 12 } },
      range: [-2, 2],
      scaleanchor: 'x1',
      scaleratio: 1,
      showgrid: false,
      zeroline: false,
      tickfont: { size: 10 }
    },
    // Subplot 2: Berry Curvature (top-right)
    xaxis2: {
      domain: [0.55, 1],
      anchor: 'y2' as const,
      title: { text: 'kₓ (Å⁻¹)', font: { size: 12 } },
      range: [-2, 2],
      showgrid: false,
      zeroline: false,
      tickfont: { size: 10 }
    },
    yaxis2: {
      domain: [0.55, 1],
      anchor: 'x2' as const,
      title: { text: 'kᵧ (Å⁻¹)', font: { size: 12 } },
      range: [-2, 2],
      scaleanchor: 'x2',
      scaleratio: 1,
      showgrid: false,
      zeroline: false,
      tickfont: { size: 10 }
    },
    // Subplot 3: Spin Texture (bottom-left)
    xaxis3: {
      domain: [0, 0.45],
      anchor: 'y3' as const,
      title: { text: 'kₓ (Å⁻¹)', font: { size: 12 } },
      range: [-2, 2],
      showgrid: false,
      zeroline: false,
      tickfont: { size: 10 }
    },
    yaxis3: {
      domain: [0, 0.45],
      anchor: 'x3' as const,
      title: { text: 'kᵧ (Å⁻¹)', font: { size: 12 } },
      range: [-2, 2],
      scaleanchor: 'x3',
      scaleratio: 1,
      showgrid: false,
      zeroline: false,
      tickfont: { size: 10 }
    },
    // Subplot 4: Band Energies (bottom-right)
    xaxis4: {
      domain: [0.55, 1],
      anchor: 'y4' as const,
      title: { text: 'kₓ (Å⁻¹)', font: { size: 12 } },
      range: [-2, 2],
      showgrid: false,
      zeroline: false,
      tickfont: { size: 10 }
    },
    yaxis4: {
      domain: [0, 0.45],
      anchor: 'x4' as const,
      title: { text: 'kᵧ (Å⁻¹)', font: { size: 12 } },
      range: [-2, 2],
      scaleanchor: 'x4',
      scaleratio: 1,
      showgrid: false,
      zeroline: false,
      tickfont: { size: 10 }
    },
    
    // Add annotations for subplot titles
    annotations: [
      {
        text: 'Valley Index τ',
        x: 0.225,
        y: 1.02,
        xref: 'paper',
        yref: 'paper',
        showarrow: false,
        font: { size: 14, family: 'Inter, sans-serif' },
        xanchor: 'center'
      },
      {
        text: 'Berry Curvature Ω(k)',
        x: 0.775,
        y: 1.02,
        xref: 'paper',
        yref: 'paper',
        showarrow: false,
        font: { size: 14, family: 'Inter, sans-serif' },
        xanchor: 'center'
      },
      {
        text: 'Spin Texture',
        x: 0.225,
        y: 0.47,
        xref: 'paper',
        yref: 'paper',
        showarrow: false,
        font: { size: 14, family: 'Inter, sans-serif' },
        xanchor: 'center'
      },
      {
        text: 'Band Energies',
        x: 0.775,
        y: 0.47,
        xref: 'paper',
        yref: 'paper',
        showarrow: false,
        font: { size: 14, family: 'Inter, sans-serif' },
        xanchor: 'center'
      }
    ],
    plot_bgcolor: 'rgba(0,0,0,0)',
    paper_bgcolor: 'rgba(0,0,0,0)',
    margin: { l: 60, r: 60, t: 80, b: 100 },
    font: { family: 'Inter, sans-serif', size: 12 }
  };

  return (
    <div className="w-full h-full">
      <Plot
        data={plotData}
        layout={layout}
        config={{
          ...PLOT.config,
          displayModeBar: true,
          modeBarButtonsToRemove: ['pan2d', 'lasso2d']
        }}
        style={{ width: '100%', height: '100%' }}
      />
    </div>
  );
};