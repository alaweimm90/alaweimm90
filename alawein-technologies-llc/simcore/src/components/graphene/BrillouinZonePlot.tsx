import React from 'react';
import Plot from 'react-plotly.js';
import { BrillouinZoneData } from '@/lib/graphene-physics-fixed';
import { 
  UNIFIED_SCIENTIFIC_PLOT_SYSTEM as PLOT
} from '@/lib/scientific-plot-system';

interface BrillouinZonePlotProps {
  data: BrillouinZoneData;
  showPath?: boolean;
}

export function BrillouinZonePlot({ data, showPath = true }: BrillouinZonePlotProps) {
  const plotData = [
    // Brillouin zone boundary - proper hexagonal shape
    {
      x: data.boundary.map(p => p[0]),
      y: data.boundary.map(p => p[1]),
      type: 'scatter' as const,
      mode: 'lines' as const,
      name: 'Brillouin Zone',
      line: { color: PLOT.colors.physics.brillouinZone, width: 3 },
      fill: 'none',
      hoverinfo: 'skip' as const
    },
    
    // High-symmetry points with corrected coordinates
    {
      x: Object.values(data.highSymmetryPoints).map(p => p[0]),
      y: Object.values(data.highSymmetryPoints).map(p => p[1]),
      type: 'scatter' as const,
      mode: 'markers+text' as const,
      name: 'High-symmetry points',
      marker: { 
        color: PLOT.colors.physics.diracPoint, 
        size: 12,
        symbol: 'circle',
        line: { color: 'hsl(var(--background))', width: 2 }
      },
      text: ['Γ', 'K', "K'", 'M'],
      textposition: 'top center' as const,
      textfont: { 
        size: 16, 
        color: PLOT.colors.physics.diracPoint,
        family: 'Computer Modern, Times, serif'
      },
      hovertemplate: '%{text}<br>k_x: %{x:.3f} Å⁻¹<br>k_y: %{y:.3f} Å⁻¹<extra></extra>'
    }
  ];

  // ✅ FIXED: Add irreducible Brillouin zone (IBZ) triangle with proper visualization
  if (data.ibz && data.ibz.length > 0) {
    plotData.push({
      x: data.ibz.map(p => p[0]),
      y: data.ibz.map(p => p[1]),
      type: 'scatter' as const,
      mode: 'lines' as const,
      name: 'Irreducible BZ (1/12)',
      line: { 
        color: PLOT.colors.physics.brillouinZone, 
        width: 3,  // ✅ THICK lines for visibility
        dash: 'dash' as any
      },
      fill: 'toself',  // ✅ FIXED: Fill the triangle
      fillcolor: `${PLOT.colors.physics.brillouinZone}30`,  // 30% opacity
      hovertemplate: 'IBZ boundary<br>k_x: %{x:.3f} Å⁻¹<br>k_y: %{y:.3f} Å⁻¹<extra></extra>'
    } as any);
  }

  // Add k-path if requested - highlight the integration path
  if (showPath) {
    plotData.push({
      x: data.path.map(p => p[0]),
      y: data.path.map(p => p[1]),
      type: 'scatter' as const,
      mode: 'lines' as const,
      name: 'k-path (Γ→M→K→Γ)',
      line: { color: PLOT.colors.physics.conduction, width: 4 },
      hoverinfo: 'skip' as const
    } as any);
  }

  // Determine proper axis range for hexagonal BZ
  const allX = data.boundary.map(p => p[0]);
  const allY = data.boundary.map(p => p[1]);
  const maxRange = Math.max(
    Math.max(...allX.map(Math.abs)),
    Math.max(...allY.map(Math.abs))
  ) * 1.2;

  const layout = {
    ...PLOT.layout,
    title: {
      text: 'Hexagonal Brillouin Zone',
      font: PLOT.layout.font
    },
    xaxis: {
      ...PLOT.layout.xaxis,
      title: 'k<sub>x</sub> [2π/a]'
    },
    yaxis: {
      ...PLOT.layout.yaxis,
      title: 'k<sub>y</sub> [2π/a]'
    }
  };

  // Special adjustments for BZ plot
  layout.xaxis = {
    ...layout.xaxis,
    scaleanchor: 'y',
    scaleratio: 1,
    range: [-maxRange, maxRange]
  };
  
  layout.yaxis = {
    ...layout.yaxis,
    scaleanchor: 'x',
    scaleratio: 1,
    range: [-maxRange, maxRange]
  };
  
  layout.showlegend = true;

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