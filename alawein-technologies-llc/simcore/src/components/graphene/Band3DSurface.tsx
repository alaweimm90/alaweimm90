import React from 'react';
import Plot from 'react-plotly.js';
import { UNIFIED_SCIENTIFIC_PLOT_SYSTEM as PLOT } from '@/lib/scientific-plot-system';

interface Band3DSurfaceProps {
  kx: number[];
  ky: number[];
  energyPlus: number[][];
  energyMinus: number[][];
  showConduction?: boolean;
  showValence?: boolean;
}

export function Band3DSurface({ 
  kx, 
  ky, 
  energyPlus, 
  energyMinus,
  showConduction = true,
  showValence = true 
}: Band3DSurfaceProps) {
  if (!kx.length || !ky.length) {
    return (
      <div className="flex items-center justify-center h-96 text-muted-foreground">
        Loading 3D surface...
      </div>
    );
  }

  const plotData = [];

  // Surface color schemes for 3D visualization
  const SURFACE_COLORS = {
    conduction: 'Reds',
    valence: 'Blues'
  };

  if (showConduction) {
    plotData.push({
      type: 'surface' as const,
      x: kx,
      y: ky,
      z: energyPlus,
      name: 'Conduction Band',
      colorscale: SURFACE_COLORS.conduction,
      opacity: 0.8,
      showscale: true,
      colorbar: {
        title: 'Energy (eV)',
        titlefont: { color: 'hsl(var(--foreground))' },
        tickfont: { color: 'hsl(var(--muted-foreground))' },
        x: 1.02
      },
      hovertemplate: 'k_x: %{x:.3f}<br>k_y: %{y:.3f}<br>E: %{z:.3f} eV<extra>Conduction</extra>'
    });
  }

  if (showValence) {
    plotData.push({
      type: 'surface' as const,
      x: kx,
      y: ky,
      z: energyMinus,
      name: 'Valence Band',
      colorscale: SURFACE_COLORS.valence,
      opacity: 0.8,
      showscale: !showConduction,
      colorbar: showConduction ? undefined : {
        title: 'Energy (eV)',
        titlefont: { color: 'hsl(var(--foreground))' },
        tickfont: { color: 'hsl(var(--muted-foreground))' }
      },
      hovertemplate: 'k_x: %{x:.3f}<br>k_y: %{y:.3f}<br>E: %{z:.3f} eV<extra>Valence</extra>'
    });
  }

  const layout = {
    ...PLOT.layout,
    title: {
      text: '3D Electronic Band Structure',
      font: { size: 18, family: "Computer Modern, serif", color: 'hsl(var(--foreground))' }
    },
    scene: {
      xaxis: {
        title: 'k_x (1/Å)',
        titlefont: { color: 'hsl(var(--foreground))' },
        tickfont: { color: 'hsl(var(--muted-foreground))' },
        gridcolor: 'hsl(var(--border))',
        backgroundcolor: 'transparent',
        showgrid: false
      },
      yaxis: {
        title: 'k_y (1/Å)',
        titlefont: { color: '#E2E8F0' },
        tickfont: { color: '#CBD5E1' },
        gridcolor: '#64748B',
        backgroundcolor: 'transparent',
        showgrid: false
      },
      zaxis: {
        title: 'Energy (eV)',
        titlefont: { color: '#E2E8F0' },
        tickfont: { color: '#CBD5E1' },
        gridcolor: '#64748B',
        backgroundcolor: 'transparent',
        showgrid: false
      },
      bgcolor: 'transparent',
      camera: {
        eye: { x: 1.2, y: 1.2, z: 1.2 }
      }
    },
    height: 600,
    margin: { l: 0, r: 0, b: 0, t: 60 }
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
}