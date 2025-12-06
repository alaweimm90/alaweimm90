import React from 'react';
import Plot from 'react-plotly.js';
import { 
  UNIFIED_SCIENTIFIC_PLOT_SYSTEM as PLOT,
  PHYSICS_AXIS_LABELS,
  PHYSICS_ENERGY_RANGES
} from '@/lib/scientific-plot-system';

interface EnergyContourPlotProps {
  kx: number[];
  ky: number[];
  energyPlus: number[][];
  energyMinus: number[][];
  showConduction?: boolean;
  showValence?: boolean;
}

export function EnergyContourPlot({ 
  kx, 
  ky, 
  energyPlus, 
  energyMinus,
  showConduction = true,
  showValence = true 
}: EnergyContourPlotProps) {
  if (!kx.length || !ky.length) {
    return (
      <div className="flex items-center justify-center h-96 text-muted-foreground">
        Loading energy contours...
      </div>
    );
  }

  const plotData = [];

  // Define colorscales for contour plots
  const CONTOUR_COLORSCALES = {
    conduction: 'Reds',
    valence: 'Blues'
  };

  // Use standardized energy range
  const energyRange = PHYSICS_ENERGY_RANGES.graphene.max; // 4 eV

  if (showConduction) {
    plotData.push({
      type: 'contour' as const,
      x: kx,
      y: ky,
      z: energyPlus,
      name: 'Conduction Band',
      colorscale: CONTOUR_COLORSCALES.conduction,
      contours: {
        start: 0,
        end: energyRange,
        size: 0.2,
        showlabels: true,
        labelfont: { size: 10, color: '#E2E8F0' }
      },
      colorbar: {
        title: PHYSICS_AXIS_LABELS.energy,
        titlefont: { color: '#E2E8F0', size: 12 },
        tickfont: { color: '#CBD5E1' },
        x: 1.02
      },
      hovertemplate: 'k_x: %{x:.3f} Å⁻¹<br>k_y: %{y:.3f} Å⁻¹<br>E: %{z:.3f} eV<extra></extra>'
    });
  }

  if (showValence) {
    plotData.push({
      type: 'contour' as const,
      x: kx,
      y: ky,
      z: energyMinus,
      name: 'Valence Band',
      colorscale: CONTOUR_COLORSCALES.valence,
      contours: {
        start: -energyRange,
        end: 0,
        size: 0.2,
        showlabels: true,
        labelfont: { size: 10, color: '#E2E8F0' }
      },
      colorbar: {
        title: PHYSICS_AXIS_LABELS.energy,
        titlefont: { color: '#E2E8F0', size: 12 },
        tickfont: { color: '#CBD5E1' },
        x: showConduction ? 1.15 : 1.02
      },
      hovertemplate: 'k_x: %{x:.3f} Å⁻¹<br>k_y: %{y:.3f} Å⁻¹<br>E: %{z:.3f} eV<extra></extra>'
    });
  }

  // Add Dirac point markers
  const a = 2.46; // lattice constant in Angstroms
  const K_x = (2 * Math.PI) / (3 * a);
  const K_y = (2 * Math.PI) / (3 * a * Math.sqrt(3));
  
  const diracPoints = [
    { x: K_x, y: K_y, label: 'K' },
    { x: -K_x, y: -K_y, label: "K'" }
  ];

  plotData.push({
    x: diracPoints.map(p => p.x),
    y: diracPoints.map(p => p.y),
    type: 'scatter' as const,
    mode: 'markers+text' as const,
    name: 'Dirac Points',
    marker: {
      color: PLOT.colors.physics.diracPoint,
      size: 12,
      symbol: 'star',
      line: { color: 'hsl(var(--background))', width: 2 }
    },
    text: diracPoints.map(p => p.label),
    textposition: 'top center' as const,
    textfont: { size: 14, color: PLOT.colors.physics.diracPoint, family: 'Computer Modern, Times, serif' },
    showlegend: false,
    hovertemplate: '%{text} point<br>k_x: %{x:.3f} Å⁻¹<br>k_y: %{y:.3f} Å⁻¹<extra></extra>'
  });

  const layout = PLOT.createLayout(
    showConduction && showValence ? 'Energy Band Contours' : 
    showConduction ? 'Conduction Band Contours' : 'Valence Band Contours',
    PHYSICS_AXIS_LABELS.kx,
    PHYSICS_AXIS_LABELS.ky
  );

  // Special adjustments for contour plot
  layout.xaxis = {
    ...layout.xaxis,
    scaleanchor: 'y',
    scaleratio: 1
  };
  
  layout.yaxis = {
    ...layout.yaxis,
    scaleanchor: 'x',
    scaleratio: 1
  };
  
  layout.height = 400;
  layout.showlegend = false;

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