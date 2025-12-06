import React from 'react';
import Plot from 'react-plotly.js';
import { DOSResult } from '@/lib/graphene-physics-fixed';
import { 
  UNIFIED_SCIENTIFIC_PLOT_SYSTEM as PLOT
} from '@/lib/scientific-plot-system';

interface DOSPlotProps {
  data: DOSResult;
  showLinearDOS?: boolean;
  fermiLevel?: number;
}

export function DOSPlot({ data, showLinearDOS = true, fermiLevel = 0 }: DOSPlotProps) {
  if (!data.energies.length) {
    return (
      <div className="flex items-center justify-center h-96 text-muted-foreground">
        Loading density of states...
      </div>
    );
  }

  const plotData: any[] = [
    // Numerical DOS
    {
      x: data.dos,
      y: data.energies,
      type: 'scatter' as const,
      mode: 'lines' as const,
      name: 'DOS (Numerical)',
      line: { color: PLOT.colors.physics.valence, width: 3 },
      orientation: 'h',
      hovertemplate: 'DOS: %{x:.3f}<br>Energy: %{y:.3f} eV<extra></extra>'
    }
  ];

  // Add linear DOS approximation near Dirac point
  if (showLinearDOS) {
    plotData.push({
      x: data.linearDOS,
      y: data.energies,
      type: 'scatter' as const,
      mode: 'lines' as const,
      name: 'Linear DOS (Dirac)',
      line: { color: PLOT.colors.physics.conduction, width: 2, dash: 'dash' },
      orientation: 'h',
      hovertemplate: 'Linear DOS: %{x:.3f}<br>Energy: %{y:.3f} eV<extra></extra>'
    });
  }

  // Add Fermi level line
  if (fermiLevel !== undefined) {
    const maxDOS = Math.max(...data.dos);
    plotData.push({
      x: [0, maxDOS * 1.1],
      y: [fermiLevel, fermiLevel],
      type: 'scatter' as const,
      mode: 'lines' as const,
      name: 'Fermi Level',
      line: { color: PLOT.colors.muted, width: 2, dash: 'dash' },
      showlegend: false,
      hoverinfo: 'skip' as const
    } as any);
  }

  const layout = {
    ...PLOT.layout,
    title: {
      text: 'Density of States',
      font: PLOT.layout.font
    },
    xaxis: {
      ...PLOT.layout.xaxis,
      title: 'DOS [states/eV]'
    },
    yaxis: {
      ...PLOT.layout.yaxis,
      title: 'Energy [eV]',
      range: [-4, 4] // graphene energy range
    }
  };

  // Special adjustments for DOS plot
  layout.yaxis = {
    ...layout.yaxis,
    showticklabels: false,  // Hide Y-axis labels to share with band plot
    side: 'right'
  };
  layout.margin = { l: 20, r: 80, b: 80, t: 60 }; // Adjusted for DOS alignment

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
}