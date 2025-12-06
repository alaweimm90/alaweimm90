import React, { useMemo } from 'react';
import Plot from 'react-plotly.js';
import { useLLGStore, vec3 } from '@/lib/llg-store';
import { UNIFIED_SCIENTIFIC_PLOT_SYSTEM as PLOT } from '@/lib/scientific-plot-system';

export function LLGTimePlots() {
  const {
    timeData,
    magnetizationHistory,
    energyHistory,
    appliedField,
    show2DPlots
  } = useLLGStore();

  // Calculate proper axis limits
  const maxTime = timeData.length > 0 ? Math.max(...timeData) : 1e-9;
  const minTime = timeData.length > 0 ? Math.min(...timeData) : 0;
  
  // Energy limits based on applied field magnitude
  const fieldMagnitude = appliedField.magnitude;
  const maxEnergyRange = fieldMagnitude * 2; // rough estimate for energy range
  const energyMin = energyHistory.length > 0 ? Math.min(...energyHistory) - maxEnergyRange * 0.1 : -maxEnergyRange;
  const energyMax = energyHistory.length > 0 ? Math.max(...energyHistory) + maxEnergyRange * 0.1 : maxEnergyRange;

  // Time unit for display
  const timeUnit = 'ps';

  const plotData = useMemo(() => {
    if (!show2DPlots || timeData.length === 0) return null;

    // Convert time to appropriate units for readability
    const timeDisplay = timeData.map(t => t * 1e12); // Convert to picoseconds for better scale
    
    const mx = magnetizationHistory.map(m => m[0]);
    const my = magnetizationHistory.map(m => m[1]);
    const mz = magnetizationHistory.map(m => m[2]);
    const mag = magnetizationHistory.map(m => vec3.magnitude(m));

    return {
      magnetization: [
        {
          x: timeDisplay,
          y: mx,
          type: 'scatter' as const,
          mode: 'lines' as const,
          name: 'mx',
          line: { color: '#dc2626', width: 2 }
        },
        {
          x: timeDisplay,
          y: my,
          type: 'scatter' as const,
          mode: 'lines' as const,
          name: 'my',
          line: { color: '#16a34a', width: 2 }
        },
        {
          x: timeDisplay,
          y: mz,
          type: 'scatter' as const,
          mode: 'lines' as const,
          name: 'mz',
          line: { color: '#2563eb', width: 2 }
        }
      ],
      magnitude: [
        {
          x: timeDisplay,
          y: mag,
          type: 'scatter' as const,
          mode: 'lines' as const,
          name: '|m|',
          line: { color: '#7c3aed', width: 3 }
        }
      ],
      energy: [
        {
          x: timeDisplay,
          y: energyHistory,
          type: 'scatter' as const,
          mode: 'lines' as const,
          name: 'Energy',
          line: { color: '#f59e0b', width: 2 }
        }
      ]
    };
  }, [timeData, magnetizationHistory, energyHistory, show2DPlots]);

  const commonLayout = {
    ...PLOT.layout,
    margin: { l: 60, r: 20, t: 40, b: 50 },
    showlegend: true,
    legend: {
      ...PLOT.layout.legend,
      x: 1,
      y: 1,
      xanchor: 'right' as const,
    },
    xaxis: {
      ...PLOT.layout.xaxis,
      title: `Time (${timeUnit})`,
      range: [minTime * 1e12, maxTime * 1e12]
    },
    yaxis: {
      ...PLOT.layout.yaxis
    }
  } as const;
  if (!plotData) {
    return (
      <div className="flex items-center justify-center h-full text-muted-foreground">
        <p>Start simulation to see time evolution plots</p>
      </div>
    );
  }

  return (
    <div className="grid grid-rows-3 gap-4 h-full">
      {/* Magnetization components */}
      <div className="bg-card rounded-lg p-2">
        <Plot
          data={plotData.magnetization}
          layout={{
            ...commonLayout,
            title: { text: 'Magnetization Components', font: { size: 14 } },
            yaxis: { 
              ...commonLayout.yaxis,
              title: 'Magnetization',
              range: [-1.1, 1.1]
            }
          }}
          config={{
            ...PLOT.config,
            displayModeBar: true,
            modeBarButtonsToRemove: ['pan2d', 'lasso2d', 'select2d'],
            displaylogo: false
          }}
          className="w-full h-full"
        />
      </div>

      {/* Magnetization magnitude */}
      <div className="bg-card rounded-lg p-2">
        <Plot
          data={plotData.magnitude}
          layout={{
            ...commonLayout,
            title: { text: 'Magnetization Magnitude |m|', font: { size: 14 } },
            yaxis: { 
              ...commonLayout.yaxis,
              title: '|m|',
              range: [0.99, 1.01]
            }
          }}
          config={{
            displayModeBar: true,
            modeBarButtonsToRemove: ['pan2d', 'lasso2d', 'select2d'],
            displaylogo: false
          }}
          className="w-full h-full"
        />
      </div>

      {/* Energy */}
      <div className="bg-card rounded-lg p-2">
        <Plot
          data={plotData.energy}
          layout={{
            ...commonLayout,
            title: { text: 'Magnetic Energy', font: { size: 14 } },
            yaxis: { 
              ...commonLayout.yaxis,
              title: 'Energy (J)',
              range: [energyMin, energyMax],
              tickformat: '.2e'
            }
          }}
          config={{
            displayModeBar: true,
            modeBarButtonsToRemove: ['pan2d', 'lasso2d', 'select2d'],
            displaylogo: false
          }}
          className="w-full h-full"
        />
      </div>
    </div>
  );
}