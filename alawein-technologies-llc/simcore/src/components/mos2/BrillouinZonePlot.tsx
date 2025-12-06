import React from 'react';
import Plot from 'react-plotly.js';
import { 
  UNIFIED_SCIENTIFIC_PLOT_SYSTEM as PLOT,
  PHYSICS_AXIS_LABELS
} from '@/lib/scientific-plot-system';

interface BrillouinZonePlotProps {
  showValleyPoints: boolean;
  showPath?: boolean;
  latticeConstant?: number;
  params?: {
    delta: number;
    lambda: number;
    velocity: number;
    electricField: number;
    magneticField: number;
    interlayerCoupling: number;
    stackingOrder: string;
  };
}

export function BrillouinZonePlot({ 
  showValleyPoints,
  showPath = false,
  latticeConstant = 3.19,
  params
}: BrillouinZonePlotProps) {
  // MoS2 lattice parameters (now responsive to parameters)
  const a = latticeConstant; // Use provided lattice constant
  
  // Debug log parameter changes
  if (params) {
    console.log(`BZ plot parameters - Gap: ${params.delta.toFixed(2)} eV, SOC: ${params.lambda.toFixed(3)} eV, Velocity: ${params.velocity.toFixed(2)} eV·Å`);
  }
  
  // Correct reciprocal lattice coordinates
  const createBZBoundary = () => {
    // Hexagon radius in reciprocal space
    const kBZ = 4 * Math.PI / (3 * a); // ≈ 1.32 Å⁻¹
    
    // Generate hexagon vertices using rotation matrix
    const vertices = [];
    for (let i = 0; i <= 6; i++) {
      const angle = (i * Math.PI) / 3; // 60° increments
      const kx = kBZ * Math.cos(angle);
      const ky = kBZ * Math.sin(angle);
      vertices.push([kx, ky]);
    }
    
    console.log(`BZ hexagon radius: ${kBZ.toFixed(3)} Å⁻¹`);
    return vertices;
  };

  const bzVertices = createBZBoundary();
  const plotData = [];

  // First Brillouin zone boundary (correct geometry)
  plotData.push({
    x: bzVertices.map(v => v[0]),
    y: bzVertices.map(v => v[1]),
    type: 'scatter' as const,
    mode: 'lines' as const,
    name: '1st BZ',
    line: { 
      color: '#64748b', 
      width: 3,
      dash: 'solid'
    },
    fill: 'none',
    showlegend: true,
    hoverinfo: 'skip' as const
  });

  // Correct high-symmetry points (literature coordinates)
  const Gamma = [0, 0];
  const K = [4 * Math.PI / (3 * a), 0];           // ≈ (1.32, 0)
  const Kp = [-4 * Math.PI / (3 * a), 0];         // ≈ (-1.32, 0)  
  const M = [Math.PI / a, Math.PI / (Math.sqrt(3) * a)]; // ≈ (0.99, 0.57)

  console.log(`High-symmetry points:`);
  console.log(`K: (${K[0].toFixed(3)}, ${K[1].toFixed(3)}) Å⁻¹`);
  console.log(`K': (${Kp[0].toFixed(3)}, ${Kp[1].toFixed(3)}) Å⁻¹`);
  console.log(`M: (${M[0].toFixed(3)}, ${M[1].toFixed(3)}) Å⁻¹`);

  // Valley points K and K' with correct coordinates
  if (showValleyPoints) {
    plotData.push({
      x: [K[0], Kp[0]],
      y: [K[1], Kp[1]],
      type: 'scatter' as const,
      mode: 'markers+text' as const,
      marker: {
        size: 15,
        color: ['#10b981', '#f59e0b'],
        symbol: 'circle',
        line: { width: 3, color: '#ffffff' }
      },
      text: ['K', "K'"],
      textposition: 'top center' as const,
      textfont: { 
        color: '#ffffff', 
        size: 16, 
        family: 'Inter, sans-serif'
      },
      name: 'Valley Points',
      showlegend: true,
      hovertemplate: '%{text} valley<br>kₓ: %{x:.3f} Å⁻¹<br>kᵧ: %{y:.3f} Å⁻¹<extra></extra>'
    });

    // Other high-symmetry points
    plotData.push({
      x: [Gamma[0], M[0]],
      y: [Gamma[1], M[1]],
      type: 'scatter' as const,
      mode: 'markers+text' as const,
      marker: { 
        size: 12, 
        color: '#6366f1',
        symbol: 'circle',
        line: { width: 2, color: '#ffffff' }
      },
      text: ['Γ', 'M'],
      textposition: 'bottom center' as const,
      textfont: { 
        color: '#ffffff', 
        size: 14, 
        family: 'Inter, sans-serif'
      },
      name: 'High-sym points',
      showlegend: false,
      hovertemplate: '%{text} point<br>kₓ: %{x:.3f} Å⁻¹<br>kᵧ: %{y:.3f} Å⁻¹<extra></extra>'
    });
  }

  // Standard DFT k-path: Γ→K→M→Γ (physically correct)
  if (showPath) {
    const pathPoints = [Gamma, K, M, Gamma];
    
    plotData.push({
      x: pathPoints.map(p => p[0]),
      y: pathPoints.map(p => p[1]),
      type: 'scatter' as const,
      mode: 'lines' as const,
      name: 'k-path (Γ→K→M→Γ)',
      line: { 
        color: '#ffffff', 
        width: 3,
        dash: 'dash'
      },
      showlegend: true,
      hoverinfo: 'skip' as const
    });
  }

const layout = PLOT.createLayout(
    'MoS₂ Brillouin Zone',
    PHYSICS_AXIS_LABELS.kx,
    PHYSICS_AXIS_LABELS.ky
  );

  // Set proper domain centered on Γ with correct scaling  
  const kRange = 1.8; // Slightly larger than BZ radius for context
  layout.xaxis = {
    ...layout.xaxis,
    range: [-kRange, kRange],
    title: { text: 'kₓ (Å⁻¹)', font: { size: 14, family: 'Inter, sans-serif' } },
    tickfont: { size: 12 }
  };
  
  layout.yaxis = {
    ...layout.yaxis,
    range: [-kRange, kRange],
    scaleanchor: 'x',
    scaleratio: 1,
    title: { text: 'kᵧ (Å⁻¹)', font: { size: 14, family: 'Inter, sans-serif' } },
    tickfont: { size: 12 }
  };

  // Optimize margins to match other plots
  layout.margin = { l: 70, r: 70, t: 70, b: 70 };
  layout.plot_bgcolor = 'rgba(0,0,0,0)';
  layout.paper_bgcolor = 'rgba(0,0,0,0)';

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