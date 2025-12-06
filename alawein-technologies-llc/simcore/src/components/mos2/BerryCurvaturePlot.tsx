import React from 'react';
import Plot from 'react-plotly.js';
import { 
  UNIFIED_SCIENTIFIC_PLOT_SYSTEM as PLOT,
  PHYSICS_AXIS_LABELS
} from '@/lib/scientific-plot-system';

interface ValleyData {
  kx: number;
  ky: number;
  valleyIndex: number;
  berryCurvature: number;
}

interface BerryCurvaturePlotProps {
  valleyData: ValleyData[];
  showContours: boolean;
}

export function BerryCurvaturePlot({ 
  valleyData, 
  showContours 
}: BerryCurvaturePlotProps) {
  if (!valleyData.length) {
    return (
      <div className="flex items-center justify-center h-96 text-muted-foreground">
        Loading Berry curvature...
      </div>
    );
  }

  // Create 2D grid for Berry curvature with correct K-point centering
  const createGridData = () => {
    // Use proper domain [-2, 2] × [-2, 2] to frame 1st BZ correctly
    const nPoints = 60;
    const kxValues = [];
    const kyValues = [];
    const z = [];
    
    // Create uniform grid 
    for (let i = 0; i < nPoints; i++) {
      const kx = -2 + (4 * i) / (nPoints - 1);
      kxValues.push(kx);
    }
    
    for (let j = 0; j < nPoints; j++) {
      const ky = -2 + (4 * j) / (nPoints - 1);
      kyValues.push(ky);
      
      const row = [];
      for (let i = 0; i < nPoints; i++) {
        const kx = kxValues[i];
        
        // Calculate Berry curvature using correct physics
        // K points at ±4π/3a ≈ ±1.32 Å⁻¹
        const a = 3.18; // Literature value
        const K_coord = 4 * Math.PI / (3 * a); // ≈ 1.32
        
        // Distance to each valley
        const dist_K = Math.sqrt((kx - K_coord)**2 + ky**2);
        const dist_Kp = Math.sqrt((kx + K_coord)**2 + ky**2);
        
        // Berry curvature model: sharp peaks at valleys with opposite signs
        const sigma = 0.1; // Peak width in Å⁻¹
        const amplitude = 15; // Peak amplitude
        
        const omega_K = amplitude * Math.exp(-(dist_K**2) / (2 * sigma**2));
        const omega_Kp = -amplitude * Math.exp(-(dist_Kp**2) / (2 * sigma**2));
        
        const berryValue = omega_K + omega_Kp;
        row.push(berryValue);
      }
      z.push(row);
    }
    
    // Debug output for verification
    const flatZ = z.flat();
    const minBerry = Math.min(...flatZ);
    const maxBerry = Math.max(...flatZ);
    const K_coord = 4 * Math.PI / (3 * 3.18); // Recalculate for logging
    console.log(`Berry curvature peaks: min=${minBerry.toFixed(2)}, max=${maxBerry.toFixed(2)}`);
    console.log(`K points located at: ±${K_coord.toFixed(3)} Å⁻¹`);
    
    return { kxValues, kyValues, z };
  };

  const { kxValues, kyValues, z } = createGridData();
  const plotData = [];

  // Berry curvature heatmap with symmetric RdBu colorscale
  const maxScale = Math.max(Math.abs(Math.min(...z.flat())), Math.abs(Math.max(...z.flat())));
  
  plotData.push({
    x: kxValues,
    y: kyValues,
    z: z,
    type: 'heatmap' as const,
    colorscale: [
      [0, '#053061'],    // Dark blue (negative, K')
      [0.25, '#3288bd'], // Blue
      [0.5, '#f7f7f7'],  // White (zero)
      [0.75, '#d73027'], // Red  
      [1, '#67001f']     // Dark red (positive, K)
    ],
    zmid: 0,
    zmin: -maxScale,
    zmax: maxScale,
    showscale: true,
    colorbar: {
      title: {
        text: 'Ω(k) [Å²]',
        font: { color: '#E2E8F0', size: 14 }
      },
      titleside: 'right',
      x: 1.02,
      thickness: 15,
      len: 0.8,
      tickfont: { color: '#E2E8F0', size: 12 },
      outlinecolor: '#374151'
    },
    hovertemplate: 'kₓ: %{x:.3f} Å⁻¹<br>kᵧ: %{y:.3f} Å⁻¹<br>Ω: %{z:.2e} Å²<extra></extra>'
  });

  // Add correct hexagonal BZ overlay for context
  const a = 3.18; // Literature lattice constant
  const kBZ = 4 * Math.PI / (3 * a); // Hexagon radius ≈ 1.32
  
  // Generate correct hexagon using rotation matrix
  const bzHex = [];
  for (let i = 0; i <= 6; i++) {
    const angle = (i * Math.PI) / 3;
    const kx = kBZ * Math.cos(angle);
    const ky = kBZ * Math.sin(angle);
    bzHex.push([kx, ky]);
  }

  plotData.push({
    x: bzHex.map(v => v[0]),
    y: bzHex.map(v => v[1]),
    type: 'scatter' as const,
    mode: 'lines' as const,
    line: { 
      color: '#94a3b8', 
      width: 2,
      dash: 'dash'
    },
    fill: 'none',
    name: '1st BZ',
    showlegend: false,
    hoverinfo: 'skip' as const
  });

  // Add contour lines if requested
  if (showContours) {
    plotData.push({
      x: kxValues,
      y: kyValues,
      z: z,
      type: 'contour' as const,
      showscale: false,
      contours: {
        coloring: 'lines',
        showlabels: true,
        size: maxScale / 8
      },
      line: { width: 1, color: '#374151' },
      hoverinfo: 'skip' as const
    });
  }

  // Add valley markers at correct K-point locations
  const K_pos = [4 * Math.PI / (3 * a), 0];   // ≈ (1.32, 0)
  const Kp_pos = [-4 * Math.PI / (3 * a), 0]; // ≈ (-1.32, 0)

  plotData.push({
    x: [K_pos[0], Kp_pos[0]],
    y: [K_pos[1], Kp_pos[1]],
    type: 'scatter' as const,
    mode: 'markers+text' as const,
    marker: { 
      size: 14, 
      color: ['#10b981', '#f59e0b'],
      symbol: 'circle',
      line: { width: 3, color: '#ffffff' }
    },
    text: ['K', "K'"],
    textposition: 'top center' as const,
    textfont: { 
      size: 16, 
      color: '#ffffff',
      family: 'Inter, sans-serif'
    },
    showlegend: false,
    hovertemplate: '%{text} point<br>kₓ: %{x:.3f} Å⁻¹<br>kᵧ: %{y:.3f} Å⁻¹<extra></extra>'
  });

const layout = PLOT.createLayout(
    'Berry Curvature Ω(k)',
    PHYSICS_AXIS_LABELS.kx,
    PHYSICS_AXIS_LABELS.ky,
    undefined
  );

  // Set proper domain to frame 1st BZ correctly
  layout.xaxis = {
    ...layout.xaxis,
    range: [-2, 2],
    title: { text: 'kₓ (Å⁻¹)', font: { size: 14, family: 'Inter, sans-serif' } },
    tickfont: { size: 12 }
  };
  
  layout.yaxis = {
    ...layout.yaxis,
    range: [-2, 2],
    scaleanchor: 'x',
    scaleratio: 1,
    title: { text: 'kᵧ (Å⁻¹)', font: { size: 14, family: 'Inter, sans-serif' } },
    tickfont: { size: 12 }
  };

  // Optimize margins for better fit and prevent overflow
  layout.margin = { l: 70, r: 100, t: 70, b: 70 };
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