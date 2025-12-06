import React from 'react';
import Plot from 'react-plotly.js';
import { 
  UNIFIED_SCIENTIFIC_PLOT_SYSTEM as PLOT,
  PHYSICS_AXIS_LABELS,
  PHYSICS_ENERGY_RANGES
} from '@/lib/scientific-plot-system';

interface ValleyData {
  kx: number;
  ky: number;
  valleyIndex: number;
  energyConduction: number;
  energyValence: number;
  spinUpEnergy: number;
  spinDownEnergy: number;
}

interface ValleyBandStructurePlotProps {
  valleyData: ValleyData[];
  showValence: boolean;
  showConduction: boolean;
  showSpinSplitting: boolean;
  fermiLevel?: number;
}

export function ValleyBandStructurePlot({ 
  valleyData, 
  showValence, 
  showConduction, 
  showSpinSplitting,
  fermiLevel = 0 
}: ValleyBandStructurePlotProps) {
  if (!valleyData.length) {
    return (
      <div className="flex items-center justify-center h-96 text-muted-foreground">
        Loading valley band structure...
      </div>
    );
  }


  // Create k-path for line cuts (Γ-K-M-Γ)
  const createKPath = () => {
    const a = 3.19; // MoS2 lattice constant
    const nPoints = 300;
    const kPath = [];
    const labels = [];
    const labelPositions = [];
    
    // Create path from Γ to K to M to Γ
    const Gamma = [0, 0];
    const K = [4 * Math.PI / (3 * a), 0];
    const M = [2 * Math.PI / (Math.sqrt(3) * a), 2 * Math.PI / (3 * a)];
    
    let currentIndex = 0;
    
    // Γ to K
    const GK_points = Math.floor(nPoints / 3);
    for (let i = 0; i <= GK_points; i++) {
      const frac = i / GK_points;
      const kx = Gamma[0] + frac * (K[0] - Gamma[0]);
      const ky = Gamma[1] + frac * (K[1] - Gamma[1]);
      kPath.push(Math.sqrt(kx**2 + ky**2)); // Distance from Gamma
    }
    labels.push('Γ');
    labelPositions.push(0);
    currentIndex += GK_points;
    
    // K to M  
    const KM_points = Math.floor(nPoints / 3);
    for (let i = 1; i <= KM_points; i++) {
      const frac = i / KM_points;
      const kx = K[0] + frac * (M[0] - K[0]);
      const ky = K[1] + frac * (M[1] - K[1]);
      const dist_from_gamma = Math.sqrt(kx**2 + ky**2);
      kPath.push(kPath[currentIndex] + Math.sqrt((kx - K[0])**2 + (ky - K[1])**2));
    }
    labels.push('K');
    labelPositions.push(currentIndex);
    currentIndex += KM_points;
    
    // M to Γ
    const MG_points = nPoints - currentIndex;
    for (let i = 1; i <= MG_points; i++) {
      const frac = i / MG_points;
      const kx = M[0] + frac * (Gamma[0] - M[0]);
      const ky = M[1] + frac * (Gamma[1] - M[1]);
      const prev_k = kPath[kPath.length - 1];
      const dist_segment = Math.sqrt((kx - M[0])**2 + (ky - M[1])**2) * frac;
      kPath.push(prev_k + dist_segment);
    }
    labels.push('M');
    labelPositions.push(currentIndex);
    labels.push('Γ');
    labelPositions.push(kPath.length - 1);
    
    return { kPath, labels, labelPositions };
  };

  // Extract valley band energies
  const extractBandData = () => {
    const { kPath, labels, labelPositions } = createKPath();
    
    // Calculate band structure along high-symmetry path
    const nk = kPath.length;
    const valenceEnergies = [];
    const conductionEnergies = [];
    const spinUpEnergies = [];
    const spinDownEnergies = [];
    
    // Parameters
    const delta = 1.8; // eV - band gap
    const lambda = 0.15; // eV - SOC
    const t = 1.1; // eV - hopping parameter
    const a = 3.19; // Å - lattice constant
    
    // Create proper k-point coordinates for band calculation
    const Gamma = [0, 0];
    const K = [4 * Math.PI / (3 * a), 0];
    const M = [2 * Math.PI / (Math.sqrt(3) * a), 2 * Math.PI / (3 * a)];
    
    const totalPoints = kPath.length;
    const segmentLength = Math.floor(totalPoints / 3);
    
    for (let i = 0; i < totalPoints; i++) {
      let kx, ky;
      
      if (i <= segmentLength) {
        // Γ to K
        const frac = i / segmentLength;
        kx = Gamma[0] + frac * (K[0] - Gamma[0]);
        ky = Gamma[1] + frac * (K[1] - Gamma[1]);
      } else if (i <= 2 * segmentLength) {
        // K to M
        const frac = (i - segmentLength) / segmentLength;
        kx = K[0] + frac * (M[0] - K[0]);
        ky = K[1] + frac * (M[1] - K[1]);
      } else {
        // M to Γ
        const frac = (i - 2 * segmentLength) / (totalPoints - 2 * segmentLength);
        kx = M[0] + frac * (Gamma[0] - M[0]);
        ky = M[1] + frac * (Gamma[1] - M[1]);
      }
      
      // Distance from valley centers
      const qK = Math.sqrt((kx - K[0])**2 + (ky - K[1])**2);
      const qKp = Math.sqrt((kx + K[0])**2 + (ky - K[1])**2);
      
      // Use nearest valley
      const q = Math.min(qK, qKp);
      const tau = qK < qKp ? 1 : -1;
      
      // MoS2 tight-binding dispersion
      const kinetic = t * q * a;
      
      // Band energies with proper MoS2 dispersion
      const E_c = delta/2 + Math.sqrt(kinetic**2 + (delta/2)**2);
      const E_v = -delta/2 - Math.sqrt(kinetic**2 + (delta/2)**2);
      
      // SOC splitting
      const soc_split = tau * lambda / 2;
      
      valenceEnergies.push(E_v);
      conductionEnergies.push(E_c);
      spinUpEnergies.push(E_c + soc_split);
      spinDownEnergies.push(E_c - soc_split);
    }
    
    return {
      kPath,
      valenceEnergies,
      conductionEnergies,
      spinUpEnergies,
      spinDownEnergies,
      labels,
      labelPositions
    };
  };

  const { kPath, valenceEnergies, conductionEnergies, spinUpEnergies, spinDownEnergies, labels, labelPositions } = extractBandData();

  const plotData = [];

  // Normalize x-axis to [0,1] for proper k-point labeling
  const xValues = kPath.map((_, i) => i / (kPath.length - 1));

  if (showValence) {
    plotData.push({
      x: xValues,
      y: valenceEnergies,
      type: 'scatter' as const,
      mode: 'lines' as const,
      name: 'Valence Band',
      line: { color: '#3B82F6', width: 3 },
      hovertemplate: 'Valence<br>k-path: %{x:.3f}<br>E: %{y:.3f} eV<extra></extra>'
    });
  }

  if (showConduction) {
    plotData.push({
      x: xValues,
      y: conductionEnergies,
      type: 'scatter' as const,
      mode: 'lines' as const,
      name: 'Conduction Band',
      line: { color: '#EF4444', width: 3 },
      hovertemplate: 'Conduction<br>k-path: %{x:.3f}<br>E: %{y:.3f} eV<extra></extra>'
    });
  }

  if (showSpinSplitting) {
    plotData.push({
      x: xValues,
      y: spinUpEnergies,
      type: 'scatter' as const,
      mode: 'lines' as const,
      name: 'Spin ↑',
      line: { color: '#10B981', width: 2, dash: 'dash' },
      hovertemplate: 'Spin ↑<br>k-path: %{x:.3f}<br>E: %{y:.3f} eV<extra></extra>'
    });

    plotData.push({
      x: xValues,
      y: spinDownEnergies,
      type: 'scatter' as const,
      mode: 'lines' as const,
      name: 'Spin ↓',
      line: { color: '#F59E0B', width: 2, dash: 'dash' },
      hovertemplate: 'Spin ↓<br>k-path: %{x:.3f}<br>E: %{y:.3f} eV<extra></extra>'
    });
  }

  // Add Fermi level line
  if (fermiLevel !== undefined && xValues.length > 0) {
    plotData.push({
      x: [0, 1],
      y: [fermiLevel, fermiLevel],
      type: 'scatter' as const,
      mode: 'lines' as const,
      name: 'Fermi Level',
      line: PLOT.lineStyles.bands.fermi,
      showlegend: false,
      hoverinfo: 'skip' as const
    });
  }

  const layout = PLOT.createLayout(
    'MoS₂ Valley Band Structure',
    'k-path',
    PHYSICS_AXIS_LABELS.energy,
    { energyRange: { min: -3, max: 3 } }
  );

  // Style consistency with other plots
  layout.xaxis = {
    ...layout.xaxis,
    title: { text: 'k-path', font: { size: 14, family: 'Inter, sans-serif' } },
    tickfont: { size: 12 }
  };

  layout.yaxis = {
    ...layout.yaxis,
    title: { text: 'Energy (eV)', font: { size: 14, family: 'Inter, sans-serif' } },
    tickfont: { size: 12 }
  };

  // Add k-point labels with proper positioning
  const tickPositions = labelPositions.map(pos => pos / kPath.length);
  layout.xaxis = {
    ...layout.xaxis,
    tickvals: tickPositions,
    ticktext: labels,
    tickmode: 'array' as const,
    range: [0, 1]
  };

  // Optimize margins to match other plots
  layout.margin = { l: 70, r: 40, t: 70, b: 70 };
  layout.plot_bgcolor = 'rgba(0,0,0,0)';
  layout.paper_bgcolor = 'rgba(0,0,0,0)';
  
  // Add legend styling
  layout.showlegend = true;
  layout.legend = {
    x: 0.02,
    y: 0.98,
    bgcolor: 'rgba(0,0,0,0.7)',
    bordercolor: '#374151',
    borderwidth: 1,
    font: { size: 12, family: 'Inter, sans-serif' }
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
}