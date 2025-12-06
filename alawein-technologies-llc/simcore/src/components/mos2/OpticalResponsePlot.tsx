import React from 'react';
import Plot from 'react-plotly.js';
import { 
  UNIFIED_SCIENTIFIC_PLOT_SYSTEM as PLOT
} from '@/lib/scientific-plot-system';

interface ValleyData {
  kx: number;
  ky: number;
  valleyIndex: number;
  energyConduction: number;
  energyValence: number;
  bandGap: number;
  socSplitting: number;
  opticalMatrixPlus: number;
  opticalMatrixMinus: number;
  circularDichroism: number;
  berryCurvature: number;
  magneticMoment: number;
  spinTextureX: number;
  spinTextureY: number;
}

interface OpticalResponsePlotProps {
  valleyData: ValleyData[];
  energyRange: [number, number];
  showDichroism: boolean;
}

export function OpticalResponsePlot({ 
  valleyData, 
  energyRange,
  showDichroism 
}: OpticalResponsePlotProps) {
  if (!valleyData.length) {
    return (
      <div className="flex items-center justify-center h-96 text-muted-foreground">
        Loading optical response...
      </div>
    );
  }

  // Calculate optical absorption spectrum using valley physics parameters
  const calculateAbsorption = () => {
    // Energy range
    const eMin = 0;
    const eMax = 3.0; // eV
    const nPoints = 400;
    const energies = Array.from({ length: nPoints }, (_, i) => 
      eMin + (i / (nPoints - 1)) * (eMax - eMin)
    );

    // Extract proper physics parameters from valley data for reactivity
    const avgParams = valleyData.length > 0 ? {
      bandGap: valleyData.reduce((sum, p) => sum + Math.abs(p.energyConduction - p.energyValence), 0) / valleyData.length,
      socSplitting: valleyData.reduce((sum, p) => sum + Math.abs(p.socSplitting || 0.15), 0) / valleyData.length || 0.15,
    } : { bandGap: 1.8, socSplitting: 0.15 };

    console.log(`🔬 Optical calc: gap=${avgParams.bandGap.toFixed(2)}eV, SOC=${avgParams.socSplitting.toFixed(3)}eV`);
    const bandGap = avgParams.bandGap;
    const socSplitting = avgParams.socSplitting; 
    const excitonBinding = 0.5; // eV - experimental value for MoS₂
    
    // Exciton energies (literature accurate)
    const A_exciton = bandGap - excitonBinding;                     // ~1.30 eV
    const B_exciton = bandGap - excitonBinding + socSplitting;      // ~1.45 eV
    const bandEdge = bandGap;                                       // Direct gap
    
    // Spectral parameters
    const excitonWidth = 0.04;  // eV (experimental linewidth)
    const continuumWidth = 0.08; // eV (continuum broadening)
    const valleySelectivity = 0.95; // 95% valley selectivity (Mak et al.)

    console.log(`Optical calculation with: gap=${bandGap.toFixed(2)}eV, SOC=${socSplitting.toFixed(3)}eV`);
    console.log(`A-exciton: ${A_exciton.toFixed(2)}eV, B-exciton: ${B_exciton.toFixed(2)}eV`);

    const sigmaPlusSpectrum = [];
    const sigmaMinusSpectrum = [];
    const circularDichroism = [];

    for (let i = 0; i < energies.length; i++) {
      const E = energies[i];
      let sigmaPlus = 0;
      let sigmaMinus = 0;

      // A-exciton contribution (σ+ → K valley)
      const deltaA = E - A_exciton;
      if (Math.abs(deltaA) < 5 * excitonWidth) {
        const lorentzianA = (excitonWidth / Math.PI) / (deltaA**2 + excitonWidth**2);
        sigmaPlus += valleySelectivity * lorentzianA * 8.0;      // Strong K coupling
        sigmaMinus += (1 - valleySelectivity) * lorentzianA * 8.0; // Weak K' coupling
      }

      // B-exciton contribution (σ- → K' valley with SOC)
      const deltaB = E - B_exciton;
      if (Math.abs(deltaB) < 5 * excitonWidth) {
        const lorentzianB = (excitonWidth / Math.PI) / (deltaB**2 + excitonWidth**2);
        sigmaMinus += valleySelectivity * lorentzianB * 6.0;     // Strong K' coupling
        sigmaPlus += (1 - valleySelectivity) * lorentzianB * 6.0;  // Weak K coupling
      }

      // Continuum states above band edge (2D density of states)
      if (E > bandEdge) {
        const excess = E - bandEdge;
        const dos2D = Math.sqrt(excess + 0.01); // 2D DOS ∝ √E
        const continuumResponse = dos2D * Math.exp(-excess / continuumWidth);
        
        // Valley-dependent coupling to continuum
        sigmaPlus += valleySelectivity * continuumResponse * 1.5;
        sigmaMinus += (1 - valleySelectivity) * continuumResponse * 1.5;
      }

      sigmaPlusSpectrum.push(sigmaPlus);
      sigmaMinusSpectrum.push(sigmaMinus);
      
      // Circular dichroism: CD = (σ+ - σ-) / (σ+ + σ-)
      const total = sigmaPlus + sigmaMinus;
      const cd = total > 1e-8 ? (sigmaPlus - sigmaMinus) / total : 0;
      circularDichroism.push(Math.max(-0.98, Math.min(0.98, cd)));
    }

    // Debug output
    const maxSigmaPlus = Math.max(...sigmaPlusSpectrum);
    const maxSigmaMinus = Math.max(...sigmaMinusSpectrum);
    const cdRange = [Math.min(...circularDichroism), Math.max(...circularDichroism)];
    
    console.log(`Peak absorption: σ+=${maxSigmaPlus.toFixed(2)}, σ-=${maxSigmaMinus.toFixed(2)}`);
    console.log(`CD range: [${cdRange[0].toFixed(3)}, ${cdRange[1].toFixed(3)}]`);

    return { 
      energies, 
      sigmaPlusSpectrum, 
      sigmaMinusSpectrum, 
      circularDichroism,
      excitonEnergies: { A_exciton, B_exciton, bandEdge }
    };
  };

  const { energies, sigmaPlusSpectrum, sigmaMinusSpectrum, circularDichroism, excitonEnergies } = calculateAbsorption();

  const plotData = [];

  // σ+ absorption (green for K valley)
  plotData.push({
    x: energies,
    y: sigmaPlusSpectrum,
    type: 'scatter' as const,
    mode: 'lines' as const,
    name: 'σ⁺ → K valley',
    line: { color: '#10b981', width: 3 },
    fill: 'tonexty',
    fillcolor: 'rgba(16, 185, 129, 0.1)',
    hovertemplate: 'σ⁺ (K valley)<br>Energy: %{x:.3f} eV<br>Absorption: %{y:.2f}<extra></extra>'
  });

  // σ- absorption (orange for K' valley)  
  plotData.push({
    x: energies,
    y: sigmaMinusSpectrum,
    type: 'scatter' as const,
    mode: 'lines' as const,
    name: 'σ⁻ → K\' valley',
    line: { color: '#f97316', width: 3 },
    fill: 'tonexty',
    fillcolor: 'rgba(249, 115, 22, 0.1)', 
    hovertemplate: 'σ⁻ (K\' valley)<br>Energy: %{x:.3f} eV<br>Absorption: %{y:.2f}<extra></extra>'
  });

  // Add vertical markers for excitonic transitions (now responsive to parameters)
  const maxAbsorption = Math.max(...sigmaPlusSpectrum, ...sigmaMinusSpectrum);
  
  // A-exciton marker (responsive to band gap)
  plotData.push({
    x: [excitonEnergies.A_exciton, excitonEnergies.A_exciton],
    y: [0, maxAbsorption * 0.8],
    type: 'scatter' as const,
    mode: 'lines' as const,
    name: 'A-exciton',
    line: { color: '#6b7280', width: 2, dash: 'dot' },
    showlegend: true,
    hovertemplate: `A-exciton<br>Energy: ${excitonEnergies.A_exciton.toFixed(2)} eV<extra></extra>`
  });

  // B-exciton marker (responsive to SOC)
  plotData.push({
    x: [excitonEnergies.B_exciton, excitonEnergies.B_exciton],
    y: [0, maxAbsorption * 0.6],
    type: 'scatter' as const,
    mode: 'lines' as const,
    name: 'B-exciton',
    line: { color: '#9ca3af', width: 2, dash: 'dot' },
    showlegend: true,
    hovertemplate: `B-exciton<br>Energy: ${excitonEnergies.B_exciton.toFixed(2)} eV<extra></extra>`
  });

  // Circular dichroism if requested (purple, right axis)
  if (showDichroism) {
    // Convert to percentage
    const dichroismPercent = circularDichroism.map(cd => cd * 100);
    
    plotData.push({
      x: energies,
      y: dichroismPercent,
      type: 'scatter' as const,
      mode: 'lines' as const,
      name: 'Circular Dichroism',
      line: { color: '#8b5cf6', width: 3, dash: 'dash' },
      yaxis: 'y2',
      hovertemplate: 'Circular Dichroism<br>Energy: %{x:.3f} eV<br>η: %{y:.1f}%<extra></extra>'
    });
  }

const layout = PLOT.createLayout(
    'Valley-Selective Optical Response',
    'Photon Energy (eV)',
    'Absorption σ(ω)',
    { energyRange: { min: 0, max: 3.0 } }
  );

  // Enhanced styling to match other plots
  layout.xaxis = {
    ...layout.xaxis,
    title: { text: 'Photon Energy (eV)', font: { size: 14, family: 'Inter, sans-serif' } },
    tickfont: { size: 12 },
    range: [0, 3.0] // Full energy range from 0
  };

  layout.yaxis = {
    ...layout.yaxis,
    title: { text: 'Absorption σ(ω)', font: { size: 14, family: 'Inter, sans-serif' } },
    tickfont: { size: 12 },
    range: [0, Math.max(...sigmaPlusSpectrum, ...sigmaMinusSpectrum) * 1.1] // From 0 to max + 10%
  };

  // Add second y-axis for dichroism with proper styling
  if (showDichroism) {
    layout.yaxis2 = {
      title: {
        text: 'Circular Dichroism (%)',
        font: { color: '#8b5cf6', size: 14, family: 'Inter, sans-serif' }
      },
      titlefont: { color: '#8b5cf6' },
      tickfont: { color: '#8b5cf6', size: 12 },
      overlaying: 'y',
      side: 'right' as const,
      range: [-100, 100],
      showgrid: false
    };
  }

  // Add band edge marker (responsive to band gap)
  const bandgapLine = {
    x: [excitonEnergies.bandEdge, excitonEnergies.bandEdge],
    y: [0, maxAbsorption],
    type: 'scatter' as const,
    mode: 'lines' as const,
    name: 'Band edge',
    line: { color: '#ef4444', width: 2, dash: 'dash' },
    showlegend: true,
    hovertemplate: `Band edge<br>${excitonEnergies.bandEdge.toFixed(2)} eV<extra></extra>`
  };
  plotData.push(bandgapLine);

  // Optimize margins and layout for better fit and prevent overflow
  layout.margin = { l: 70, r: 90, t: 70, b: 70 };
  layout.showlegend = true;
  layout.legend = {
    x: 0.02,
    y: 0.98,
    bgcolor: 'rgba(0,0,0,0.8)',
    bordercolor: '#374151',
    borderwidth: 1,
    font: { size: 12, family: 'Inter, sans-serif' }
  };
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