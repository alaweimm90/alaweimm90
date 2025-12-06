import React, { useEffect } from 'react';
import Plot from 'react-plotly.js';
import { BandStructureResult, DOSResult } from '@/lib/graphene-physics-exact';
import { UNIFIED_SCIENTIFIC_PLOT_SYSTEM as PLOT, PHYSICS_AXIS_LABELS, PHYSICS_ENERGY_RANGES } from '@/lib/scientific-plot-system';
// Utility to check for NaNs or bad values
function hasInvalidValues(arr: number[]): boolean {
  return arr.some(val => !isFinite(val) || isNaN(val));
}

// Utility to check k-path distance monotonicity
function isKPathMonotonic(kPath: number[]): boolean {
  for (let i = 1; i < kPath.length; i++) {
    if (kPath[i] <= kPath[i - 1]) return false;
  }
  return true;
}

type DebugProps = {
  dataNN: BandStructureResult;
  dataNNN: BandStructureResult;
  dosNN: DOSResult;
  dosNNN: DOSResult;
  enableDebug?: boolean;
};

const GrapheneDebugWrapper: React.FC<DebugProps> = ({ 
  dataNN, 
  dataNNN, 
  dosNN, 
  dosNNN, 
  enableDebug = true 
}) => {
  useEffect(() => {
    if (!enableDebug) return;
    
    console.group("🔍 Graphene Debug Panel");

    // Band structure validation
    if (!dataNN.kPath.length || !dataNN.valence.length) console.warn("❌ Empty NN band arrays");
    if (!dataNNN.kPath.length || !dataNNN.valence.length) console.warn("❌ Empty NNN band arrays");
    
    if (hasInvalidValues(dataNN.valence)) console.warn("⚠️ NN valence contains NaN or Infinity");
    if (hasInvalidValues(dataNN.conduction)) console.warn("⚠️ NN conduction contains NaN or Infinity");
    if (hasInvalidValues(dataNNN.valence)) console.warn("⚠️ NNN valence contains NaN or Infinity");
    if (hasInvalidValues(dataNNN.conduction)) console.warn("⚠️ NNN conduction contains NaN or Infinity");
    if (hasInvalidValues(dataNN.kPath)) console.warn("⚠️ NN k_dist contains NaN or Infinity");
    if (hasInvalidValues(dataNNN.kPath)) console.warn("⚠️ NNN k_dist contains NaN or Infinity");
    
    if (!isKPathMonotonic(dataNN.kPath)) console.warn("⚠️ NN k_dist is not monotonic");
    if (!isKPathMonotonic(dataNNN.kPath)) console.warn("⚠️ NNN k_dist is not monotonic");
    
    const kRangeNN = Math.max(...dataNN.kPath) - Math.min(...dataNN.kPath);
    const kRangeNNN = Math.max(...dataNNN.kPath) - Math.min(...dataNNN.kPath);
    if (kRangeNN < 1e-3) console.warn("⚠️ NN k_dist appears collapsed");
    if (kRangeNNN < 1e-3) console.warn("⚠️ NNN k_dist appears collapsed");

    // DOS validation
    if (hasInvalidValues(dosNN.dos)) console.warn("⚠️ NN DOS contains NaN or Infinity");
    if (hasInvalidValues(dosNNN.dos)) console.warn("⚠️ NNN DOS contains NaN or Infinity");
    if (hasInvalidValues(dosNN.energies)) console.warn("⚠️ NN DOS energies contain NaN or Infinity");
    if (hasInvalidValues(dosNNN.energies)) console.warn("⚠️ NNN DOS energies contain NaN or Infinity");

    // Data samples
    console.log("📊 NN Band Structure:");
    console.log("  k-path sample:", dataNN.kPath.slice(0, 5));
    console.log("  k-path range:", Math.min(...dataNN.kPath), "to", Math.max(...dataNN.kPath));
    console.log("  valence sample:", dataNN.valence.slice(0, 5));
    console.log("  conduction sample:", dataNN.conduction.slice(0, 5));
    console.log("  valence range:", Math.min(...dataNN.valence), "to", Math.max(...dataNN.valence));
    console.log("  conduction range:", Math.min(...dataNN.conduction), "to", Math.max(...dataNN.conduction));
    
    console.log("📊 NNN Band Structure:");
    console.log("  k-path sample:", dataNNN.kPath.slice(0, 5));
    console.log("  k-path range:", Math.min(...dataNNN.kPath), "to", Math.max(...dataNNN.kPath));
    console.log("  valence sample:", dataNNN.valence.slice(0, 5));
    console.log("  conduction sample:", dataNNN.conduction.slice(0, 5));
    console.log("  valence range:", Math.min(...dataNNN.valence), "to", Math.max(...dataNNN.valence));
    console.log("  conduction range:", Math.min(...dataNNN.conduction), "to", Math.max(...dataNNN.conduction));

    console.log("📊 DOS Analysis:");
    console.log("  NN DOS energy range:", dosNN.energies[0], "to", dosNN.energies[dosNN.energies.length - 1]);
    console.log("  NN DOS values (first 5):", dosNN.dos.slice(0, 5));
    console.log("  NN DOS max:", Math.max(...dosNN.dos));
    console.log("  NNN DOS energy range:", dosNNN.energies[0], "to", dosNNN.energies[dosNNN.energies.length - 1]);
    console.log("  NNN DOS values (first 5):", dosNNN.dos.slice(0, 5));
    console.log("  NNN DOS max:", Math.max(...dosNNN.dos));

    console.log("📊 High-symmetry points:");
    console.log("  Labels:", dataNN.labels);
    console.log("  Label positions:", dataNN.labelPositions);

    console.groupEnd();
  }, [dataNN, dataNNN, dosNN, dosNNN, enableDebug]);

  // Safety checks for rendering
  const nnBandSafeToRender =
    dataNN.kPath.length &&
    dataNN.valence.length &&
    !hasInvalidValues(dataNN.valence) &&
    !hasInvalidValues(dataNN.kPath) &&
    Math.max(...dataNN.kPath) - Math.min(...dataNN.kPath) > 1e-3;

  const nnnBandSafeToRender =
    dataNNN.kPath.length &&
    dataNNN.valence.length &&
    !hasInvalidValues(dataNNN.valence) &&
    !hasInvalidValues(dataNNN.kPath) &&
    Math.max(...dataNNN.kPath) - Math.min(...dataNNN.kPath) > 1e-3;

  const nnDosSafeToRender =
    dosNN.dos && dosNN.dos.length && dosNN.energies.length &&
    !hasInvalidValues(dosNN.dos) && !hasInvalidValues(dosNN.energies);

  const nnnDosSafeToRender =
    dosNNN.dos && dosNNN.dos.length && dosNNN.energies.length &&
    !hasInvalidValues(dosNNN.dos) && !hasInvalidValues(dosNNN.energies);

  // Fallback band structure data
  const fallbackBandData = [{ 
    x: [0, 1], 
    y: [0, 0], 
    type: 'scatter' as const, 
    name: 'Error: Invalid band data',
    line: { color: 'red' }
  }];

  // Fallback DOS data
  const fallbackDosData = [{ 
    x: [0], 
    y: [0], 
    name: 'Error: No DOS data', 
    type: 'scatter' as const,
    line: { color: 'red' }
  }];

  // Band structure plot data
  const bandPlotData = [];
  
  if (nnBandSafeToRender) {
    bandPlotData.push(PLOT.createBandTrace(
      dataNN.kPath,
      dataNN.valence,
      'valence',
      'NN Valence'
    ) as any);
    bandPlotData.push(PLOT.createBandTrace(
      dataNN.kPath,
      dataNN.conduction,
      'conduction',
      'NN Conduction'
    ) as any);
  }

  if (nnnBandSafeToRender) {
    bandPlotData.push({
      ...(PLOT.createBandTrace(dataNNN.kPath, dataNNN.valence, 'valence', 'NNN Valence') as any),
      line: { ...PLOT.lineStyles.bands.valence, dash: 'dot' }
    });
    bandPlotData.push({
      ...(PLOT.createBandTrace(dataNNN.kPath, dataNNN.conduction, 'conduction', 'NNN Conduction') as any),
      line: { ...PLOT.lineStyles.bands.conduction, dash: 'dot' }
    });
  }

  // DOS plot data
  const dosPlotData = [];
  
  if (nnDosSafeToRender) {
    dosPlotData.push(PLOT.createDOSTrace(
      dosNN.energies,
      dosNN.dos,
      'numerical',
      'NN DOS'
    ) as any);
  }

  if (nnnDosSafeToRender) {
    dosPlotData.push({
      ...(PLOT.createDOSTrace(dosNNN.energies, dosNNN.dos, 'analytical', 'NNN DOS') as any),
      line: { ...PLOT.lineStyles.dos.analytical }
    });
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'row', gap: '20px' }}>
      {/* Band Structure */}
      <div style={{ width: '60%' }}>
        <Plot
          data={bandPlotData.length ? bandPlotData : fallbackBandData}
          layout={{
            ...PLOT.createLayout(
              'Electronic Band Structure (Debug)',
              PHYSICS_AXIS_LABELS.momentum,
              PHYSICS_AXIS_LABELS.energy,
              { energyRange: PHYSICS_ENERGY_RANGES.graphene, showLegend: true, height: 400 }
            ),
            xaxis: {
              ...PLOT.layout.xaxis,
              tickvals: nnBandSafeToRender ? dataNN.labelPositions : undefined,
              ticktext: nnBandSafeToRender ? dataNN.labels : undefined,
              tickmode: 'array' as const
            }
          }}
          config={{ ...PLOT.config, displayModeBar: false }}
        />
      </div>

      {/* DOS */}
      <div style={{ width: '40%' }}>
        <Plot
          data={dosPlotData.length ? dosPlotData : fallbackDosData}
          layout={{
            ...PLOT.createLayout(
              'Density of States (Debug)',
              PHYSICS_AXIS_LABELS.dos,
              PHYSICS_AXIS_LABELS.energy,
              { energyRange: PHYSICS_ENERGY_RANGES.graphene, showLegend: false, height: 400 }
            )
          }}
          config={{ ...PLOT.config, displayModeBar: false }}
        />
      </div>
    </div>
  );
};

export default GrapheneDebugWrapper;