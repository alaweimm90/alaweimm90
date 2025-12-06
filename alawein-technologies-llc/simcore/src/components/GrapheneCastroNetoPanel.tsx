import React, { useMemo, memo } from 'react';
import Plot from 'react-plotly.js';
import { 
  calculateBandStructure, 
  calculateDOS, 
  validateDiracPoint,
  GRAPHENE_CONSTANTS 
} from '@/lib/graphene-tight-binding';
import { UNIFIED_SCIENTIFIC_PLOT_SYSTEM } from '@/lib/scientific-plot-system';

interface GrapheneCastroNetoPanelProps {
  className?: string;
}

export const GrapheneCastroNetoPanel: React.FC<GrapheneCastroNetoPanelProps> = memo(({ className }) => {
  // Calculate band structure and DOS using corrected physics
  const bandData = useMemo(() => calculateBandStructure(100), []);
  const dosData = useMemo(() => calculateDOS([-3, 3], 150, 150), []);
  const validation = useMemo(() => validateDiracPoint(), []);

  // Use unified scientific plotting system
  const plotConfig = UNIFIED_SCIENTIFIC_PLOT_SYSTEM.config;
  
  const commonLayout = {
    ...UNIFIED_SCIENTIFIC_PLOT_SYSTEM.layout,
    font: { 
      family: UNIFIED_SCIENTIFIC_PLOT_SYSTEM.typography.fonts.serif,
      color: UNIFIED_SCIENTIFIC_PLOT_SYSTEM.colors.foreground,
      size: 12 
    },
    margin: { l: 60, r: 20, t: 40, b: 50 },
  };

  return (
    <div className={`grid grid-cols-1 lg:grid-cols-3 gap-6 ${className}`}>
      {/* Band Structure Plot */}
      <div className="lg:col-span-2">
        <div className="bg-card border rounded-lg p-4">
          <h3 className="text-lg font-semibold mb-3">
            Castro Neto Tight-Binding Band Structure
          </h3>
          <Plot
            data={[
              {
                x: bandData.distances,
                y: bandData.energiesPlus,
                mode: 'lines',
                name: 'Conduction',
                line: UNIFIED_SCIENTIFIC_PLOT_SYSTEM.lineStyles.bands.conduction,
                type: 'scatter'
              },
              {
                x: bandData.distances,
                y: bandData.energiesMinus,
                mode: 'lines',
                name: 'Valence',
                line: UNIFIED_SCIENTIFIC_PLOT_SYSTEM.lineStyles.bands.valence,
                type: 'scatter'
              },
              {
                x: [bandData.distances[0], bandData.distances[bandData.distances.length - 1]],
                y: [0, 0],
                mode: 'lines',
                name: 'Fermi Level',
                line: UNIFIED_SCIENTIFIC_PLOT_SYSTEM.lineStyles.bands.fermi,
                showlegend: false,
                type: 'scatter'
              }
            ]}
            layout={{
              ...commonLayout,
              title: '',
              width: undefined,
              height: 400,
              xaxis: {
                title: 'k-path',
                tickvals: bandData.labelPositions,
                ticktext: bandData.labels,
                showgrid: true,
                gridcolor: `${UNIFIED_SCIENTIFIC_PLOT_SYSTEM.colors.muted}30`,
                zeroline: false,
                color: UNIFIED_SCIENTIFIC_PLOT_SYSTEM.colors.foreground
              },
              yaxis: {
                title: 'E / t',
                range: [-3, 3],
                showgrid: true,
                gridcolor: `${UNIFIED_SCIENTIFIC_PLOT_SYSTEM.colors.muted}30`,
                zeroline: true,
                zerolinecolor: `${UNIFIED_SCIENTIFIC_PLOT_SYSTEM.colors.muted}80`,
                color: UNIFIED_SCIENTIFIC_PLOT_SYSTEM.colors.foreground
              },
              legend: {
                x: 0.02,
                y: 0.98,
                bgcolor: UNIFIED_SCIENTIFIC_PLOT_SYSTEM.colors.background.legend,
                font: { color: UNIFIED_SCIENTIFIC_PLOT_SYSTEM.colors.foreground }
              },
              showlegend: true
            }}
            config={plotConfig}
            className="w-full h-[400px]"
            useResizeHandler
          />
        </div>
      </div>

      {/* DOS Plot */}
      <div className="lg:col-span-1">
        <div className="bg-card border rounded-lg p-4">
          <h3 className="text-lg font-semibold mb-3">
            Density of States
          </h3>
          <Plot
            data={[
              {
                x: dosData.dos,
                y: dosData.energies,
                mode: 'lines',
                fill: 'tozerox',
                fillcolor: `${UNIFIED_SCIENTIFIC_PLOT_SYSTEM.colors.physics.dos}30`,
                line: UNIFIED_SCIENTIFIC_PLOT_SYSTEM.lineStyles.dos.numerical,
                name: 'DOS',
                type: 'scatter'
              },
              {
                x: [0, Math.max(...dosData.dos)],
                y: [0, 0],
                mode: 'lines',
                line: UNIFIED_SCIENTIFIC_PLOT_SYSTEM.lineStyles.bands.fermi,
                showlegend: false,
                name: 'Fermi Level',
                type: 'scatter'
              }
            ]}
            layout={{
              ...commonLayout,
              title: '',
              width: undefined,
              height: 400,
              xaxis: {
                title: 'DOS (arb. units)',
                showgrid: true,
                gridcolor: `${UNIFIED_SCIENTIFIC_PLOT_SYSTEM.colors.muted}30`,
                color: UNIFIED_SCIENTIFIC_PLOT_SYSTEM.colors.foreground
              },
              yaxis: {
                title: '',
                range: [-3, 3],
                showgrid: true,
                gridcolor: `${UNIFIED_SCIENTIFIC_PLOT_SYSTEM.colors.muted}30`,
                zeroline: true,
                zerolinecolor: `${UNIFIED_SCIENTIFIC_PLOT_SYSTEM.colors.muted}80`,
                color: UNIFIED_SCIENTIFIC_PLOT_SYSTEM.colors.foreground,
                showticklabels: false
              },
              showlegend: false
            }}
            config={plotConfig}
            className="w-full h-[400px]"
            useResizeHandler
          />
        </div>
      </div>

      {/* Validation Panel */}
      <div className="lg:col-span-3">
        <div className="bg-card border rounded-lg p-4">
          <h3 className="text-lg font-semibold mb-3">Physical Validation</h3>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div className="text-center">
              <div className="text-2xl font-bold text-primary">
                {validation.isValid ? '✓' : '✗'}
              </div>
              <div className="text-sm text-muted-foreground">Dirac Point</div>
              <div className="text-xs">
                |f₁(K)| = {validation.f1Magnitude.toExponential(2)}
              </div>
            </div>
            
            <div className="text-center">
              <div className="text-2xl font-bold">
                {(validation.gapAtK * 1000).toFixed(1)}
              </div>
              <div className="text-sm text-muted-foreground">Gap at K (meV)</div>
              <div className="text-xs">Should be ≈ 0</div>
            </div>
            
            <div className="text-center">
              <div className="text-2xl font-bold">
                {GRAPHENE_CONSTANTS.t.toFixed(1)}
              </div>
              <div className="text-sm text-muted-foreground">t (eV)</div>
              <div className="text-xs">NN hopping</div>
            </div>
            
            <div className="text-center">
              <div className="text-2xl font-bold">
                {GRAPHENE_CONSTANTS.tPrime.toFixed(2)}
              </div>
              <div className="text-sm text-muted-foreground">t' (eV)</div>
              <div className="text-xs">NNN hopping</div>
            </div>
          </div>
          
          <div className="mt-4 p-3 bg-muted rounded text-sm">
            <strong>Castro Neto Physics:</strong> Correct A→B bond vectors, proper K-point at (4π/3√3a, 0), 
            +3t' energy shift for Dirac point centering, and Gaussian-broadened DOS.
          </div>
        </div>
      </div>
    </div>
  );
});

// Display name for debugging
GrapheneCastroNetoPanel.displayName = 'GrapheneCastroNetoPanel';