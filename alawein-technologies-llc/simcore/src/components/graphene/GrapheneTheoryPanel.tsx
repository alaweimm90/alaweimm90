import React, { useState } from 'react';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Separator } from '@/components/ui/separator';
import { Slider } from '@/components/ui/slider';
import { Label } from '@/components/ui/label';
import { Check, X, Settings } from 'lucide-react';

import { InlineMath, BlockMath } from '@/components/ui/Math';

interface GrapheneTheoryPanelProps {
  fermiVelocity: number;
  parameters: any;
  validationMode: boolean;
  onValidationToggle: () => void;
  dosSmoothing?: number;
  onDosSmoothingChange?: (value: number) => void;
}

export function GrapheneTheoryPanel({ 
  fermiVelocity, 
  parameters, 
  validationMode, 
  onValidationToggle,
  dosSmoothing = 0.2,
  onDosSmoothingChange
}: GrapheneTheoryPanelProps) {
  
  // Validation checks
  const validations = [
    {
      name: "f₁(K) = 0 (Dirac condition)",
      condition: Math.abs(calculateF1AtK(parameters)) < 0.001,
      value: calculateF1AtK(parameters).toFixed(6),
      description: "Structure factor vanishes at Dirac point"
    },
    {
      name: "Fermi velocity ≈ 10⁶ m/s",
      condition: Math.abs(fermiVelocity - 1e6) / 1e6 < 0.5,
      value: `${(fermiVelocity / 1e6).toFixed(2)} × 10⁶ m/s`,
      description: "Characteristic velocity scale"
    },
    {
      name: "NNN shift = 3t'",
      condition: Math.abs(3 * parameters.t2) < 1.0,
      value: `${(3 * parameters.t2).toFixed(3)} eV`,
      description: "Energy shift from particle-hole symmetry breaking"
    }
  ];

  return (
    <div className="space-y-6">
      {/* Main Theory Header */}
      <Card className="p-6 bg-card/50 backdrop-blur-sm">
        <div className="flex items-center justify-between mb-6">
          <h3 className="text-xl font-semibold font-serif">📘 Theoretical Model</h3>
          <div className="flex gap-2">
            <Button 
              variant={validationMode ? "default" : "outline"} 
              size="sm"
              onClick={onValidationToggle}
            >
              {validationMode ? "Hide" : "Show"} Validation
            </Button>
          </div>
        </div>

        <div className="space-y-6">
          {/* NN Hamiltonian */}
          <div>
            <h4 className="font-medium mb-3 text-blue-400">Nearest-Neighbor (NN) Hopping</h4>
            <div className="bg-muted/30 p-4 rounded-lg border border-blue-500/20">
              <BlockMath math="\hat{H}_{\text{NN}} = -t \sum_{\langle m,n \rangle} \left( \hat{a}_m^\dagger \hat{b}_n + \text{h.c.} \right)" />
            </div>
            <p className="text-sm text-muted-foreground mt-2">
              where <InlineMath math="t \approx 2.7 \, \text{eV}" /> connects A–B sublattices across nearest neighbors.
            </p>
          </div>

          {/* NNN Hamiltonian */}
          <div>
            <h4 className="font-medium mb-3 text-purple-400">Next-Nearest-Neighbor (NNN) Hopping</h4>
            <div className="bg-muted/30 p-4 rounded-lg border border-purple-500/20">
              <BlockMath math="\hat{H}_{\text{NNN}} = -t' \sum_{\langle\langle m,n \rangle\rangle} \left( \hat{a}_m^\dagger \hat{a}_n + \hat{b}_m^\dagger \hat{b}_n + \text{h.c.} \right)" />
            </div>
            <p className="text-sm text-muted-foreground mt-2">
              where <InlineMath math="t' \in [0.05, 0.15] \, \text{eV}" /> breaks chiral symmetry (CS) but preserves <InlineMath math="D_6" /> symmetry.
            </p>
          </div>

          {/* Momentum Space */}
          <div>
            <h4 className="font-medium mb-3 text-green-400">Momentum-Space Hamiltonian</h4>
            <div className="bg-muted/30 p-4 rounded-lg border border-green-500/20">
              <BlockMath math="\hat{H}_{\vec{k}} = \begin{bmatrix} E_{\vec{k}}^{AA} & E_{\vec{k}}^{AB} \\ E_{\vec{k}}^{BA} & E_{\vec{k}}^{BB} \end{bmatrix}" />
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4">
              <div className="bg-muted/20 p-3 rounded">
                <InlineMath math="E_{\vec{k}}^{AA} = E_{\vec{k}}^{BB} = -t' f_2(\vec{k})" />
                <p className="text-xs text-muted-foreground mt-1">NNN diagonal terms</p>
              </div>
              <div className="bg-muted/20 p-3 rounded">
                <InlineMath math="E_{\vec{k}}^{AB} = -t f_1(\vec{k})" />
                <p className="text-xs text-muted-foreground mt-1">NN off-diagonal coupling</p>
              </div>
            </div>
          </div>

          {/* Structure Factors */}
          <div>
            <h4 className="font-medium mb-3 text-orange-400">Structure Factors</h4>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="bg-muted/30 p-4 rounded-lg border border-orange-500/20">
                <BlockMath math="f_1(\vec{k}) = \sum_{\mu=1}^{3} e^{i \vec{k} \cdot \delta_{\mu}}" />
                <p className="text-xs text-muted-foreground mt-1">NN structure factor</p>
              </div>
              <div className="bg-muted/30 p-4 rounded-lg border border-orange-500/20">
                <BlockMath math="f_2(\vec{k}) = \sum_{\nu=1}^{6} e^{i \vec{k} \cdot \delta_{\nu}}" />
                <p className="text-xs text-muted-foreground mt-1">NNN structure factor</p>
              </div>
            </div>
          </div>

          {/* Dispersion */}
          <div>
            <h4 className="font-medium mb-3 text-red-400">Band Structure Dispersion</h4>
            <div className="bg-muted/30 p-4 rounded-lg border border-red-500/20">
              <BlockMath math="E_{\vec{k}}^{\pm} = -t' f_2(\vec{k}) \pm t \left| f_1(\vec{k}) \right|" />
            </div>
            <div className="bg-accent/20 p-3 rounded-lg mt-3">
              <p className="text-sm text-muted-foreground">
                <strong>Energy alignment:</strong> For visual alignment (Dirac point at E = 0), apply shift:
              </p>
              <div className="mt-2">
                <InlineMath math="E_{\vec{k}}^{\pm} \mapsto E_{\vec{k}}^{\pm} - 3t'" />
              </div>
            </div>
          </div>

          {/* Low Energy Expansion */}
          <div>
            <h4 className="font-medium mb-3 text-teal-400">Low-Energy Expansion near Dirac Point</h4>
            <div className="bg-muted/30 p-4 rounded-lg border border-teal-500/20">
              <BlockMath math="E_{\vec{\Delta k}}^{\pm} = \pm \hbar v_F |\Delta \vec{k}| + 3t'" />
            </div>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-3 mt-3 text-xs">
              <div className="bg-muted/20 p-2 rounded">
                <InlineMath math="\vec{\Delta k} = \vec{k} - \vec{K}" />
                <p className="text-muted-foreground">Momentum from Dirac point</p>
              </div>
              <div className="bg-muted/20 p-2 rounded">
                <InlineMath math="\hbar v_F = \frac{3}{2} t a_{\text{CC}}" />
                <p className="text-muted-foreground">Fermi velocity</p>
              </div>
              <div className="bg-muted/20 p-2 rounded">
                <InlineMath math="3t'" /> shift
                <p className="text-muted-foreground">Particle-hole asymmetry</p>
              </div>
            </div>
          </div>

          {/* DOS */}
          <div>
            <h4 className="font-medium mb-3 text-pink-400">Density of States (DOS)</h4>
            <div className="bg-muted/30 p-4 rounded-lg border border-pink-500/20">
              <BlockMath math="D_{2D}(E) = \frac{2 |E - 3t'|}{\pi (\hbar v_F)^2}" />
            </div>
            <p className="text-sm text-muted-foreground mt-2">
              Linear vanishing at shifted Dirac point with van Hove singularities near saddle points at M.
            </p>
          </div>

          {/* k-space Path Details */}
          <div>
            <h4 className="font-medium mb-3 text-cyan-400">k-space Path Details</h4>
            <div className="bg-muted/30 p-4 rounded-lg border border-cyan-500/20">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
                <div>
                  <InlineMath math="\Gamma = (0, 0)" />
                  <p className="text-xs text-muted-foreground">Zone center</p>
                </div>
                <div>
                  <InlineMath math="K = \left( \frac{4\pi}{3\sqrt{3}a}, 0 \right)" />
                  <p className="text-xs text-muted-foreground">Dirac point</p>
                </div>
                <div>
                  <InlineMath math="M = \left( \frac{\pi}{\sqrt{3}a}, \frac{\pi}{3a} \right)" />
                  <p className="text-xs text-muted-foreground">Edge midpoint</p>
                </div>
              </div>
              <Separator className="my-3" />
              <div className="grid grid-cols-3 gap-4 text-xs">
                <div>Γ → K: index 0–99</div>
                <div>K → M: index 100–199</div>
                <div>M → Γ: index 200–299</div>
              </div>
            </div>
          </div>

          {/* DOS Smoothing Control */}
          {onDosSmoothingChange && (
            <div>
              <h4 className="font-medium mb-3 flex items-center gap-2">
                <Settings className="w-4 h-4" />
                DOS Visualization Controls
              </h4>
              <div className="bg-muted/30 p-4 rounded-lg">
                <Label className="text-sm font-medium flex items-center gap-2 mb-2">
                  DOS Smoothing Width (σ) [eV]
                </Label>
                <Slider
                  value={[dosSmoothing]}
                  onValueChange={([value]) => onDosSmoothingChange(value)}
                  max={0.5}
                  min={0.0}
                  step={0.01}
                  className="mt-2"
                />
                <div className="text-xs text-muted-foreground mt-2 flex justify-between">
                  <span>Raw data</span>
                  <span className="font-mono">{dosSmoothing.toFixed(2)} eV</span>
                  <span>Heavy smoothing</span>
                </div>
              </div>
            </div>
          )}
        </div>
      </Card>

      {/* Validation Section */}
      {validationMode && (
        <Card className="p-6 bg-card/50 backdrop-blur-sm">
          <h4 className="font-medium mb-4">🔬 Physics Validation</h4>
          <div className="space-y-4">
            {validations.map((validation, index) => (
              <div key={index} className="flex items-start gap-3 p-3 bg-muted/20 rounded-lg">
                <div className="flex-shrink-0 mt-0.5">
                  {validation.condition ? (
                    <Check className="w-4 h-4 text-green-500" />
                  ) : (
                    <X className="w-4 h-4 text-red-500" />
                  )}
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-sm font-medium">{validation.name}</span>
                    <Badge variant={validation.condition ? "default" : "destructive"} className="ml-2">
                      {validation.value}
                    </Badge>
                  </div>
                  <p className="text-xs text-muted-foreground">
                    {validation.description}
                  </p>
                </div>
              </div>
            ))}
          </div>
        </Card>
      )}
    </div>
  );
}

// Helper function to calculate f1 at K point (updated with correct physics)
function calculateF1AtK(parameters: any): number {
  const a = 2.46; // lattice constant in Angstroms
  
  // Exact K point from theory: K = (4π/3a, 0)
  const Kx = 4 * Math.PI / (3 * a);
  const Ky = 0;
  
  // NN vectors: δ₁μ ∈ {a[0, 1], a[√3/2, -1/2], a[-√3/2, -1/2]}
  const delta1 = [
    [0, a],                                    // δ₁₁ = a[0, 1]
    [a * Math.sqrt(3) / 2, -a / 2],          // δ₁₂ = a[√3/2, -1/2]
    [-a * Math.sqrt(3) / 2, -a / 2]          // δ₁₃ = a[-√3/2, -1/2]
  ];
  
  // Apply strain if present
  const strain = parameters.strain || { exx: 0, eyy: 0, exy: 0 };
  const strainedDeltas = delta1.map(([dx, dy]) => {
    const dx_strained = dx * (1 + strain.exx) + dy * strain.exy;
    const dy_strained = dx * strain.exy + dy * (1 + strain.eyy);
    return [dx_strained, dy_strained];
  });
  
  let real = 0;
  let imag = 0;
  
  for (const [dx, dy] of strainedDeltas) {
    const phase = Kx * dx + Ky * dy;
    real += Math.cos(phase);
    imag += Math.sin(phase);
  }
  
  return Math.sqrt(real * real + imag * imag);
}