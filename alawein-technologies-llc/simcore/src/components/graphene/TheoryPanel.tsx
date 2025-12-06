import React from 'react';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Separator } from '@/components/ui/separator';
import { Check, X } from 'lucide-react';

import { InlineMath, BlockMath } from '@/components/ui/Math';

interface TheoryPanelProps {
  fermiVelocity: number;
  parameters: any;
  validationMode: boolean;
  onValidationToggle: () => void;
}

export function TheoryPanel({ fermiVelocity, parameters, validationMode, onValidationToggle }: TheoryPanelProps) {
  // Validation checks
  const validations = [
    {
      name: "f₁(K) = 0 (Dirac condition)",
      condition: Math.abs(calculateF1AtK(parameters)) < 0.001,
      value: calculateF1AtK(parameters).toFixed(6)
    },
    {
      name: "Fermi velocity ≈ 10⁶ m/s",
      condition: Math.abs(fermiVelocity - 1e6) / 1e6 < 0.5,
      value: `${(fermiVelocity / 1e6).toFixed(2)} × 10⁶ m/s`
    },
    {
      name: "E(K) = E_F at Dirac point",
      condition: Math.abs(3 * parameters.t2) < 0.1,
      value: `${(3 * parameters.t2).toFixed(3)} eV`
    }
  ];

  return (
    <Card className="p-6 bg-card/50 backdrop-blur-sm">
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-xl font-semibold font-serif">Theoretical Framework</h3>
        <Button 
          variant={validationMode ? "default" : "outline"} 
          size="sm"
          onClick={onValidationToggle}
        >
          {validationMode ? "Hide" : "Show"} Validation
        </Button>
      </div>

      <div className="space-y-6">
        {/* Tight-binding Hamiltonian */}
        <div>
          <h4 className="font-medium mb-3">Tight-Binding Hamiltonian</h4>
          <div className="bg-muted/30 p-4 rounded-lg">
            <BlockMath math="H = -t_1 \sum_{\langle i,j \rangle} c_i^\dagger c_j - t_2 \sum_{\langle\langle i,j \rangle\rangle} c_i^\dagger c_j + \varepsilon_0 \sum_i c_i^\dagger c_i" />
          </div>
          <p className="text-sm text-muted-foreground mt-2">
            NN hopping <InlineMath math="t_1" /> and NNN hopping <InlineMath math="t_2" /> with onsite energy <InlineMath math="\varepsilon_0" />
          </p>
        </div>

        {/* Structure Factors */}
        <div>
          <h4 className="font-medium mb-3">Structure Factors</h4>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="bg-muted/30 p-4 rounded-lg">
              <BlockMath math="f_1(\mathbf{k}) = \sum_{\delta} e^{i\mathbf{k} \cdot \boldsymbol{\delta}}" />
              <p className="text-xs text-muted-foreground mt-1">NN structure factor</p>
            </div>
            <div className="bg-muted/30 p-4 rounded-lg">
              <BlockMath math="f_2(\mathbf{k}) = \sum_{\delta'} \cos(\mathbf{k} \cdot \boldsymbol{\delta}')" />
              <p className="text-xs text-muted-foreground mt-1">NNN structure factor</p>
            </div>
          </div>
        </div>

        {/* Energy Eigenvalues */}
        <div>
          <h4 className="font-medium mb-3">Energy Eigenvalues</h4>
          <div className="bg-muted/30 p-4 rounded-lg">
            <BlockMath math="E_\pm(\mathbf{k}) = \varepsilon_0 - t_2 f_2(\mathbf{k}) \pm t_1 |f_1(\mathbf{k})|" />
          </div>
        </div>

        {/* Dirac Point Expansion */}
        <div>
          <h4 className="font-medium mb-3">Dirac Cone Near K-point</h4>
          <div className="bg-muted/30 p-4 rounded-lg">
            <BlockMath math="E_\pm(\mathbf{K} + \mathbf{q}) \approx \varepsilon_0 - t_2 f_2(\mathbf{K}) \pm \hbar v_F |\mathbf{q}|" />
          </div>
          <p className="text-sm text-muted-foreground mt-2">
            Fermi velocity: <InlineMath math="v_F = \frac{3 t_1 a}{2\hbar}" />
          </p>
        </div>

        {/* DOS Expression */}
        <div>
          <h4 className="font-medium mb-3">Density of States</h4>
          <div className="bg-muted/30 p-4 rounded-lg">
            <BlockMath math="D(E) = \frac{2|E|A}{\pi (\hbar v_F)^2} \quad \text{(linear near Dirac point)}" />
          </div>
        </div>

        {/* Validation Section */}
        {validationMode && (
          <>
            <Separator />
            <div>
              <h4 className="font-medium mb-3">Physics Validation</h4>
              <div className="space-y-3">
                {validations.map((validation, index) => (
                  <div key={index} className="flex items-center justify-between p-3 bg-muted/20 rounded-lg">
                    <div className="flex items-center gap-2">
                      {validation.condition ? (
                        <Check className="w-4 h-4 text-green-500" />
                      ) : (
                        <X className="w-4 h-4 text-red-500" />
                      )}
                      <span className="text-sm">{validation.name}</span>
                    </div>
                    <Badge variant={validation.condition ? "default" : "destructive"}>
                      {validation.value}
                    </Badge>
                  </div>
                ))}
              </div>
            </div>
          </>
        )}
      </div>
    </Card>
  );
}

// Helper function to calculate f1 at K point
function calculateF1AtK(parameters: any): number {
  const a = 2.46; // lattice constant
  
  // Correct K point coordinates using reciprocal lattice vectors
  const b1x = 2 * Math.PI / (a * Math.sqrt(3));
  const b1y = 2 * Math.PI / a;
  const b2x = 2 * Math.PI / (a * Math.sqrt(3));
  const b2y = -2 * Math.PI / a;
  
  const Kx = (2/3) * b1x + (1/3) * b2x;
  const Ky = (2/3) * b1y + (1/3) * b2y;
  
  // NN vectors in real space
  const delta1 = [
    [a / 2, a * Math.sqrt(3) / 2],
    [-a / 2, a * Math.sqrt(3) / 2],
    [-a, 0]
  ];
  
  let real = 0;
  let imag = 0;
  
  for (const [dx, dy] of delta1) {
    const phase = Kx * dx + Ky * dy;
    real += Math.cos(phase);
    imag += Math.sin(phase);
  }
  
  return Math.sqrt(real * real + imag * imag);
}