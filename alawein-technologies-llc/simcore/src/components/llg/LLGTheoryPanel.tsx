import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Badge } from '@/components/ui/badge';
import { BookOpen, Atom, Zap, Cpu } from 'lucide-react';
import { InlineMath, BlockMath } from '@/components/ui/Math';

export function LLGTheoryPanel() {
  return (
    <div className="space-y-6">
      <Tabs defaultValue="llg-equation" className="w-full">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="llg-equation">LLG Equation</TabsTrigger>
          <TabsTrigger value="spintronics">Spintronics</TabsTrigger>
          <TabsTrigger value="resonance">Magnetic Resonance</TabsTrigger>
          <TabsTrigger value="numerical">Numerical Methods</TabsTrigger>
        </TabsList>

        <TabsContent value="llg-equation" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <BookOpen className="h-5 w-5" />
                Landau-Lifshitz-Gilbert Equation
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              <div>
                <h3 className="text-lg font-semibold mb-3">Fundamental Equation</h3>
                <p className="text-sm text-muted-foreground mb-4">
                  The LLG equation describes the time evolution of magnetization in magnetic materials,
                  combining precession around the effective field with Gilbert damping.
                </p>
                
                <div className="bg-muted p-6 rounded-lg">
                  <BlockMath math="\frac{d\mathbf{m}}{dt} = -\gamma(\mathbf{m} \times \mathbf{H}_{\text{eff}}) - \alpha\gamma(\mathbf{m} \times (\mathbf{m} \times \mathbf{H}_{\text{eff}}))" />
                </div>
                
                <div className="mt-4 space-y-2 text-sm">
                  <div><strong>m:</strong> Normalized magnetization vector (|m| = 1)</div>
                  <div><strong>γ:</strong> Gyromagnetic ratio ≈ 2.21×10⁵ rad·s⁻¹·T⁻¹</div>
                  <div><strong>α:</strong> Gilbert damping parameter (0 &lt; α &lt; 1)</div>
                  <div><strong>H_eff:</strong> Effective magnetic field</div>
                </div>
              </div>

              <div>
                <h3 className="text-lg font-semibold mb-3">Physical Interpretation</h3>
                <div className="grid md:grid-cols-2 gap-4">
                  <div className="bg-muted p-4 rounded-lg">
                    <h4 className="font-medium mb-2 text-red-600">Precession Term</h4>
                    <BlockMath math="-\gamma(\mathbf{m} \times \mathbf{H}_{\text{eff}})" />
                    <p className="text-sm mt-2">
                      Conservative motion causing magnetization to precess around the effective field
                      at the Larmor frequency ω = γH.
                    </p>
                  </div>
                  
                  <div className="bg-muted p-4 rounded-lg">
                    <h4 className="font-medium mb-2 text-green-600">Damping Term</h4>
                    <BlockMath math="-\alpha\gamma(\mathbf{m} \times (\mathbf{m} \times \mathbf{H}_{\text{eff}}))" />
                    <p className="text-sm mt-2">
                      Dissipative term that drives the magnetization toward equilibrium
                      alignment with the effective field.
                    </p>
                  </div>
                </div>
              </div>

              <div>
                <h3 className="text-lg font-semibold mb-3">Effective Field Components</h3>
                <div className="bg-muted p-4 rounded-lg">
                  <BlockMath math="\mathbf{H}_{\text{eff}} = \mathbf{H}_{\text{app}} + \mathbf{H}_{\text{anis}} + \mathbf{H}_{\text{demag}} + \mathbf{H}_{\text{exch}}" />
                  
                  <div className="mt-4 space-y-3 text-sm">
                    <div>
                      <strong>Applied Field:</strong> External magnetic field
                      <BlockMath math="\mathbf{H}_{\text{app}} = \mathbf{H}_{\text{external}}" />
                    </div>
                    
                    <div>
                      <strong>Anisotropy Field:</strong> Crystal or shape anisotropy
                      <BlockMath math="\mathbf{H}_{\text{anis}} = \frac{2K}{\mu_0 M_s}(\mathbf{m} \cdot \hat{\mathbf{u}})\hat{\mathbf{u}}" />
                    </div>
                    
                    <div>
                      <strong>Demagnetization Field:</strong> Dipolar interactions
                      <BlockMath math="\mathbf{H}_{\text{demag}} = -N \mathbf{M}" />
                    </div>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="spintronics" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Cpu className="h-5 w-5" />
                Spintronic Applications
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              <div>
                <h3 className="text-lg font-semibold mb-3">Magnetic Memory Devices</h3>
                <p className="text-sm text-muted-foreground mb-4">
                  LLG dynamics govern the switching behavior in magnetic memory technologies.
                </p>
                
                <div className="grid md:grid-cols-2 gap-4">
                  <div className="bg-muted p-4 rounded-lg">
                    <h4 className="font-medium mb-2">MRAM (Magnetic RAM)</h4>
                    <ul className="text-sm space-y-1">
                      <li>• Fast switching (&lt;10 ns)</li>
                      <li>• Non-volatile storage</li>
                      <li>• Low power consumption</li>
                      <li>• Spin-transfer torque switching</li>
                    </ul>
                  </div>
                  
                  <div className="bg-muted p-4 rounded-lg">
                    <h4 className="font-medium mb-2">STT-RAM</h4>
                    <ul className="text-sm space-y-1">
                      <li>• Spin-transfer torque driven</li>
                      <li>• Current-induced switching</li>
                      <li>• Scalable to sub-20nm</li>
                      <li>• Next-generation memory</li>
                    </ul>
                  </div>
                </div>
              </div>

              <div>
                <h3 className="text-lg font-semibold mb-3">Spin-Transfer Torque</h3>
                <div className="bg-muted p-4 rounded-lg">
                  <p className="text-sm mb-3">
                    Current flowing through a magnetic layer can exert torque on the magnetization:
                  </p>
                  <BlockMath math="\frac{d\mathbf{m}}{dt} = -\gamma(\mathbf{m} \times \mathbf{H}_{\text{eff}}) - \alpha\gamma(\mathbf{m} \times (\mathbf{m} \times \mathbf{H}_{\text{eff}})) + \gamma(\mathbf{m} \times \boldsymbol{\tau}_{\text{STT}})" />
                  
                  <div className="mt-3 text-sm">
                    <strong>Applications:</strong>
                    <ul className="list-disc list-inside mt-2 space-y-1">
                      <li>Current-controlled magnetic switching</li>
                      <li>Spin-valve devices</li>
                      <li>Magnetic tunnel junctions</li>
                      <li>Spin-torque oscillators</li>
                    </ul>
                  </div>
                </div>
              </div>

              <div>
                <h3 className="text-lg font-semibold mb-3">Magnetic Logic Devices</h3>
                <div className="grid md:grid-cols-3 gap-4">
                  <div className="bg-muted p-4 rounded-lg">
                    <h4 className="font-medium mb-2">Domain Wall Logic</h4>
                    <p className="text-sm">
                      Information encoded in magnetic domain walls,
                      manipulated by magnetic fields or currents.
                    </p>
                  </div>
                  
                  <div className="bg-muted p-4 rounded-lg">
                    <h4 className="font-medium mb-2">Skyrmion Logic</h4>
                    <p className="text-sm">
                      Topologically protected magnetic textures
                      for ultra-low power computation.
                    </p>
                  </div>
                  
                  <div className="bg-muted p-4 rounded-lg">
                    <h4 className="font-medium mb-2">Spin Wave Logic</h4>
                    <p className="text-sm">
                      Information processing using magnon currents
                      for beyond-CMOS computing.
                    </p>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="resonance" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Zap className="h-5 w-5" />
                Magnetic Resonance Physics
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              <div>
                <h3 className="text-lg font-semibold mb-3">Ferromagnetic Resonance (FMR)</h3>
                <p className="text-sm text-muted-foreground mb-4">
                  FMR occurs when the driving frequency matches the natural precession frequency.
                </p>
                
                <div className="bg-muted p-4 rounded-lg">
                  <h4 className="font-medium mb-2">Resonance Condition</h4>
                  <BlockMath math="\omega_{\text{res}} = \gamma \sqrt{H_{\text{eff}} (H_{\text{eff}} + M_{\text{eff}})}" />
                  
                  <div className="mt-3 text-sm">
                    <p><strong>For thin films:</strong></p>
                    <BlockMath math="\omega_{\text{res}} = \gamma \sqrt{(H_0 + H_k)(H_0 + H_k + 4\pi M_s)}" />
                    
                    <div className="mt-2">
                      <div>H₀: Applied field</div>
                      <div>Hₖ: Anisotropy field</div>
                      <div>Ms: Saturation magnetization</div>
                    </div>
                  </div>
                </div>
              </div>

              <div>
                <h3 className="text-lg font-semibold mb-3">Dynamic Susceptibility</h3>
                <div className="bg-muted p-4 rounded-lg">
                  <p className="text-sm mb-3">
                    The response to small AC magnetic fields reveals the magnetic properties:
                  </p>
                  <BlockMath math="\chi(\omega) = \frac{\gamma M_s \omega_M}{(\omega_0^2 - \omega^2) + i\alpha\gamma M_s \omega}" />
                  
                  <div className="mt-3 text-sm space-y-1">
                    <div>ω₀: Resonance frequency</div>
                    <div>ωₘ: 4πγMs (gyromagnetic frequency)</div>
                    <div>Peak width ∝ α (damping parameter)</div>
                  </div>
                </div>
              </div>

              <div>
                <h3 className="text-lg font-semibold mb-3">Applications</h3>
                <div className="grid md:grid-cols-2 gap-4">
                  <div className="bg-muted p-4 rounded-lg">
                    <h4 className="font-medium mb-2">Material Characterization</h4>
                    <ul className="text-sm space-y-1">
                      <li>• Gyromagnetic ratio measurement</li>
                      <li>• Damping parameter extraction</li>
                      <li>• Anisotropy field determination</li>
                      <li>• Interface exchange coupling</li>
                    </ul>
                  </div>
                  
                  <div className="bg-muted p-4 rounded-lg">
                    <h4 className="font-medium mb-2">Device Applications</h4>
                    <ul className="text-sm space-y-1">
                      <li>• RF filters and circulators</li>
                      <li>• Microwave generators</li>
                      <li>• Magnetic field sensors</li>
                      <li>• Spin-wave devices</li>
                    </ul>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="numerical" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Atom className="h-5 w-5" />
                Numerical Methods
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              <div>
                <h3 className="text-lg font-semibold mb-3">Integration Schemes</h3>
                <p className="text-sm text-muted-foreground mb-4">
                  Solving the LLG equation requires careful numerical treatment to maintain |m| = 1.
                </p>
                
                <div className="grid md:grid-cols-2 gap-4">
                  <div className="bg-muted p-4 rounded-lg">
                    <h4 className="font-medium mb-2 text-red-600">Euler Method (Simple)</h4>
                    <BlockMath math="\mathbf{m}_{n+1} = \mathbf{m}_n + \Delta t \frac{d\mathbf{m}}{dt}\bigg|_n" />
                    <div className="text-sm mt-2">
                      <strong>Pros:</strong> Simple, fast<br/>
                      <strong>Cons:</strong> Poor stability, |m| drift
                    </div>
                  </div>
                  
                  <div className="bg-muted p-4 rounded-lg">
                    <h4 className="font-medium mb-2 text-green-600">RK4 Method (Used Here)</h4>
                    <BlockMath math="\mathbf{m}_{n+1} = \mathbf{m}_n + \frac{\Delta t}{6}(\mathbf{k}_1 + 2\mathbf{k}_2 + 2\mathbf{k}_3 + \mathbf{k}_4)" />
                    <div className="text-sm mt-2">
                      <strong>Pros:</strong> Higher accuracy, better stability<br/>
                      <strong>Cons:</strong> More computational cost
                    </div>
                  </div>
                </div>
              </div>

              <div>
                <h3 className="text-lg font-semibold mb-3">Normalization Constraint</h3>
                <div className="bg-muted p-4 rounded-lg">
                  <p className="text-sm mb-3">
                    Critical requirement: magnetization magnitude must remain unity.
                  </p>
                  <BlockMath math="|\mathbf{m}| = 1 \text{ at all times}" />
                  
                  <div className="mt-3 text-sm">
                    <strong>Implementation:</strong>
                    <ol className="list-decimal list-inside mt-2 space-y-1">
                      <li>Solve LLG equation for Δm</li>
                      <li>Update: m_new = m_old + Δm</li>
                      <li>Normalize: m_final = m_new / |m_new|</li>
                      <li>Check: |m_final| = 1.000000</li>
                    </ol>
                  </div>
                </div>
              </div>

              <div>
                <h3 className="text-lg font-semibold mb-3">Stability Considerations</h3>
                <div className="space-y-4">
                  <div className="bg-muted p-4 rounded-lg">
                    <h4 className="font-medium mb-2">Time Step Selection</h4>
                    <BlockMath math="\Delta t < \frac{1}{\gamma H_{\text{max}}}" />
                    <p className="text-sm mt-2">
                      Time step must be smaller than the inverse of the maximum frequency
                      to avoid numerical instabilities.
                    </p>
                  </div>
                  
                  <div className="bg-muted p-4 rounded-lg">
                    <h4 className="font-medium mb-2">Energy Conservation</h4>
                    <p className="text-sm">
                      In the absence of damping (α = 0), total energy should be conserved.
                      Energy drift indicates numerical errors.
                    </p>
                  </div>
                </div>
              </div>

              <div>
                <h3 className="text-lg font-semibold mb-3">Advanced Methods</h3>
                <div className="grid md:grid-cols-3 gap-4 text-sm">
                  <div className="bg-muted p-4 rounded-lg">
                    <h4 className="font-medium mb-2">Implicit Methods</h4>
                    <p>Better stability for stiff problems, used in micromagnetic simulations.</p>
                  </div>
                  
                  <div className="bg-muted p-4 rounded-lg">
                    <h4 className="font-medium mb-2">Midpoint Method</h4>
                    <p>Preserves |m| = 1 exactly, popular in OOMMF simulations.</p>
                  </div>
                  
                  <div className="bg-muted p-4 rounded-lg">
                    <h4 className="font-medium mb-2">Adaptive Stepping</h4>
                    <p>Dynamic time step adjustment based on error estimates.</p>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}