import React from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { 
  BookOpen, Zap, Settings, Cpu, Target, TrendingUp, 
  Thermometer, Magnet, Activity, GitBranch 
} from 'lucide-react';
import { BlockMath, InlineMath } from '@/components/ui/Math';

export const IsingTheoryPanel: React.FC = () => {
  return (
    <div className="space-y-6">
      <Tabs defaultValue="theory" className="w-full">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="theory" className="flex items-center gap-2">
            <BookOpen className="w-4 h-4" />
            Theory
          </TabsTrigger>
          <TabsTrigger value="critical" className="flex items-center gap-2">
            <TrendingUp className="w-4 h-4" />
            Critical Phenomena
          </TabsTrigger>
          <TabsTrigger value="algorithms" className="flex items-center gap-2">
            <Settings className="w-4 h-4" />
            Algorithms
          </TabsTrigger>
          <TabsTrigger value="applications" className="flex items-center gap-2">
            <Target className="w-4 h-4" />
            Applications
          </TabsTrigger>
        </TabsList>

        {/* Core Theory */}
        <TabsContent value="theory" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Zap className="w-5 h-5 text-primary" />
                The 2D Ising Model
              </CardTitle>
              <CardDescription>
                Fundamental statistical mechanics model for ferromagnetism and phase transitions
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <h4 className="font-semibold mb-2">Hamiltonian</h4>
                <p className="text-sm text-muted-foreground mb-2">
                  The energy of the system with nearest-neighbor interactions:
                </p>
                <BlockMath math="H = -J \sum_{\langle i,j \rangle} \sigma_i \sigma_j - h \sum_i \sigma_i" />
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4 text-sm">
                  <div className="space-y-2">
                    <Badge variant="outline">σᵢ ∈ {'{-1, +1}'}</Badge>
                    <p className="text-xs">Spin variables (up/down)</p>
                  </div>
                  <div className="space-y-2">
                    <Badge variant="outline">J {'>'} 0</Badge>
                    <p className="text-xs">Ferromagnetic coupling</p>
                  </div>
                  <div className="space-y-2">
                    <Badge variant="outline">⟨i,j⟩</Badge>
                    <p className="text-xs">Nearest neighbor pairs</p>
                  </div>
                  <div className="space-y-2">
                    <Badge variant="outline">h</Badge>
                    <p className="text-xs">External magnetic field</p>
                  </div>
                </div>
              </div>

              <Separator />

              <div>
                <h4 className="font-semibold mb-2">Order Parameter</h4>
                <p className="text-sm text-muted-foreground mb-2">
                  The magnetization per spin quantifies magnetic order:
                </p>
                <BlockMath math="m = \frac{1}{N} \sum_{i=1}^N \sigma_i = \frac{1}{N} \sum_{i=1}^N \langle \sigma_i \rangle" />
                <div className="text-sm mt-2 space-y-1">
                  <p>• <strong>Ordered phase</strong> (T {'<'} Tc): m ≠ 0 (spontaneous magnetization)</p>
                  <p>• <strong>Disordered phase</strong> (T {'>'} Tc): m = 0 (no net magnetization)</p>
                </div>
              </div>

              <Separator />

              <div>
                <h4 className="font-semibold mb-2">Onsager's Exact Solution</h4>
                <p className="text-sm text-muted-foreground mb-2">
                  Critical temperature for the 2D square lattice:
                </p>
                <BlockMath math="T_c = \frac{2J}{k_B \ln(1 + \sqrt{2})} \approx 2.269 J/k_B" />
                <p className="text-sm text-muted-foreground">
                  Exact result derived by Lars Onsager in 1944 using transfer matrix methods.
                </p>
              </div>

              <Alert>
                <Thermometer className="h-4 w-4" />
                <AlertDescription className="text-sm">
                  The 2D Ising model exhibits a continuous phase transition at Tc with 
                  infinite correlation length and critical fluctuations.
                </AlertDescription>
              </Alert>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Statistical Mechanics Framework</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <h4 className="font-semibold mb-2">Partition Function</h4>
                <BlockMath math="Z = \sum_{\{\sigma\}} \exp(-\beta H[\{\sigma\}])" />
                <p className="text-sm text-muted-foreground">
                  Sum over all 2^N possible spin configurations, β = 1/(kBT)
                </p>
              </div>

              <div>
                <h4 className="font-semibold mb-2">Thermodynamic Observables</h4>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
                  <div>
                    <Badge variant="outline" className="mb-2">Free Energy</Badge>
                    <BlockMath math="F = -k_B T \ln Z" />
                  </div>
                  <div>
                    <Badge variant="outline" className="mb-2">Internal Energy</Badge>
                    <BlockMath math="U = \langle H \rangle = -\frac{\partial \ln Z}{\partial \beta}" />
                  </div>
                  <div>
                    <Badge variant="outline" className="mb-2">Heat Capacity</Badge>
                    <BlockMath math="C_v = \frac{1}{k_B T^2} (\langle H^2 \rangle - \langle H \rangle^2)" />
                  </div>
                  <div>
                    <Badge variant="outline" className="mb-2">Susceptibility</Badge>
                    <BlockMath math="\chi = \frac{1}{k_B T} (\langle M^2 \rangle - \langle M \rangle^2)" />
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Critical Phenomena */}
        <TabsContent value="critical" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <TrendingUp className="w-5 h-5 text-primary" />
                Critical Phenomena & Scaling
              </CardTitle>
              <CardDescription>
                Universal behavior near the phase transition
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <h4 className="font-semibold mb-2">Critical Exponents</h4>
                <p className="text-sm text-muted-foreground mb-3">
                  Power law behavior near Tc defines universality class:
                </p>
                
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="space-y-3">
                    <div>
                      <Badge variant="outline" className="mb-1">Magnetization</Badge>
                      <BlockMath math="m \sim |T - T_c|^\beta, \quad \beta = \frac{1}{8}" />
                    </div>
                    <div>
                      <Badge variant="outline" className="mb-1">Susceptibility</Badge>
                      <BlockMath math="\chi \sim |T - T_c|^{-\gamma}, \quad \gamma = \frac{7}{4}" />
                    </div>
                  </div>
                  <div className="space-y-3">
                    <div>
                      <Badge variant="outline" className="mb-1">Heat Capacity</Badge>
                      <BlockMath math="C_v \sim |T - T_c|^{-\alpha}, \quad \alpha = 0" />
                    </div>
                    <div>
                      <Badge variant="outline" className="mb-1">Correlation Length</Badge>
                      <BlockMath math="\xi \sim |T - T_c|^{-\nu}, \quad \nu = 1" />
                    </div>
                  </div>
                </div>

                <Alert className="mt-4">
                  <Activity className="h-4 w-4" />
                  <AlertDescription className="text-sm">
                    These exponents are universal - they depend only on dimensionality and 
                    symmetry, not on microscopic details of the system.
                  </AlertDescription>
                </Alert>
              </div>

              <Separator />

              <div>
                <h4 className="font-semibold mb-2">Scaling Relations</h4>
                <p className="text-sm text-muted-foreground mb-2">
                  Critical exponents satisfy exact relations:
                </p>
                <div className="space-y-2">
                  <BlockMath math="\alpha + 2\beta + \gamma = 2 \quad \text{(Rushbrooke)}" />
                  <BlockMath math="\gamma = \beta(\delta - 1) \quad \text{(Widom)}" />
                  <BlockMath math="\nu d = 2 - \alpha \quad \text{(Fisher)}" />
                </div>
                <p className="text-sm text-muted-foreground mt-2">
                  Where d = 2 (dimension) and δ = 15 (critical isotherm exponent).
                </p>
              </div>

              <Separator />

              <div>
                <h4 className="font-semibold mb-2">Finite Size Effects</h4>
                <p className="text-sm text-muted-foreground mb-2">
                  In finite systems, critical behavior is modified:
                </p>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
                  <div>
                    <Badge variant="outline" className="mb-2">Correlation Length</Badge>
                    <p>ξ ≤ L (system size cutoff)</p>
                  </div>
                  <div>
                    <Badge variant="outline" className="mb-2">Critical Temperature</Badge>
                    <p>Tc(L) - Tc(∞) ∼ L^(-1/ν)</p>
                  </div>
                  <div>
                    <Badge variant="outline" className="mb-2">Heat Capacity</Badge>
                    <p>Cv peak ∼ L^(α/ν) ∼ L^0 (logarithmic)</p>
                  </div>
                  <div>
                    <Badge variant="outline" className="mb-2">Susceptibility</Badge>
                    <p>χ peak ∼ L^(γ/ν) ∼ L^(7/4)</p>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <GitBranch className="w-5 h-5 text-primary" />
                Correlation Functions
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <h4 className="font-semibold mb-2">Spin-Spin Correlation</h4>
                <BlockMath math="G(r) = \langle \sigma_0 \sigma_r \rangle - \langle \sigma_0 \rangle \langle \sigma_r \rangle" />
                <p className="text-sm text-muted-foreground">
                  Measures how correlated spins are at distance r.
                </p>
              </div>

              <div>
                <h4 className="font-semibold mb-2">Exponential Decay</h4>
                <BlockMath math="G(r) \sim \exp(-r/\xi) \quad \text{for } T \neq T_c" />
                <p className="text-sm text-muted-foreground">
                  Correlation length ξ diverges as |T - Tc|^(-ν) at the critical point.
                </p>
              </div>

              <div>
                <h4 className="font-semibold mb-2">Critical Point Behavior</h4>
                <BlockMath math="G(r) \sim r^{-\eta} \quad \text{at } T = T_c" />
                <p className="text-sm text-muted-foreground">
                  Power law decay with η = 1/4 for the 2D Ising model.
                </p>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Algorithms */}
        <TabsContent value="algorithms" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Cpu className="w-5 h-5 text-primary" />
                Monte Carlo Algorithms
              </CardTitle>
              <CardDescription>
                Different approaches to sample the Boltzmann distribution
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div>
                <h4 className="font-semibold mb-2 flex items-center gap-2">
                  <Badge variant="outline">Metropolis</Badge>
                  Single-Spin Flip Algorithm
                </h4>
                <div className="space-y-2 text-sm">
                  <p><strong>Procedure:</strong></p>
                  <ol className="list-decimal list-inside space-y-1 ml-4">
                    <li>Select random spin σᵢ</li>
                    <li>Calculate energy change: ΔE = 2σᵢ(Σⱼσⱼ + h)</li>
                    <li>Accept flip with probability: min(1, e^(-βΔE))</li>
                    <li>Repeat for all spins (one Monte Carlo step)</li>
                  </ol>
                  <div className="mt-3 p-3 bg-muted/50 rounded">
                    <p className="font-medium">Advantages:</p>
                    <p>• Simple to implement</p>
                    <p>• Works at all temperatures</p>
                    <p>• Local detailed balance</p>
                    <p className="font-medium mt-2">Disadvantages:</p>
                    <p>• Critical slowing down near Tc</p>
                    <p>• τ ∼ ξ^z with z ≈ 2</p>
                  </div>
                </div>
              </div>

              <Separator />

              <div>
                <h4 className="font-semibold mb-2 flex items-center gap-2">
                  <Badge variant="outline">Wolff</Badge>
                  Cluster Algorithm
                </h4>
                <div className="space-y-2 text-sm">
                  <p><strong>Procedure:</strong></p>
                  <ol className="list-decimal list-inside space-y-1 ml-4">
                    <li>Select random seed spin</li>
                    <li>Add aligned neighbors with probability p = 1 - e^(-2βJ)</li>
                    <li>Recursively grow cluster</li>
                    <li>Flip entire cluster</li>
                  </ol>
                  <div className="mt-3 p-3 bg-muted/50 rounded">
                    <p className="font-medium">Advantages:</p>
                    <p>• Eliminates critical slowing down</p>
                    <p>• τ ∼ const near Tc</p>
                    <p>• Non-local updates</p>
                    <p className="font-medium mt-2">Disadvantages:</p>
                    <p>• More complex implementation</p>
                    <p>• Less efficient away from Tc</p>
                  </div>
                </div>
              </div>

              <Separator />

              <div>
                <h4 className="font-semibold mb-2 flex items-center gap-2">
                  <Badge variant="outline">Heat Bath</Badge>
                  Direct Sampling
                </h4>
                <div className="space-y-2 text-sm">
                  <p><strong>Procedure:</strong></p>
                  <ol className="list-decimal list-inside space-y-1 ml-4">
                    <li>For each spin, calculate local field: h_local = Σⱼσⱼ + h</li>
                    <li>Set σᵢ = +1 with probability: p = e^(βh_local)/(e^(βh_local) + e^(-βh_local))</li>
                    <li>Otherwise set σᵢ = -1</li>
                  </ol>
                  <div className="mt-3 p-3 bg-muted/50 rounded">
                    <p className="font-medium">Advantages:</p>
                    <p>• Direct equilibrium sampling</p>
                    <p>• Fast thermalization</p>
                    <p>• No rejection steps</p>
                    <p className="font-medium mt-2">Disadvantages:</p>
                    <p>• Still suffers critical slowing down</p>
                    <p>• Requires full scan each step</p>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Equilibration and Autocorrelation</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <h4 className="font-semibold mb-2">Autocorrelation Time</h4>
                <BlockMath math="\tau = \sum_{t=0}^{\infty} C(t), \quad C(t) = \frac{\langle O(0)O(t) \rangle - \langle O \rangle^2}{\langle O^2 \rangle - \langle O \rangle^2}" />
                <p className="text-sm text-muted-foreground">
                  Measures how long it takes for the system to "forget" its initial state.
                </p>
              </div>

              <div>
                <h4 className="font-semibold mb-2">Error Analysis</h4>
                <BlockMath math="\sigma_{\bar{O}} = \sqrt{\frac{2\tau}{N}} \sigma_O" />
                <p className="text-sm text-muted-foreground">
                  Statistical error in Monte Carlo averages including autocorrelation effects.
                </p>
              </div>

              <Alert>
                <Settings className="h-4 w-4" />
                <AlertDescription className="text-sm">
                  Near Tc, always allow for sufficient equilibration (≫ τ steps) before 
                  collecting measurements to avoid systematic errors.
                </AlertDescription>
              </Alert>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Applications */}
        <TabsContent value="applications" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Target className="w-5 h-5 text-primary" />
                Physical Systems & Applications
              </CardTitle>
              <CardDescription>
                The Ising model appears throughout physics and beyond
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <h4 className="font-semibold mb-2 flex items-center gap-2">
                    <Magnet className="w-4 h-4" />
                    Magnetic Materials
                  </h4>
                  <ul className="text-sm space-y-1">
                    <li>• Iron, nickel, cobalt ferromagnets</li>
                    <li>• Magnetic domain formation</li>
                    <li>• Curie temperature transitions</li>
                    <li>• Hysteresis loops</li>
                    <li>• Spin glasses and frustration</li>
                  </ul>
                </div>

                <div>
                  <h4 className="font-semibold mb-2 flex items-center gap-2">
                    <Activity className="w-4 h-4" />
                    Binary Alloys
                  </h4>
                  <ul className="text-sm space-y-1">
                    <li>• Order-disorder transitions</li>
                    <li>• Solid solution vs. compound formation</li>
                    <li>• Critical mixing temperatures</li>
                    <li>• Phase separation dynamics</li>
                    <li>• Atomic ordering in crystals</li>
                  </ul>
                </div>

                <div>
                  <h4 className="font-semibold mb-2 flex items-center gap-2">
                    <Cpu className="w-4 h-4" />
                    Neural Networks
                  </h4>
                  <ul className="text-sm space-y-1">
                    <li>• Hopfield networks</li>
                    <li>• Memory storage and retrieval</li>
                    <li>• Pattern recognition</li>
                    <li>• Spin glass models of learning</li>
                    <li>• Boltzmann machines</li>
                  </ul>
                </div>

                <div>
                  <h4 className="font-semibold mb-2 flex items-center gap-2">
                    <BookOpen className="w-4 h-4" />
                    Social Dynamics
                  </h4>
                  <ul className="text-sm space-y-1">
                    <li>• Opinion formation models</li>
                    <li>• Social consensus dynamics</li>
                    <li>• Cultural transmission</li>
                    <li>• Voter models</li>
                    <li>• Market herding behavior</li>
                  </ul>
                </div>

                <div>
                  <h4 className="font-semibold mb-2 flex items-center gap-2">
                    <Zap className="w-4 h-4" />
                    Biological Systems
                  </h4>
                  <ul className="text-sm space-y-1">
                    <li>• Protein folding transitions</li>
                    <li>• DNA denaturation</li>
                    <li>• Membrane phase transitions</li>
                    <li>• Collective cell behavior</li>
                    <li>• Epidemic spreading</li>
                  </ul>
                </div>

                <div>
                  <h4 className="font-semibold mb-2 flex items-center gap-2">
                    <TrendingUp className="w-4 h-4" />
                    Financial Markets
                  </h4>
                  <ul className="text-sm space-y-1">
                    <li>• Price fluctuation models</li>
                    <li>• Market crash dynamics</li>
                    <li>• Herding in trading</li>
                    <li>• Volatility clustering</li>
                    <li>• Agent-based models</li>
                  </ul>
                </div>
              </div>

              <Alert className="mt-6">
                <Target className="h-4 w-4" />
                <AlertDescription>
                  <strong>Universal Behavior:</strong> The 2D Ising model belongs to the same 
                  universality class as liquid-gas transitions, binary alloy unmixing, and 
                  many other systems with Z₂ symmetry breaking.
                </AlertDescription>
              </Alert>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Modern Research Directions</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
                <div>
                  <h4 className="font-semibold mb-2">Quantum Extensions</h4>
                  <ul className="space-y-1">
                    <li>• Quantum Ising models</li>
                    <li>• Transverse field effects</li>
                    <li>• Quantum phase transitions</li>
                    <li>• Quantum annealing</li>
                  </ul>
                </div>
                
                <div>
                  <h4 className="font-semibold mb-2">Non-equilibrium</h4>
                  <ul className="space-y-1">
                    <li>• Driven systems</li>
                    <li>• Quench dynamics</li>
                    <li>• Aging phenomena</li>
                    <li>• Active matter</li>
                  </ul>
                </div>
                
                <div>
                  <h4 className="font-semibold mb-2">Machine Learning</h4>
                  <ul className="space-y-1">
                    <li>• Neural network training</li>
                    <li>• Phase recognition algorithms</li>
                    <li>• Variational methods</li>
                    <li>• Tensor networks</li>
                  </ul>
                </div>
                
                <div>
                  <h4 className="font-semibold mb-2">Complex Networks</h4>
                  <ul className="space-y-1">
                    <li>• Random graphs</li>
                    <li>• Small-world networks</li>
                    <li>• Scale-free topologies</li>
                    <li>• Multilayer systems</li>
                  </ul>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default IsingTheoryPanel;