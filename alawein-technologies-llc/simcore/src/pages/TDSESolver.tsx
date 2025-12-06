import React, { useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import { ArrowLeft } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { PhysicsModuleLayout } from '@/components/PhysicsModuleLayout';
import { PhysicsModuleHeader } from '@/components/PhysicsModuleHeader';
import { EnhancedTDSEVisualization } from '@/components/EnhancedTDSEVisualization';
import { PerformanceMonitorDashboard } from '@/components/PerformanceMonitorDashboard';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { InlineMath, BlockMath } from '@/components/ui/Math';

import { useSEO } from '@/hooks/use-seo';

const TDSESolver = () => {
  const navigate = useNavigate();
  useSEO({ title: 'TDSE Solver – SimCore', description: 'Advanced quantum simulation with real-time visualization of wave function evolution.' });
  const schemaLD = useMemo(() => ({
    '@context': 'https://schema.org',
    '@type': 'TechArticle',
    headline: 'Time-Dependent Schrödinger Equation Solver',
    description: 'Advanced quantum simulation with real-time visualization of wave function evolution.',
    inLanguage: 'en',
    author: { '@type': 'Organization', name: 'SimCore' },
    publisher: { '@type': 'Organization', name: 'SimCore' },
    keywords: 'TDSE, Schrödinger equation, quantum simulation, wave function',
    about: ['Quantum mechanics', 'TDSE', 'Numerical methods'],
    mainEntityOfPage: typeof window !== 'undefined' ? window.location.href : ''
  }), []);
  const breadcrumbsLD = useMemo(() => ({
    '@context': 'https://schema.org',
    '@type': 'BreadcrumbList',
    itemListElement: [
      { '@type': 'ListItem', position: 1, name: 'Home', item: typeof window !== 'undefined' ? window.location.origin : '/' },
      { '@type': 'ListItem', position: 2, name: 'Modules', item: typeof window !== 'undefined' ? `${window.location.origin}/#modules-section` : '/#modules-section' },
      { '@type': 'ListItem', position: 3, name: 'Time-Dependent Schrödinger Equation Solver', item: typeof window !== 'undefined' ? window.location.href : '' }
    ]
  }), []);

  return (
    <PhysicsModuleLayout>
      <script type="application/ld+json" dangerouslySetInnerHTML={{ __html: JSON.stringify(schemaLD) }} />
      
      <PhysicsModuleHeader
        title="Time-Dependent Schrödinger Equation Solver"
        description="Advanced quantum simulation with real-time visualization of wave function evolution"
        category="Quantum Mechanics"
        difficulty="Advanced"
        equation="i\hbar\frac{\partial\Psi}{\partial t} = \hat{H}\Psi"
      />

      <div className="container mx-auto px-4 py-8">
        <div className="max-w-7xl mx-auto">
          {/* Navigation */}
          <div className="mb-6">
            <Button
              variant="outline"
              onClick={() => navigate('/')}
              className="flex items-center gap-2"
            >
              <ArrowLeft className="w-4 h-4" />
              Back to Modules
            </Button>
          </div>

          <Tabs defaultValue="simulation" className="w-full">
            <TabsList className="grid w-full grid-cols-1 sm:grid-cols-3 gap-2 sm:gap-0 bg-surface/80 backdrop-blur-sm border border-muted/30">
              <TabsTrigger 
                value="simulation" 
                className="data-[state=active]:bg-interactive data-[state=active]:text-white transition-all duration-300"
              >
                Interactive Simulation
              </TabsTrigger>
              <TabsTrigger 
                value="theory" 
                className="data-[state=active]:bg-interactive data-[state=active]:text-white transition-all duration-300"
              >
                Theory & Mathematics
              </TabsTrigger>
              <TabsTrigger 
                value="performance" 
                className="data-[state=active]:bg-interactive data-[state=active]:text-white transition-all duration-300"
              >
                Performance Monitor
              </TabsTrigger>
            </TabsList>

            <TabsContent value="simulation" className="space-y-6">
              <EnhancedTDSEVisualization />
            </TabsContent>

            <TabsContent value="theory" className="space-y-6">
              <div className="grid gap-6">
                {/* Theoretical Foundation */}
                <div className="bg-surface/90 backdrop-blur-sm rounded-lg p-6 border border-muted/30 shadow-elevation1">
                  <h3 className="text-2xl font-bold mb-4 text-textPrimary">Time-Dependent Schrödinger Equation</h3>
                  
                  <div className="space-y-4">
                    <p className="text-muted-foreground">
                      The time-dependent Schrödinger equation is a fundamental equation in quantum mechanics 
                      that describes how quantum states evolve over time.
                    </p>

                    <div className="bg-muted/20 p-4 rounded-lg">
                      <BlockMath math="i\hbar\frac{\partial\Psi(x,t)}{\partial t} = \hat{H}\Psi(x,t)" />
                    </div>

                    <p className="text-sm text-muted-foreground">
                      Where <InlineMath math="\hat{H}" /> is the Hamiltonian operator, typically:
                    </p>

                    <div className="bg-muted/20 p-4 rounded-lg">
                      <BlockMath math="\hat{H} = -\frac{\hbar^2}{2m}\frac{\partial^2}{\partial x^2} + V(x,t)" />
                    </div>
                  </div>
                </div>

                {/* Numerical Methods */}
                <div className="bg-card rounded-lg p-6 border">
                  <h3 className="text-xl font-bold mb-4">Numerical Solution Methods</h3>
                  
                  <div className="space-y-4">
                    <div>
                      <h4 className="font-semibold mb-2">Split-Operator Method</h4>
                      <p className="text-sm text-muted-foreground mb-2">
                        The most accurate method for time evolution, using operator splitting:
                      </p>
                      <div className="bg-muted/20 p-3 rounded">
                        <BlockMath math="e^{-i\hat{H}\Delta t/\hbar} \approx e^{-i\hat{T}\Delta t/2\hbar}e^{-i\hat{V}\Delta t/\hbar}e^{-i\hat{T}\Delta t/2\hbar}" />
                      </div>
                    </div>

                    <div>
                      <h4 className="font-semibold mb-2">Crank-Nicolson Method</h4>
                      <p className="text-sm text-muted-foreground mb-2">
                        Implicit finite difference method with excellent stability:
                      </p>
                      <div className="bg-muted/20 p-3 rounded">
                        <BlockMath math="\Psi^{n+1} = \Psi^n + \frac{\Delta t}{2i\hbar}(\hat{H}\Psi^{n+1} + \hat{H}\Psi^n)" />
                      </div>
                    </div>

                    <div>
                      <h4 className="font-semibold mb-2">Conservation Laws</h4>
                      <p className="text-sm text-muted-foreground mb-2">
                        The simulation preserves important quantum mechanical conservation laws:
                      </p>
                      <ul className="text-sm text-muted-foreground space-y-1 ml-4">
                        <li>• Probability conservation: <InlineMath math="\int |\Psi|^2 dx = 1" /></li>
                        <li>• Energy conservation (for time-independent Hamiltonians)</li>
                        <li>• Unitary evolution: <InlineMath math="|\langle\psi_1|\psi_2\rangle|" /> preserved</li>
                      </ul>
                    </div>
                  </div>
                </div>

                {/* Applications */}
                <div className="bg-card rounded-lg p-6 border">
                  <h3 className="text-xl font-bold mb-4">Physical Applications</h3>
                  
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div>
                      <h4 className="font-semibold mb-2">Quantum Tunneling</h4>
                      <p className="text-sm text-muted-foreground">
                        Study how wave packets tunnel through potential barriers, 
                        demonstrating the probabilistic nature of quantum mechanics.
                      </p>
                    </div>

                    <div>
                      <h4 className="font-semibold mb-2">Harmonic Oscillator</h4>
                      <p className="text-sm text-muted-foreground">
                        Observe coherent states evolving in quadratic potentials,
                        showing classical-like oscillatory behavior.
                      </p>
                    </div>

                    <div>
                      <h4 className="font-semibold mb-2">Wave Packet Dynamics</h4>
                      <p className="text-sm text-muted-foreground">
                        Track spreading and dispersion of localized wave packets,
                        illustrating the uncertainty principle in action.
                      </p>
                    </div>

                    <div>
                      <h4 className="font-semibold mb-2">Potential Wells</h4>
                      <p className="text-sm text-muted-foreground">
                        Explore bound and scattering states in various potential
                        configurations, from infinite wells to double wells.
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            </TabsContent>

            <TabsContent value="performance" className="space-y-6">
              <PerformanceMonitorDashboard />
            </TabsContent>
          </Tabs>
        </div>
      </div>
    </PhysicsModuleLayout>
  );
};

export default TDSESolver;