import React, { Suspense } from 'react';
import { Card } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Button } from '@/components/ui/button';
import { Play, BookOpen, BarChart3, Settings } from 'lucide-react';
import { PhysicsModuleLayout, PhysicsContentCard } from '@/components/PhysicsModuleLayout';
import { PhysicsModuleHeader } from '@/components/PhysicsModuleHeader';
import { LLG3DVisualization } from '@/components/llg/LLG3DVisualization';
import { LLGTimePlots } from '@/components/llg/LLGTimePlots';
import { LLGControlPanel } from '@/components/llg/LLGControlPanel';
import { LLGTheoryPanel } from '@/components/llg/LLGTheoryPanel';
import { useLLGStore } from '@/lib/llg-store';
import { useSEO } from '@/hooks/use-seo';

export default function LLGDynamics() {
  const { isRunning, timeData } = useLLGStore();
  useSEO({ title: 'LLG Magnetization Dynamics – SimCore', description: 'Simulate Landau-Lifshitz-Gilbert magnetization dynamics in external fields.' });
  const schemaLD = {
    '@context': 'https://schema.org',
    '@type': 'TechArticle',
    headline: 'LLG Magnetization Dynamics',
    description: 'Interactive Landau-Lifshitz-Gilbert magnetization dynamics simulation with 3D visualization and analysis.',
    inLanguage: 'en',
    author: { '@type': 'Organization', name: 'SimCore' },
    publisher: { '@type': 'Organization', name: 'SimCore' },
    keywords: 'LLG, magnetization dynamics, spintronics, Gilbert damping, magnetic resonance',
    about: ['LLG equation', 'Magnetization precession', 'Gilbert damping', 'Spin dynamics'],
    mainEntityOfPage: typeof window !== 'undefined' ? window.location.href : ''
  } as const;

  return (
    <PhysicsModuleLayout background="fields">
      <script type="application/ld+json" dangerouslySetInnerHTML={{ __html: JSON.stringify(schemaLD) }} />
      <PhysicsModuleHeader
        title="LLG Magnetization Dynamics"
        description="Real-time simulation of the Landau-Lifshitz-Gilbert equation governing magnetic moment evolution in external fields"
        category="Spin & Magnetism"
        difficulty="Advanced"
        equation="\frac{d\vec{m}}{dt} = -\gamma \vec{m} \times \vec{H}_{eff} + \alpha \vec{m} \times \frac{d\vec{m}}{dt}"
        isRunning={isRunning}
        currentStep={timeData.length}
        onReset={() => useLLGStore.getState().resetSimulation()}
        onExport={() => useLLGStore.getState().exportTrajectory()}
      />
      <Tabs defaultValue="simulation" className="space-y-6">
        <TabsList className="grid w-full grid-cols-2 sm:grid-cols-4 gap-2 sm:gap-0 bg-surface/80 backdrop-blur-sm border border-muted/30">
          <TabsTrigger 
            value="simulation" 
            className="flex items-center gap-1 sm:gap-2 data-[state=active]:bg-interactive data-[state=active]:text-white transition-all duration-300"
          >
            <Play className="h-3 w-3 sm:h-4 sm:w-4" />
            <span className="hidden sm:inline">Simulation</span>
            <span className="sm:hidden">Sim</span>
          </TabsTrigger>
          <TabsTrigger 
            value="analysis" 
            className="flex items-center gap-1 sm:gap-2 data-[state=active]:bg-interactive data-[state=active]:text-white transition-all duration-300"
          >
            <BarChart3 className="h-3 w-3 sm:h-4 sm:w-4" />
            <span className="hidden sm:inline">Analysis</span>
            <span className="sm:hidden">Data</span>
          </TabsTrigger>
          <TabsTrigger 
            value="theory" 
            className="flex items-center gap-1 sm:gap-2 data-[state=active]:bg-interactive data-[state=active]:text-white transition-all duration-300"
          >
            <BookOpen className="h-3 w-3 sm:h-4 sm:w-4" />
            <span className="hidden sm:inline">Theory</span>
            <span className="sm:hidden">Info</span>
          </TabsTrigger>
          <TabsTrigger 
            value="controls" 
            className="flex items-center gap-1 sm:gap-2 data-[state=active]:bg-interactive data-[state=active]:text-white transition-all duration-300"
          >
            <Settings className="h-3 w-3 sm:h-4 sm:w-4" />
            <span className="hidden sm:inline">Controls</span>
            <span className="sm:hidden">Ctrl</span>
          </TabsTrigger>
        </TabsList>

        <TabsContent value="simulation" className="space-y-4 sm:space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-4 gap-4 sm:gap-6">
            {/* Left Panel - Controls - Mobile First */}
            <div className="lg:col-span-1 order-1 lg:order-first">
              <div className="sticky top-4">
                <PhysicsContentCard title="Controls" variant="glass">
                  <LLGControlPanel />
                </PhysicsContentCard>
              </div>
            </div>

            {/* Right Panel - Visualizations - Mobile Responsive */}
            <div className="lg:col-span-3 space-y-4 sm:space-y-6">
              {/* Top Row: 3D + 2D Side by Side - Stack on Mobile */}
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 sm:gap-6">
                <PhysicsContentCard 
                  title="3D Magnetization Dynamics"
                  description="Real-time LLG evolution on unit sphere"
                  variant="elegant"
                  glow
                  className="h-[500px]"
                >
                  <div className="h-[400px]">
                    <Suspense 
                      fallback={
                        <div className="h-full flex items-center justify-center">
                          <div className="text-center">
                            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary mx-auto mb-4"></div>
                            <p className="text-sm text-muted-foreground">Loading 3D visualization...</p>
                          </div>
                        </div>
                      }
                    >
                      <LLG3DVisualization />
                    </Suspense>
                  </div>
                </PhysicsContentCard>

                <PhysicsContentCard 
                  title="Time Evolution"
                  description="m(t), |m(t)|, and energy dynamics"
                  variant="elegant"
                  glow
                  className="h-[500px]"
                >
                  <div className="h-[400px] p-2">
                    <LLGTimePlots />
                  </div>
                </PhysicsContentCard>
              </div>

              {/* Bottom Row: Physics Information */}
              <PhysicsContentCard title="Physics Overview" variant="glass" className="p-6">
                <div className="grid md:grid-cols-3 gap-6">
                  <div>
                    <h4 className="font-semibold mb-2">LLG Equation</h4>
                    <p className="text-sm text-muted-foreground">
                      The Landau-Lifshitz-Gilbert equation describes magnetization dynamics
                      with precession around the effective field and Gilbert damping toward equilibrium.
                    </p>
                  </div>
                  
                  <div>
                    <h4 className="font-semibold mb-2">Numerical Stability</h4>
                    <p className="text-sm text-muted-foreground">
                      Uses RK4 integration with automatic renormalization to maintain |m| = 1.
                      Time step: 1 ps for femtosecond-scale magnetic dynamics.
                    </p>
                  </div>
                  
                  <div>
                    <h4 className="font-semibold mb-2">Applications</h4>
                    <p className="text-sm text-muted-foreground">
                      Fundamental to MRAM, spintronics, magnetic recording, and
                      ferromagnetic resonance device design.
                    </p>
                  </div>
                </div>
              </PhysicsContentCard>
            </div>
          </div>
        </TabsContent>

        <TabsContent value="analysis" className="space-y-6">
          <div className="grid lg:grid-cols-3 gap-6">
            <div className="lg:col-span-2">
              <PhysicsContentCard 
                title="Complete Time Series Analysis"
                variant="elegant"
                glow
                className="h-[600px]"
              >
                <div className="h-[530px] p-2">
                  <LLGTimePlots />
                </div>
              </PhysicsContentCard>
            </div>
            
            <div className="space-y-4">
              <PhysicsContentCard title="Analysis Tools" variant="glass">
                <div className="space-y-4">
                  <Button className="w-full" onClick={() => useLLGStore.getState().exportTrajectory()}>
                    Export CSV Data
                  </Button>
                  <Button variant="outline" className="w-full" onClick={() => useLLGStore.getState().clearHistory()}>
                    Clear History
                  </Button>
                </div>
              </PhysicsContentCard>

              <PhysicsContentCard title="Simulation Stats" variant="glass">
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span>Data Points:</span>
                    <span className="font-mono">{timeData.length}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Total Time:</span>
                    <span className="font-mono">
                      {(useLLGStore.getState().time * 1e9).toFixed(2)} ns
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span>Integration:</span>
                    <span className="font-mono">RK4</span>
                  </div>
                </div>
              </PhysicsContentCard>
            </div>
          </div>
        </TabsContent>

        <TabsContent value="theory" className="space-y-6">
          <LLGTheoryPanel />
        </TabsContent>

        <TabsContent value="controls" className="space-y-6">
          <div className="grid lg:grid-cols-2 gap-6">
            <PhysicsContentCard title="Controls" variant="elegant">
              <LLGControlPanel />
            </PhysicsContentCard>
            
            <div className="space-y-4">
              <PhysicsContentCard title="Keyboard Shortcuts" variant="glass">
                <div className="space-y-3 text-sm">
                  <div className="flex justify-between">
                    <span className="font-mono bg-muted px-2 py-1 rounded">Space</span>
                    <span>Play/Pause simulation</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="font-mono bg-muted px-2 py-1 rounded">R</span>
                    <span>Reset simulation</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="font-mono bg-muted px-2 py-1 rounded">E</span>
                    <span>Export trajectory</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="font-mono bg-muted px-2 py-1 rounded">C</span>
                    <span>Clear history</span>
                  </div>
                </div>
              </PhysicsContentCard>

              <PhysicsContentCard title="Physical Parameters" variant="glass">
                <div className="space-y-3 text-sm">
                  <div>
                    <strong>Gyromagnetic ratio:</strong> γ = 2.21×10⁵ rad·s⁻¹·T⁻¹
                  </div>
                  <div>
                    <strong>Time step:</strong> Δt = 1 ps
                  </div>
                  <div>
                    <strong>Integration:</strong> 4th-order Runge-Kutta
                  </div>
                  <div>
                    <strong>Constraint:</strong> |m| = 1 (enforced)
                  </div>
                </div>
              </PhysicsContentCard>
            </div>
          </div>
        </TabsContent>
      </Tabs>
    </PhysicsModuleLayout>
  );
}