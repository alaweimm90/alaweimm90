import React, { Suspense } from 'react';
import { useNavigate } from 'react-router-dom';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { ArrowLeft, Atom, Eye, BookOpen, Settings } from 'lucide-react';
import { PhysicsModuleLayout } from '@/components/PhysicsModuleLayout';
import { PhysicsModuleHeader } from '@/components/PhysicsModuleHeader';
import { CrystalVisualization } from '@/components/crystal/CrystalVisualization';
import { ControlPanels } from '@/components/crystal/ControlPanels';
import { TheoryPanel } from '@/components/crystal/TheoryPanel';
import { useCrystalStore } from '@/lib/crystal-store';
import { useSEO } from '@/hooks/use-seo';

export default function CrystalVisualizer() {
  useSEO({ title: 'Crystal Structure Visualizer – SimCore', description: 'Interactive 3D crystallography with Miller planes and symmetry operations.' });
  const navigate = useNavigate();
  const { atoms, selectedStructure } = useCrystalStore();
  const schemaLD = {
    '@context': 'https://schema.org',
    '@type': 'TechArticle',
    headline: 'Crystal Structure Visualizer',
    description: 'Interactive 3D crystallography with lattice parameters, Miller planes, and symmetry operations.',
    inLanguage: 'en',
    author: { '@type': 'Organization', name: 'SimCore' },
    publisher: { '@type': 'Organization', name: 'SimCore' },
    keywords: 'crystallography, crystal lattice, Miller indices, symmetry operations, reciprocal lattice',
    about: ['Crystal systems', 'Lattice vectors', 'Miller planes', 'Reciprocal lattice'],
    mainEntityOfPage: typeof window !== 'undefined' ? window.location.href : ''
  } as const;

  const exportData = () => {
    // Export logic here
    console.log('Export functionality to be implemented');
  };

  const resetView = () => {
    // Reset logic here
    console.log('Reset functionality to be implemented');
  };

  return (
    <PhysicsModuleLayout>
      <script type="application/ld+json" dangerouslySetInnerHTML={{ __html: JSON.stringify(schemaLD) }} />
      <PhysicsModuleHeader
        title="Crystal Structure Visualizer"
        description="Interactive 3D crystallography and lattice analysis. Explore crystal systems, Miller planes, and symmetry operations with real-time rendering."
        category="Crystallography"
        difficulty="Intermediate"
        equation="\\vec{R} = n_1\\vec{a}_1 + n_2\\vec{a}_2 + n_3\\vec{a}_3"
        onExport={exportData}
        onReset={resetView}
        className="mb-6"
      />

      {/* Main Content */}
      <div className="w-full">
        <Tabs defaultValue="visualization" className="space-y-6">
          <TabsList className="grid w-full grid-cols-1 sm:grid-cols-3 bg-surface/80 backdrop-blur-sm border border-muted/30">
            <TabsTrigger 
              value="visualization" 
              className="flex items-center gap-2 text-sm px-4 py-2 data-[state=active]:bg-interactive data-[state=active]:text-surface transition-smooth"
            >
              <Eye className="h-4 w-4" />
              <span className="hidden sm:inline">Visualization</span>
              <span className="sm:hidden">3D View</span>
            </TabsTrigger>
            <TabsTrigger 
              value="controls" 
              className="flex items-center gap-2 text-sm px-4 py-2 data-[state=active]:bg-interactive data-[state=active]:text-surface transition-smooth"
            >
              <Settings className="h-4 w-4" />
              Controls
            </TabsTrigger>
            <TabsTrigger 
              value="theory" 
              className="flex items-center gap-2 text-sm px-4 py-2 data-[state=active]:bg-interactive data-[state=active]:text-surface transition-smooth"
            >
              <BookOpen className="h-4 w-4" />
              Theory
            </TabsTrigger>
          </TabsList>

          <TabsContent value="visualization" className="space-y-6">
            <div className="grid lg:grid-cols-4 gap-6">
              {/* Left Panel - Quick Controls */}
              <div className="lg:col-span-1">
                <ControlPanels />
              </div>

              {/* Right Panel - 3D Visualization */}
              <div className="lg:col-span-3">
                <Card className="h-[800px]">
                  <Suspense 
                    fallback={
                      <div className="h-full flex items-center justify-center">
                        <div className="text-center">
                          <Atom className="h-12 w-12 animate-spin mx-auto mb-4 text-primary" />
                          <p className="text-muted-foreground">Loading crystal structure...</p>
                        </div>
                      </div>
                    }
                  >
                    <CrystalVisualization />
                  </Suspense>
                </Card>
              </div>
            </div>

            {/* Bottom Panel - Structure Information */}
            <Card className="p-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold">Structure Information</h3>
                <Badge variant="secondary">{selectedStructure}</Badge>
              </div>
              
              <div className="grid md:grid-cols-3 gap-6">
                <div>
                  <h4 className="font-medium mb-2">Crystal System</h4>
                  <p className="text-sm text-muted-foreground">
                    Interactive controls allow real-time manipulation of lattice parameters,
                    supercell dimensions, and visualization options.
                  </p>
                </div>
                
                <div>
                  <h4 className="font-medium mb-2">Features</h4>
                  <ul className="text-sm text-muted-foreground space-y-1">
                    <li>• Real-time 3D rendering</li>
                    <li>• Miller plane visualization</li>
                    <li>• Reciprocal lattice display</li>
                    <li>• Symmetry operations</li>
                    <li>• CIF/XYZ export</li>
                  </ul>
                </div>
                
                <div>
                  <h4 className="font-medium mb-2">Performance</h4>
                  <p className="text-sm text-muted-foreground">
                    Level-of-detail (LOD) rendering and efficient instancing provide
                    smooth interaction even with large supercells.
                  </p>
                </div>
              </div>
            </Card>
          </TabsContent>

          <TabsContent value="controls" className="space-y-6">
            <div className="grid lg:grid-cols-2 gap-6">
              <div>
                <ControlPanels />
              </div>
              
              <div>
                <Card className="p-6">
                  <h3 className="text-lg font-semibold mb-4">Keyboard Shortcuts</h3>
                  <div className="space-y-3 text-sm">
                    <div className="flex justify-between">
                      <span className="font-mono bg-muted px-2 py-1 rounded">R</span>
                      <span>Reset view</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="font-mono bg-muted px-2 py-1 rounded">A</span>
                      <span>Toggle atoms</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="font-mono bg-muted px-2 py-1 rounded">B</span>
                      <span>Toggle bonds</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="font-mono bg-muted px-2 py-1 rounded">U</span>
                      <span>Toggle unit cell</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="font-mono bg-muted px-2 py-1 rounded">X</span>
                      <span>Toggle axes</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="font-mono bg-muted px-2 py-1 rounded">+/-</span>
                      <span>Adjust atom scale</span>
                    </div>
                  </div>
                </Card>

                <Card className="p-6 mt-6">
                  <h3 className="text-lg font-semibold mb-4">Mouse Controls</h3>
                  <div className="space-y-3 text-sm">
                    <div className="flex justify-between">
                      <span className="font-medium">Left click + drag</span>
                      <span>Rotate view</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="font-medium">Right click + drag</span>
                      <span>Pan view</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="font-medium">Scroll wheel</span>
                      <span>Zoom in/out</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="font-medium">Double click atom</span>
                      <span>Center view</span>
                    </div>
                  </div>
                </Card>
              </div>
            </div>
          </TabsContent>

          <TabsContent value="theory" className="space-y-6">
            <TheoryPanel />
          </TabsContent>
        </Tabs>
      </div>
    </PhysicsModuleLayout>
  );
}