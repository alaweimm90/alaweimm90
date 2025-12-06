import React from 'react';
import { PhysicsModuleHeader } from '@/components/PhysicsModuleHeader';
import { PhysicsModuleLayout, PhysicsContentCard, PhysicsTag, PhysicsTabSystem } from '@/components/PhysicsModuleLayout';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { ArrowLeft, Play, Pause, RotateCcw, Download, Settings, BookOpen, BarChart3, Zap } from 'lucide-react';
import { useNavigate } from 'react-router-dom';

// Example usage of the new styling system
export default function ExamplePhysicsModule() {
  const navigate = useNavigate();
  const [isRunning, setIsRunning] = React.useState(false);
  const [currentStep, setCurrentStep] = React.useState(1234);

  const handleExport = () => {
    console.log('Exporting data...');
  };

  const handleReset = () => {
    setCurrentStep(0);
    setIsRunning(false);
  };

  const tabs = [
    {
      id: 'simulation',
      label: 'Simulation',
      icon: <Zap className="w-4 h-4" />,
      content: (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Parameter Controls */}
          <PhysicsContentCard
            title="Quantum Parameters"
            description="Adjust the fundamental parameters of the quantum system"
            variant="elegant"
            glow
          >
            <div className="space-y-4">
              <div className="space-y-2">
                <label className="text-sm font-medium">Energy Level (eV)</label>
                <input
                  type="range"
                  min="0"
                  max="10"
                  step="0.1"
                  className="w-full h-2 bg-muted rounded-lg appearance-none cursor-pointer slider-thumb"
                />
              </div>
              <div className="space-y-2">
                <label className="text-sm font-medium">Temperature (K)</label>
                <input
                  type="range"
                  min="0"
                  max="1000"
                  step="10"
                  className="w-full h-2 bg-muted rounded-lg appearance-none cursor-pointer slider-thumb"
                />
              </div>
              <div className="flex gap-2">
                <PhysicsTag variant="success" glow animated>Active</PhysicsTag>
                <PhysicsTag variant="info" animated>Quantum</PhysicsTag>
              </div>
            </div>
          </PhysicsContentCard>

          {/* Visualization */}
          <div className="lg:col-span-2">
            <PhysicsContentCard
              title="Wave Function Evolution"
              description="Real-time visualization of quantum state evolution"
              variant="glass"
              className="h-96"
            >
              {/* Placeholder for physics visualization */}
              <div className="h-full bg-gradient-to-br from-primary/5 to-accent/5 rounded-xl flex items-center justify-center border-2 border-dashed border-border/50">
                <div className="text-center space-y-2">
                  <div className="w-16 h-16 bg-primary/20 rounded-full flex items-center justify-center mx-auto animate-quantum-pulse">
                    <Zap className="w-8 h-8 text-primary" />
                  </div>
                  <p className="text-muted-foreground">Physics Visualization</p>
                </div>
              </div>
            </PhysicsContentCard>
          </div>
        </div>
      )
    },
    {
      id: 'theory',
      label: 'Theory',
      icon: <BookOpen className="w-4 h-4" />,
      content: (
        <PhysicsContentCard
          title="Theoretical Background"
          description="Mathematical foundations and physical principles"
          variant="elegant"
        >
          <div className="space-y-6">
            <div className="p-6 bg-background/50 rounded-xl border border-border/50">
              <h3 className="text-lg font-semibold mb-3">Schrödinger Equation</h3>
              <div className="text-center py-4">
                <code className="text-lg">iℏ ∂ψ/∂t = Ĥψ</code>
              </div>
              <p className="text-muted-foreground">
                The time-dependent Schrödinger equation governs the quantum evolution of the system,
                where ψ represents the wave function and Ĥ is the Hamiltonian operator.
              </p>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="p-4 bg-primary/5 rounded-lg border border-primary/20">
                <h4 className="font-medium text-primary mb-2">Key Concepts</h4>
                <ul className="text-sm space-y-1 text-muted-foreground">
                  <li>• Wave-particle duality</li>
                  <li>• Quantum superposition</li>
                  <li>• Measurement collapse</li>
                  <li>• Uncertainty principle</li>
                </ul>
              </div>
              
              <div className="p-4 bg-accent/5 rounded-lg border border-accent/20">
                <h4 className="font-medium text-accent mb-2">Applications</h4>
                <ul className="text-sm space-y-1 text-muted-foreground">
                  <li>• Quantum computing</li>
                  <li>• Molecular dynamics</li>
                  <li>• Electronic devices</li>
                  <li>• Material design</li>
                </ul>
              </div>
            </div>
          </div>
        </PhysicsContentCard>
      )
    },
    {
      id: 'analysis',
      label: 'Analysis',
      icon: <BarChart3 className="w-4 h-4" />,
      content: (
        <PhysicsContentCard
          title="Data Analysis"
          description="Computational results and statistical analysis"
          variant="minimal"
        >
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="space-y-4">
              <h3 className="text-lg font-semibold">Energy Spectrum</h3>
              <div className="h-48 bg-gradient-to-r from-blue-500/10 to-purple-500/10 rounded-xl border border-border/50 flex items-center justify-center">
                <span className="text-muted-foreground">Energy Plot Placeholder</span>
              </div>
            </div>
            
            <div className="space-y-4">
              <h3 className="text-lg font-semibold">Statistics</h3>
              <div className="space-y-3">
                <div className="flex justify-between items-center p-3 bg-background/50 rounded-lg">
                  <span className="text-sm font-medium">Average Energy:</span>
                  <PhysicsTag variant="info" size="sm">2.34 eV</PhysicsTag>
                </div>
                <div className="flex justify-between items-center p-3 bg-background/50 rounded-lg">
                  <span className="text-sm font-medium">Standard Deviation:</span>
                  <PhysicsTag variant="warning" size="sm">0.12 eV</PhysicsTag>
                </div>
                <div className="flex justify-between items-center p-3 bg-background/50 rounded-lg">
                  <span className="text-sm font-medium">Convergence:</span>
                  <PhysicsTag variant="success" size="sm">99.8%</PhysicsTag>
                </div>
              </div>
            </div>
          </div>
        </PhysicsContentCard>
      )
    }
  ];

  return (
    <PhysicsModuleLayout background="quantum">
      <PhysicsModuleHeader
        title="Quantum Time Evolution Solver"
        description="Advanced numerical solution of the time-dependent Schrödinger equation with real-time wave packet dynamics and quantum state visualization."
        category="Quantum Dynamics"
        difficulty="Advanced"
        equation="iℏ ∂ψ/∂t = Ĥψ"
        onExport={handleExport}
        onReset={handleReset}
        isRunning={isRunning}
        currentStep={currentStep}
        className="mb-8"
      />

      <PhysicsTabSystem
        tabs={tabs}
        defaultTab="simulation"
        className="space-y-6"
      />
    </PhysicsModuleLayout>
  );
}