import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { PhysicsModuleLayout } from '@/components/PhysicsModuleLayout';
import { PhysicsModuleHeader } from '@/components/PhysicsModuleHeader';
import { ArrowLeft, Download, Activity, BookOpen, Settings, TrendingUp } from 'lucide-react';
import { useToast } from '@/hooks/use-toast';
import { useSEO } from '@/hooks/use-seo';

// Import new modular components
import { useIsingStore } from '@/components/ising/IsingSimulationStore';
import { IsingSimulator } from '@/components/ising/IsingSimulator';
import IsingVisualization from '@/components/ising/IsingVisualization';
import IsingControlPanel from '@/components/ising/IsingControlPanel';
import IsingPlots from '@/components/ising/IsingPlots';
import IsingTheoryPanel from '@/components/ising/IsingTheoryPanel';

export default function IsingModel() {
  const navigate = useNavigate();
  const { toast } = useToast();
  useSEO({ title: 'Ising Model – SimCore', description: 'Interactive 2D Ising Monte Carlo with phase transitions and domain analysis.' });
  const schemaLD = useMemo(() => ({
    '@context': 'https://schema.org',
    '@type': 'TechArticle',
    headline: '2D Ising Model Monte Carlo',
    description: 'Interactive 2D Ising Monte Carlo with phase transitions and domain analysis.',
    inLanguage: 'en',
    author: { '@type': 'Organization', name: 'SimCore' },
    publisher: { '@type': 'Organization', name: 'SimCore' },
    keywords: 'Ising model, Monte Carlo, phase transition, magnetization, critical phenomena',
    about: ['Ising model', 'Phase transition', 'Monte Carlo'],
    mainEntityOfPage: typeof window !== 'undefined' ? window.location.href : ''
  }), []);
  const breadcrumbsLD = useMemo(() => ({
    '@context': 'https://schema.org',
    '@type': 'BreadcrumbList',
    itemListElement: [
      { '@type': 'ListItem', position: 1, name: 'Home', item: typeof window !== 'undefined' ? window.location.origin : '/' },
      { '@type': 'ListItem', position: 2, name: 'Modules', item: typeof window !== 'undefined' ? `${window.location.origin}/#modules-section` : '/#modules-section' },
      { '@type': 'ListItem', position: 3, name: '2D Ising Model Monte Carlo', item: typeof window !== 'undefined' ? window.location.href : '' }
    ]
  }), []);
  // Get state and actions from store
  const {
    size, temperature, magneticField, algorithm, isRunning, animationSpeed,
    toggleSimulation, resetSimulation, stepSimulation, 
    updateObservables, updateMagnetizationHistogram,
    setPhaseTransitionData, setCorrelationData, setDomainData,
    clearHistory, exportData, step
  } = useIsingStore();

  // Simulation state
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [currentMagnetization, setCurrentMagnetization] = useState(0);
  const [currentEnergy, setCurrentEnergy] = useState(0);

  // Create simulator instance
  const simulator = useMemo(() => 
    new IsingSimulator(size, temperature, magneticField, algorithm),
    [size, temperature, magneticField, algorithm]
  );

  // Update simulator parameters when store changes
  useEffect(() => {
    simulator.updateParameters(temperature, magneticField, algorithm);
  }, [temperature, magneticField, algorithm, simulator]);

  // Simulation step function
  const performStep = useCallback(() => {
    simulator.step();
    stepSimulation();
    
    // Calculate observables
    const mag = simulator.calculateMagnetization();
    const energy = simulator.calculateEnergy();
    
    setCurrentMagnetization(mag);
    setCurrentEnergy(energy);
    
    // Update store with observables (simplified - proper heat capacity/susceptibility would need sliding window)
    updateObservables(mag, energy, 0, 0);
    
    // Update analysis data periodically
    if (step % 10 === 0) {
      const correlation = simulator.calculateCorrelationFunction();
      const correlationLength = simulator.calculateCorrelationLength();
      setCorrelationData({
        ...correlation,
        correlationLength
      });
      
      const domains = simulator.findDomains();
      const averageDomainSize = domains.domainSizes.length > 0 
        ? domains.domainSizes.reduce((a, b) => a + b, 0) / domains.domainSizes.length 
        : 0;
      setDomainData({
        ...domains,
        averageDomainSize
      });
    }
  }, [simulator, stepSimulation, updateObservables, setCorrelationData, setDomainData, step]);

  // Animation loop
  useEffect(() => {
    if (!isRunning) return;
    
    const interval = setInterval(performStep, animationSpeed);
    return () => clearInterval(interval);
  }, [isRunning, performStep, animationSpeed]);

  // Reset function
  const handleReset = useCallback(() => {
    simulator.reset();
    resetSimulation();
    setCurrentMagnetization(0);
    setCurrentEnergy(0);
    toast({
      title: "Simulation Reset",
      description: "Lattice reinitialized with random configuration"
    });
  }, [simulator, resetSimulation, toast]);

  // Phase transition analysis
  const runPhaseTransitionAnalysis = useCallback(async () => {
    setIsAnalyzing(true);
    toast({
      title: "Running Phase Transition Analysis",
      description: "Sweeping temperature range..."
    });
    
    const temperatures = Array.from({ length: 30 }, (_, i) => 1.0 + i * 0.15);
    const results = {
      temperatures,
      magnetizations: [] as number[],
      heatCapacities: [] as number[],
      susceptibilities: [] as number[]
    };
    
    for (const T of temperatures) {
      simulator.updateParameters(T, 0);
      simulator.reset();
      
      // Equilibration
      for (let i = 0; i < 200; i++) {
        simulator.step();
      }
      
      // Measurement
      let magSum = 0, magSqSum = 0, energySum = 0, energySqSum = 0;
      const numMeasurements = 100;
      
      for (let i = 0; i < numMeasurements; i++) {
        simulator.step();
        const mag = Math.abs(simulator.calculateMagnetization());
        const energy = simulator.calculateEnergy();
        
        magSum += mag;
        magSqSum += mag * mag;
        energySum += energy;
        energySqSum += energy * energy;
      }
      
      const avgMag = magSum / numMeasurements;
      const avgMagSq = magSqSum / numMeasurements;
      const avgEnergy = energySum / numMeasurements;
      const avgEnergySq = energySqSum / numMeasurements;
      
      const heatCapacity = (avgEnergySq - avgEnergy * avgEnergy) / (T * T);
      const susceptibility = (avgMagSq - avgMag * avgMag) / T;
      
      results.magnetizations.push(avgMag);
      results.heatCapacities.push(heatCapacity);
      results.susceptibilities.push(susceptibility);
    }
    
    // Restore original parameters
    simulator.updateParameters(temperature, magneticField);
    setPhaseTransitionData(results);
    setIsAnalyzing(false);
    
    toast({
      title: "Analysis Complete",
      description: "Phase transition data ready for visualization"
    });
  }, [simulator, temperature, magneticField, setPhaseTransitionData, toast]);

  // Export data function
  const handleExportData = useCallback(() => {
    const dataStr = exportData();
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `ising_model_data_${new Date().toISOString().slice(0,10)}.json`;
    link.click();
    URL.revokeObjectURL(url);
    
    toast({
      title: "Data Exported",
      description: "Simulation data saved to file"
    });
  }, [exportData, toast]);

  // Set configuration function
  const handleSetConfiguration = useCallback((config: 'all_up' | 'all_down' | 'random' | 'checkerboard') => {
    simulator.setConfiguration(config);
    setCurrentMagnetization(simulator.calculateMagnetization());
    setCurrentEnergy(simulator.calculateEnergy());
    clearHistory();
  }, [simulator, clearHistory]);

  return (
    <PhysicsModuleLayout background="statistical">
      <script type="application/ld+json" dangerouslySetInnerHTML={{ __html: JSON.stringify(schemaLD) }} />
      
      <PhysicsModuleHeader
          title="2D Ising Model Monte Carlo Engine"
          description="Research-grade Monte Carlo simulation with critical phenomena analysis. Explore ferromagnetic phase transitions, domain formation, and universal scaling behavior."
          category="Spin & Magnetism"
          difficulty="Research"
          equation="H = -J \sum_{\langle i,j \rangle} s_i s_j - h \sum_i s_i"
          onExport={handleExportData}
          onReset={handleReset}
          isRunning={isRunning}
          currentStep={step}
          className="mb-8"
        />

        <Tabs defaultValue="simulation" className="w-full">
          <TabsList className="grid w-full grid-cols-2 sm:grid-cols-4 mb-6 sm:mb-8 bg-surface/80 backdrop-blur-sm border border-muted/30">
            <TabsTrigger 
              value="simulation" 
              className="flex items-center gap-1 sm:gap-2 text-xs sm:text-sm px-2 sm:px-4 py-2 data-[state=active]:bg-interactive data-[state=active]:text-surface transition-smooth"
            >
              <Activity className="w-3 h-3 sm:w-4 sm:h-4" />
              <span className="hidden sm:inline">Simulation</span>
              <span className="sm:hidden">Sim</span>
            </TabsTrigger>
            <TabsTrigger 
              value="analysis" 
              className="flex items-center gap-1 sm:gap-2 text-xs sm:text-sm px-2 sm:px-4 py-2 data-[state=active]:bg-interactive data-[state=active]:text-surface transition-smooth"
            >
              <TrendingUp className="w-3 h-3 sm:w-4 sm:h-4" />
              <span className="hidden sm:inline">Analysis</span>
              <span className="sm:hidden">Data</span>
            </TabsTrigger>
            <TabsTrigger 
              value="theory" 
              className="flex items-center gap-1 sm:gap-2 text-xs sm:text-sm px-2 sm:px-4 py-2 data-[state=active]:bg-interactive data-[state=active]:text-surface transition-smooth"
            >
              <BookOpen className="w-3 h-3 sm:w-4 sm:h-4" />
              <span className="hidden sm:inline">Theory</span>
              <span className="sm:hidden">Info</span>
            </TabsTrigger>
            <TabsTrigger 
              value="algorithms" 
              className="flex items-center gap-1 sm:gap-2 text-xs sm:text-sm px-2 sm:px-4 py-2 data-[state=active]:bg-interactive data-[state=active]:text-surface transition-smooth"
            >
              <Settings className="w-3 h-3 sm:w-4 sm:h-4" />
              <span className="hidden sm:inline">Methods</span>
              <span className="sm:hidden">Opts</span>
            </TabsTrigger>
          </TabsList>

          {/* Main Simulation */}
          <TabsContent value="simulation" className="space-y-4 sm:space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-4 gap-4 sm:gap-6">
              {/* Control Panel - Mobile First */}
              <div className="lg:col-span-1 order-1 lg:order-first">
                <div className="sticky top-4">
                  <IsingControlPanel
                    onRunPhaseTransition={runPhaseTransitionAnalysis}
                    onExportData={handleExportData}
                    onSetConfiguration={handleSetConfiguration}
                    currentMagnetization={currentMagnetization}
                    currentEnergy={currentEnergy}
                    isAnalyzing={isAnalyzing}
                  />
                </div>
              </div>

              {/* Visualization and Plots - Mobile Responsive */}
              <div className="lg:col-span-3 space-y-4 sm:space-y-6">
                {/* 3D Visualization */}
                <Card className="overflow-hidden bg-surface/90 backdrop-blur-sm border-muted/30">
                  <CardHeader className="pb-4">
                    <CardTitle className="text-textPrimary text-lg sm:text-xl">Spin Configuration</CardTitle>
                    <CardDescription className="text-textSecondary text-sm">
                      Interactive 3D lattice visualization with domain analysis
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="p-2 sm:p-6">
                    <div className="rounded-lg overflow-hidden">
                      <IsingVisualization
                        lattice={simulator.getLattice()}
                        size={size}
                        showDomains={useIsingStore.getState().showDomains}
                        height="400px"
                      />
                    </div>
                  </CardContent>
                </Card>

                {/* Real-time Plots - Responsive Height */}
                <div className="w-full">
                  <IsingPlots height={300} />
                </div>
              </div>
            </div>
          </TabsContent>

          {/* Analysis Tab */}
          <TabsContent value="analysis" className="space-y-6">
            <IsingPlots height={400} />
          </TabsContent>

          {/* Theory Tab */}
          <TabsContent value="theory" className="space-y-6">
            <IsingTheoryPanel />
          </TabsContent>

          {/* Algorithms Tab */}
          <TabsContent value="algorithms" className="space-y-6">
            <IsingTheoryPanel />
          </TabsContent>
        </Tabs>
      </PhysicsModuleLayout>
  );
}